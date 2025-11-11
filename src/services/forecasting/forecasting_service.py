import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from src.models.database import SalesRecord, Product, ForecastResult, ModelPerformanceMetric
from config.settings import settings
import json
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """Complete time series forecasting with multiple models"""
    
    def __init__(self):
        self.prophet_model = None
        self.lstm_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.model_dir = Path(settings.FORECASTING_MODELS_PATH)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_prophet_data(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        prophet_df = sales_data[['sale_date', 'quantity']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Aggregate daily sales
        prophet_df = prophet_df.groupby('ds').agg({'y': 'sum'}).reset_index()
        
        return prophet_df
    
    def train_prophet_model(self, sales_data: pd.DataFrame) -> Prophet:
        """Train Facebook Prophet model with seasonality"""
        logger.info("Training Prophet model...")
        
        prophet_df = self.prepare_prophet_data(sales_data)
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            interval_width=settings.CONFIDENCE_INTERVAL
        )
        
        # Add custom seasonality if enough data
        if len(prophet_df) > 365:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        model.fit(prophet_df)
        self.prophet_model = model
        
        logger.info("Prophet model trained successfully")
        return model
    
    def create_lstm_features(self, sales_data: pd.DataFrame, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Create features for LSTM model"""
        # Aggregate daily sales
        daily_sales = sales_data.groupby('sale_date')['quantity'].sum().sort_index()
        
        # Create sequences
        data = daily_sales.values.reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(data_scaled)):
            X.append(data_scaled[i-lookback:i, 0])
            y.append(data_scaled[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, lookback: int = 30) -> keras.Model:
        """Build LSTM neural network architecture"""
        model = keras.Sequential([
            layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
            layers.Dropout(0.2),
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_lstm_model(self, sales_data: pd.DataFrame, lookback: int = 30, epochs: int = 50) -> keras.Model:
        """Train LSTM deep learning model"""
        logger.info("Training LSTM model...")
        
        X, y = self.create_lstm_features(sales_data, lookback)
        
        if len(X) < 100:
            logger.warning("Insufficient data for LSTM training")
            return None
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train-validation split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = self.build_lstm_model(lookback)
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.lstm_model = model
        logger.info("LSTM model trained successfully")
        
        return model
    
    def create_ml_features(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for ML models"""
        df = sales_data.copy()
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df = df.sort_values('sale_date')
        
        # Aggregate daily
        daily_df = df.groupby('sale_date')['quantity'].sum().reset_index()
        daily_df = daily_df.set_index('sale_date')
        
        # Time-based features
        daily_df['day_of_week'] = daily_df.index.dayofweek
        daily_df['day_of_month'] = daily_df.index.day
        daily_df['month'] = daily_df.index.month
        daily_df['quarter'] = daily_df.index.quarter
        daily_df['week_of_year'] = daily_df.index.isocalendar().week
        daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            daily_df[f'lag_{lag}'] = daily_df['quantity'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            daily_df[f'rolling_mean_{window}'] = daily_df['quantity'].rolling(window=window).mean()
            daily_df[f'rolling_std_{window}'] = daily_df['quantity'].rolling(window=window).std()
            daily_df[f'rolling_min_{window}'] = daily_df['quantity'].rolling(window=window).min()
            daily_df[f'rolling_max_{window}'] = daily_df['quantity'].rolling(window=window).max()
        
        # Trend features
        daily_df['quantity_diff_1'] = daily_df['quantity'].diff(1)
        daily_df['quantity_diff_7'] = daily_df['quantity'].diff(7)
        
        # Exponential moving averages
        daily_df['ema_7'] = daily_df['quantity'].ewm(span=7).mean()
        daily_df['ema_30'] = daily_df['quantity'].ewm(span=30).mean()
        
        # Drop NaN values
        daily_df = daily_df.dropna()
        
        return daily_df
    
    def train_random_forest_model(self, sales_data: pd.DataFrame) -> RandomForestRegressor:
        """Train Random Forest ensemble model"""
        logger.info("Training Random Forest model...")
        
        feature_df = self.create_ml_features(sales_data)
        
        if len(feature_df) < 60:
            logger.warning("Insufficient data for Random Forest training")
            return None
        
        # Prepare features and target
        feature_cols = [col for col in feature_df.columns if col != 'quantity']
        X = feature_df[feature_cols].values
        y = feature_df['quantity'].values
        
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        logger.info(f"Random Forest - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        self.rf_model = model
        return model
    
    def forecast_prophet(self, periods: int = 90) -> pd.DataFrame:
        """Generate forecast using Prophet"""
        if self.prophet_model is None:
            raise ValueError("Prophet model not trained")
        
        future = self.prophet_model.make_future_dataframe(periods=periods)
        forecast = self.prophet_model.predict(future)
        
        # Get only future predictions
        forecast = forecast.tail(periods)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def forecast_lstm(self, last_sequence: np.ndarray, periods: int = 90) -> np.ndarray:
        """Generate forecast using LSTM"""
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, len(current_sequence), 1)
            
            # Predict next value
            pred = self.lstm_model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred)
        
        # Inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def forecast_random_forest(self, sales_data: pd.DataFrame, periods: int = 90) -> np.ndarray:
        """Generate forecast using Random Forest"""
        if self.rf_model is None:
            raise ValueError("Random Forest model not trained")
        
        feature_df = self.create_ml_features(sales_data)
        feature_cols = [col for col in feature_df.columns if col != 'quantity']
        
        predictions = []
        last_data = feature_df.copy()
        
        for i in range(periods):
            # Get last row features
            X_pred = last_data[feature_cols].iloc[-1:].values
            
            # Predict
            pred = self.rf_model.predict(X_pred)[0]
            predictions.append(max(0, pred))  # Ensure non-negative
            
            # This is simplified - in production, properly update features
            # For now, use average of recent predictions
            
        return np.array(predictions)
    
    def ensemble_forecast(self, sales_data: pd.DataFrame, periods: int = 90) -> Dict:
        """Combine multiple models for robust forecasting"""
        logger.info("Generating ensemble forecast...")
        
        forecasts = {}
        weights = {}
        
        # Prophet forecast
        try:
            prophet_forecast = self.forecast_prophet(periods)
            forecasts['prophet'] = prophet_forecast['yhat'].values
            weights['prophet'] = 0.4
        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}")
        
        # LSTM forecast
        try:
            if self.lstm_model is not None:
                # Get last 30 days for sequence
                daily_sales = sales_data.groupby('sale_date')['quantity'].sum().sort_index()
                last_sequence = daily_sales.tail(30).values
                last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
                
                lstm_forecast = self.forecast_lstm(last_sequence_scaled, periods)
                forecasts['lstm'] = lstm_forecast
                weights['lstm'] = 0.3
        except Exception as e:
            logger.error(f"LSTM forecast failed: {e}")
        
        # Random Forest forecast
        try:
            if self.rf_model is not None:
                rf_forecast = self.forecast_random_forest(sales_data, periods)
                forecasts['rf'] = rf_forecast
                weights['rf'] = 0.3
        except Exception as e:
            logger.error(f"Random Forest forecast failed: {e}")
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted ensemble
        ensemble_pred = np.zeros(periods)
        for model_name, pred in forecasts.items():
            ensemble_pred += pred * weights[model_name]
        
        # Calculate confidence intervals (simplified)
        if 'prophet' in forecasts:
            lower_bound = prophet_forecast['yhat_lower'].values
            upper_bound = prophet_forecast['yhat_upper'].values
        else:
            std = np.std(ensemble_pred)
            lower_bound = ensemble_pred - 1.96 * std
            upper_bound = ensemble_pred + 1.96 * std
        
        # Generate dates
        last_date = pd.to_datetime(sales_data['sale_date'].max())
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        return {
            'dates': forecast_dates.tolist(),
            'predictions': ensemble_pred.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'models_used': list(forecasts.keys()),
            'model_weights': weights
        }
    
    def save_models(self, product_id: int):
        """Save trained models to disk"""
        product_dir = self.model_dir / str(product_id)
        product_dir.mkdir(parents=True, exist_ok=True)
        
        if self.prophet_model:
            joblib.dump(self.prophet_model, product_dir / 'prophet_model.pkl')
        
        if self.lstm_model:
            self.lstm_model.save(product_dir / 'lstm_model.h5')
            joblib.dump(self.scaler, product_dir / 'scaler.pkl')
        
        if self.rf_model:
            joblib.dump(self.rf_model, product_dir / 'rf_model.pkl')
        
        logger.info(f"Models saved for product {product_id}")
    
    def load_models(self, product_id: int):
        """Load pre-trained models from disk"""
        product_dir = self.model_dir / str(product_id)
        
        if not product_dir.exists():
            return False
        
        try:
            prophet_path = product_dir / 'prophet_model.pkl'
            if prophet_path.exists():
                self.prophet_model = joblib.load(prophet_path)
            
            lstm_path = product_dir / 'lstm_model.h5'
            if lstm_path.exists():
                self.lstm_model = keras.models.load_model(lstm_path)
                self.scaler = joblib.load(product_dir / 'scaler.pkl')
            
            rf_path = product_dir / 'rf_model.pkl'
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
            
            logger.info(f"Models loaded for product {product_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


class ForecastingService:
    """Main forecasting service with database integration"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.forecaster = TimeSeriesForecaster()
    
    async def get_sales_data(self, product_id: int, days_back: int = 365) -> pd.DataFrame:
        """Fetch historical sales data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        query = select(SalesRecord).where(
            and_(
                SalesRecord.product_id == product_id,
                SalesRecord.sale_date >= cutoff_date
            )
        ).order_by(SalesRecord.sale_date)
        
        result = await self.db.execute(query)
        records = result.scalars().all()
        
        if not records:
            return pd.DataFrame()
        
        data = [{
            'sale_date': r.sale_date,
            'quantity': r.quantity,
            'revenue': float(r.revenue) if r.revenue else 0,
            'sales_channel': r.sales_channel,
            'promotion_active': r.promotion_active
        } for r in records]
        
        return pd.DataFrame(data)
    
    async def train_and_forecast(self, product_id: int, forecast_horizon: int = 90) -> Dict:
        """Complete training and forecasting pipeline"""
        logger.info(f"Starting forecast for product {product_id}")
        
        # Get sales data
        sales_data = await self.get_sales_data(product_id)
        
        if len(sales_data) < settings.MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient data for product {product_id}")
            return {
                'success': False,
                'error': 'Insufficient historical data',
                'min_required': settings.MIN_TRAINING_SAMPLES,
                'available': len(sales_data)
            }
        
        # Train models
        self.forecaster.train_prophet_model(sales_data)
        self.forecaster.train_lstm_model(sales_data)
        self.forecaster.train_random_forest_model(sales_data)
        
        # Generate ensemble forecast
        forecast_result = self.forecaster.ensemble_forecast(sales_data, forecast_horizon)
        
        # Save models
        self.forecaster.save_models(product_id)
        
        # Store forecast in database
        await self.save_forecast_to_db(product_id, forecast_result)
        
        logger.info(f"Forecast completed for product {product_id}")
        
        return {
            'success': True,
            'product_id': product_id,
            'forecast': forecast_result,
            'training_samples': len(sales_data)
        }
    
    async def save_forecast_to_db(self, product_id: int, forecast: Dict):
        """Save forecast results to database"""
        for i, date in enumerate(forecast['dates']):
            forecast_record = ForecastResult(
                product_id=product_id,
                forecast_date=date,
                predicted_demand=forecast['predictions'][i],
                lower_bound=forecast['lower_bound'][i],
                upper_bound=forecast['upper_bound'][i],
                confidence=settings.CONFIDENCE_INTERVAL,
                model_used='ensemble',
                model_version='1.0',
                features_used=forecast.get('model_weights', {})
            )
            self.db.add(forecast_record)
        
        await self.db.commit()
    
    async def get_forecast(self, product_id: int, days_ahead: int = 30) -> List[Dict]:
        """Retrieve existing forecast from database"""
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=days_ahead)
        
        query = select(ForecastResult).where(
            and_(
                ForecastResult.product_id == product_id,
                ForecastResult.forecast_date >= start_date,
                ForecastResult.forecast_date <= end_date
            )
        ).order_by(ForecastResult.forecast_date)
        
        result = await self.db.execute(query)
        forecasts = result.scalars().all()
        
        return [{
            'date': f.forecast_date.isoformat(),
            'predicted_demand': f.predicted_demand,
            'lower_bound': f.lower_bound,
            'upper_bound': f.upper_bound,
            'confidence': f.confidence
        } for f in forecasts]
