"""
Multi-Horizon Sales Forecasting Service

Provides forecasting at different time granularities:
- Hourly: For real-time operations
- Daily: For operational planning
- Weekly: For tactical planning
- Monthly: For strategic planning
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
import logging
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.database import SalesRecord, Product
from src.models.sales_forecasting_models import (
    SalesForecastResult, ForecastAccuracyMetric
)

logger = logging.getLogger(__name__)


class MultiHorizonForecaster:
    """
    Advanced forecasting service supporting multiple time horizons
    """

    HORIZONS = {
        'hourly': {'periods': 168, 'freq': 'H'},  # 7 days
        'daily': {'periods': 90, 'freq': 'D'},    # 90 days
        'weekly': {'periods': 52, 'freq': 'W'},    # 52 weeks
        'monthly': {'periods': 12, 'freq': 'M'}    # 12 months
    }

    def __init__(self, db: AsyncSession):
        self.db = db
        self.prophet_models = {}
        self.lstm_models = {}
        self.rf_models = {}
        self.scalers = {}

    async def forecast_all_horizons(
        self,
        product_id: int,
        confidence_level: float = 0.95
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for all time horizons

        Returns:
            Dict with keys: 'hourly', 'daily', 'weekly', 'monthly'
        """
        logger.info(f"Generating multi-horizon forecast for product {product_id}")

        results = {}

        for horizon, config in self.HORIZONS.items():
            try:
                forecast = await self.forecast_single_horizon(
                    product_id=product_id,
                    horizon=horizon,
                    periods=config['periods'],
                    confidence_level=confidence_level
                )
                results[horizon] = forecast

                # Save to database
                await self._save_forecast_results(
                    product_id=product_id,
                    horizon=horizon,
                    forecast=forecast
                )

            except Exception as e:
                logger.error(f"Error forecasting {horizon} for product {product_id}: {e}")
                results[horizon] = pd.DataFrame()

        return results

    async def forecast_single_horizon(
        self,
        product_id: int,
        horizon: str,
        periods: Optional[int] = None,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Generate forecast for a specific time horizon

        Args:
            product_id: Product to forecast
            horizon: One of 'hourly', 'daily', 'weekly', 'monthly'
            periods: Number of periods to forecast (default from HORIZONS)
            confidence_level: Confidence interval (default 0.95)

        Returns:
            DataFrame with columns: date, predicted_sales, predicted_revenue, lower_bound, upper_bound
        """
        if horizon not in self.HORIZONS:
            raise ValueError(f"Invalid horizon: {horizon}. Must be one of {list(self.HORIZONS.keys())}")

        config = self.HORIZONS[horizon]
        periods = periods or config['periods']
        freq = config['freq']

        # Get historical data aggregated at the appropriate frequency
        sales_data = await self._get_aggregated_sales_data(product_id, freq)

        if len(sales_data) < 30:  # Minimum data requirement
            raise ValueError(f"Insufficient data for product {product_id}. Need at least 30 data points.")

        # Get product info for revenue calculation
        product = await self._get_product(product_id)

        # Generate ensemble forecast
        prophet_forecast = await self._forecast_prophet(sales_data, periods, freq)
        lstm_forecast = await self._forecast_lstm(sales_data, periods, freq)
        rf_forecast = await self._forecast_rf(sales_data, periods, freq)

        # Ensemble with adaptive weights
        weights = self._calculate_adaptive_weights(
            prophet_forecast, lstm_forecast, rf_forecast, sales_data
        )

        ensemble_forecast = (
            weights['prophet'] * prophet_forecast['yhat'] +
            weights['lstm'] * lstm_forecast['prediction'] +
            weights['rf'] * rf_forecast['prediction']
        )

        # Calculate confidence intervals
        ensemble_std = np.sqrt(
            weights['prophet'] * prophet_forecast['yhat_std']**2 +
            weights['lstm'] * lstm_forecast['std']**2 +
            weights['rf'] * rf_forecast['std']**2
        )

        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence_level) / 2)

        # Build result DataFrame
        result_df = pd.DataFrame({
            'date': prophet_forecast['ds'],
            'predicted_sales': np.maximum(0, ensemble_forecast),
            'predicted_revenue': np.maximum(0, ensemble_forecast) * product.unit_price,
            'lower_bound': np.maximum(0, ensemble_forecast - z_score * ensemble_std),
            'upper_bound': ensemble_forecast + z_score * ensemble_std,
            'confidence_level': confidence_level,
            'model_weights': [weights] * len(ensemble_forecast)
        })

        return result_df

    async def _get_aggregated_sales_data(
        self,
        product_id: int,
        freq: str
    ) -> pd.DataFrame:
        """
        Get sales data aggregated at specified frequency
        """
        query = select(SalesRecord).where(
            SalesRecord.product_id == product_id
        ).order_by(SalesRecord.sale_date)

        result = await self.db.execute(query)
        records = result.scalars().all()

        if not records:
            raise ValueError(f"No sales data found for product {product_id}")

        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': r.sale_date,
            'quantity': r.quantity,
            'revenue': r.quantity * r.unit_price
        } for r in records])

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Resample to desired frequency
        if freq == 'H':
            # For hourly, we need to distribute daily sales across hours
            # Using historical hour-of-day patterns if available
            df_hourly = self._distribute_to_hourly(df)
            return df_hourly
        else:
            # Aggregate to desired frequency
            agg_df = df.resample(freq).agg({
                'quantity': 'sum',
                'revenue': 'sum'
            }).reset_index()
            return agg_df

    def _distribute_to_hourly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Distribute daily sales to hourly using realistic patterns
        """
        # Define typical retail hour-of-day pattern
        # Higher sales during business hours (9 AM - 8 PM)
        hourly_pattern = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 12 AM - 6 AM
            0.02, 0.03, 0.04, 0.06, 0.08, 0.09,  # 6 AM - 12 PM
            0.09, 0.08, 0.07, 0.08, 0.09, 0.10,  # 12 PM - 6 PM
            0.08, 0.06, 0.04, 0.02, 0.01, 0.01   # 6 PM - 12 AM
        ])
        hourly_pattern = hourly_pattern / hourly_pattern.sum()

        hourly_data = []
        for date, row in daily_df.iterrows():
            for hour in range(24):
                hourly_date = pd.Timestamp(date) + pd.Timedelta(hours=hour)
                hourly_data.append({
                    'date': hourly_date,
                    'quantity': row['quantity'] * hourly_pattern[hour],
                    'revenue': row['revenue'] * hourly_pattern[hour]
                })

        return pd.DataFrame(hourly_data)

    async def _forecast_prophet(
        self,
        sales_data: pd.DataFrame,
        periods: int,
        freq: str
    ) -> pd.DataFrame:
        """
        Generate forecast using Prophet
        """
        # Prepare data for Prophet
        prophet_df = sales_data[['date', 'quantity']].copy()
        prophet_df.columns = ['ds', 'y']

        # Configure Prophet based on frequency
        if freq == 'H':
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95
            )
        elif freq == 'D':
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95
            )
        elif freq == 'W':
            model = Prophet(
                weekly_seasonality=False,
                yearly_seasonality=True,
                interval_width=0.95
            )
        else:  # Monthly
            model = Prophet(
                yearly_seasonality=True,
                interval_width=0.95
            )

        # Fit model
        model.fit(prophet_df)

        # Generate forecast
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # Get only future predictions
        forecast = forecast.tail(periods)

        # Calculate standard deviation from confidence intervals
        forecast['yhat_std'] = (forecast['yhat_upper'] - forecast['yhat_lower']) / 3.92

        return forecast[['ds', 'yhat', 'yhat_std', 'yhat_lower', 'yhat_upper']]

    async def _forecast_lstm(
        self,
        sales_data: pd.DataFrame,
        periods: int,
        freq: str
    ) -> pd.DataFrame:
        """
        Generate forecast using LSTM neural network
        """
        # Prepare data
        values = sales_data['quantity'].values

        # Scale data
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values.reshape(-1, 1))

        # Create sequences
        lookback = min(30, len(values) // 3)
        X, y = [], []
        for i in range(lookback, len(scaled_values)):
            X.append(scaled_values[i-lookback:i])
            y.append(scaled_values[i])

        if len(X) < 20:
            # Not enough data for LSTM, return simple moving average
            ma = np.mean(values[-30:])
            dates = pd.date_range(
                start=sales_data['date'].iloc[-1] + pd.Timedelta(1, freq),
                periods=periods,
                freq=freq
            )
            return pd.DataFrame({
                'ds': dates,
                'prediction': [ma] * periods,
                'std': [np.std(values[-30:])] * periods
            })

        X = np.array(X)
        y = np.array(y)

        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
            layers.Dropout(0.2),
            layers.LSTM(50, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(25, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.2)

        # Generate predictions
        predictions = []
        current_sequence = scaled_values[-lookback:].flatten()

        for _ in range(periods):
            X_pred = current_sequence.reshape(1, lookback, 1)
            pred = model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred)
            current_sequence = np.append(current_sequence[1:], pred)

        # Inverse transform
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        # Calculate uncertainty
        pred_std = np.std(values[-30:]) * 1.5  # Slightly higher uncertainty for LSTM

        dates = pd.date_range(
            start=sales_data['date'].iloc[-1] + pd.Timedelta(1, freq),
            periods=periods,
            freq=freq
        )

        return pd.DataFrame({
            'ds': dates,
            'prediction': predictions,
            'std': [pred_std] * periods
        })

    async def _forecast_rf(
        self,
        sales_data: pd.DataFrame,
        periods: int,
        freq: str
    ) -> pd.DataFrame:
        """
        Generate forecast using Random Forest with feature engineering
        """
        # Feature engineering
        df = sales_data.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Lag features
        for lag in [1, 7, 14, 30]:
            if lag < len(df):
                df[f'lag_{lag}'] = df['quantity'].shift(lag)

        # Rolling features
        for window in [7, 14, 30]:
            if window < len(df):
                df[f'rolling_mean_{window}'] = df['quantity'].rolling(window).mean()
                df[f'rolling_std_{window}'] = df['quantity'].rolling(window).std()

        # Drop NaN values
        df = df.dropna()

        if len(df) < 30:
            # Fallback to simple average
            ma = np.mean(sales_data['quantity'].values[-30:])
            dates = pd.date_range(
                start=sales_data['date'].iloc[-1] + pd.Timedelta(1, freq),
                periods=periods,
                freq=freq
            )
            return pd.DataFrame({
                'ds': dates,
                'prediction': [ma] * periods,
                'std': [np.std(sales_data['quantity'].values[-30:])] * periods
            })

        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['date', 'quantity', 'revenue']]
        X = df[feature_cols].values
        y = df['quantity'].values

        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X, y)

        # Generate future features and predictions
        predictions = []
        last_row = df.iloc[-1:].copy()

        for i in range(periods):
            # Update time features
            next_date = last_row['date'].iloc[0] + pd.Timedelta(1, freq)
            last_row['date'] = next_date
            last_row['day_of_week'] = next_date.dayofweek
            last_row['day_of_month'] = next_date.day
            last_row['month'] = next_date.month
            last_row['quarter'] = next_date.quarter
            last_row['week_of_year'] = next_date.isocalendar().week
            last_row['is_weekend'] = int(next_date.dayofweek >= 5)

            # Make prediction
            X_pred = last_row[feature_cols].values
            pred = rf_model.predict(X_pred)[0]
            predictions.append(max(0, pred))

            # Update lag features (simplified)
            last_row['lag_1'] = pred

        pred_std = np.std(y[-30:]) * 1.2

        dates = pd.date_range(
            start=sales_data['date'].iloc[-1] + pd.Timedelta(1, freq),
            periods=periods,
            freq=freq
        )

        return pd.DataFrame({
            'ds': dates,
            'prediction': predictions,
            'std': [pred_std] * periods
        })

    def _calculate_adaptive_weights(
        self,
        prophet_forecast: pd.DataFrame,
        lstm_forecast: pd.DataFrame,
        rf_forecast: pd.DataFrame,
        historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on recent model performance
        """
        # Default weights
        weights = {
            'prophet': 0.4,
            'lstm': 0.3,
            'rf': 0.3
        }

        # In production, calculate based on recent accuracy metrics
        # For now, return default weights
        return weights

    async def _save_forecast_results(
        self,
        product_id: int,
        horizon: str,
        forecast: pd.DataFrame
    ):
        """
        Save forecast results to database
        """
        for _, row in forecast.iterrows():
            result = SalesForecastResult(
                product_id=product_id,
                forecast_date=row['date'].date(),
                horizon=horizon,
                predicted_sales=float(row['predicted_sales']),
                predicted_revenue=float(row['predicted_revenue']),
                lower_bound=float(row['lower_bound']),
                upper_bound=float(row['upper_bound']),
                confidence_level=float(row['confidence_level']),
                model_used='ensemble',
                metadata={'weights': row['model_weights']}
            )
            self.db.add(result)

        await self.db.commit()
        logger.info(f"Saved {len(forecast)} {horizon} forecast results for product {product_id}")

    async def _get_product(self, product_id: int) -> Product:
        """Get product details"""
        query = select(Product).where(Product.id == product_id)
        result = await self.db.execute(query)
        product = result.scalar_one_or_none()

        if not product:
            raise ValueError(f"Product {product_id} not found")

        return product
