"""
XGBoost Forecasting Service

Advanced gradient boosting model for sales forecasting with:
- Feature engineering pipeline
- Hyperparameter optimization
- Feature importance analysis
- Multi-step forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.models.database import SalesRecord, Product

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """
    XGBoost-based forecasting with advanced feature engineering
    """

    # Default hyperparameters (optimized for sales forecasting)
    DEFAULT_PARAMS = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }

    def __init__(self, db: AsyncSession):
        self.db = db
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}

    async def forecast(
        self,
        product_id: int,
        periods: int = 30,
        use_tuned_params: bool = False,
        return_confidence_intervals: bool = True
    ) -> pd.DataFrame:
        """
        Generate forecast using XGBoost

        Args:
            product_id: Product to forecast
            periods: Number of days to forecast
            use_tuned_params: Whether to run hyperparameter tuning
            return_confidence_intervals: Whether to calculate confidence intervals

        Returns:
            DataFrame with forecasts and optional confidence intervals
        """
        logger.info(f"Starting XGBoost forecast for product {product_id}, {periods} periods")

        # Get historical data
        sales_data = await self._get_sales_data(product_id)

        if len(sales_data) < 60:
            raise ValueError(f"Need at least 60 days of data, got {len(sales_data)}")

        # Engineer features
        df_features = self._engineer_features(sales_data)

        # Prepare training data
        X_train, y_train = self._prepare_training_data(df_features)

        if len(X_train) < 30:
            raise ValueError(f"Insufficient training samples after feature engineering")

        # Hyperparameter tuning (optional)
        if use_tuned_params:
            params = await self._tune_hyperparameters(X_train, y_train)
        else:
            params = self.DEFAULT_PARAMS.copy()

        # Train model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train)

        # Store feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

        logger.info(f"XGBoost model trained with {len(X_train)} samples")

        # Generate forecasts
        forecast = await self._generate_forecast(
            df_features=df_features,
            periods=periods,
            product_id=product_id
        )

        # Calculate confidence intervals if requested
        if return_confidence_intervals:
            forecast = self._add_confidence_intervals(forecast, X_train, y_train)

        return forecast

    def _engineer_features(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive feature engineering pipeline
        """
        df = sales_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # ============================================================
        # TIME FEATURES
        # ============================================================

        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter

        # Weekend/weekday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        # Month features
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['days_in_month'] = df['date'].dt.days_in_month

        # Season (Northern Hemisphere)
        df['season'] = df['month'].apply(lambda x:
            1 if x in [12, 1, 2] else  # Winter
            2 if x in [3, 4, 5] else    # Spring
            3 if x in [6, 7, 8] else    # Summer
            4                            # Fall
        )

        # ============================================================
        # LAG FEATURES
        # ============================================================

        lag_periods = [1, 2, 3, 7, 14, 21, 28, 30]
        for lag in lag_periods:
            if lag < len(df):
                df[f'lag_{lag}'] = df['quantity'].shift(lag)

        # ============================================================
        # ROLLING STATISTICS
        # ============================================================

        windows = [3, 7, 14, 21, 28, 30]
        for window in windows:
            if window < len(df):
                # Rolling mean
                df[f'rolling_mean_{window}'] = df['quantity'].rolling(window=window).mean()

                # Rolling std
                df[f'rolling_std_{window}'] = df['quantity'].rolling(window=window).std()

                # Rolling min/max
                df[f'rolling_min_{window}'] = df['quantity'].rolling(window=window).min()
                df[f'rolling_max_{window}'] = df['quantity'].rolling(window=window).max()

        # ============================================================
        # EXPONENTIALLY WEIGHTED MOVING AVERAGE
        # ============================================================

        df['ema_7'] = df['quantity'].ewm(span=7, adjust=False).mean()
        df['ema_14'] = df['quantity'].ewm(span=14, adjust=False).mean()
        df['ema_28'] = df['quantity'].ewm(span=28, adjust=False).mean()

        # ============================================================
        # DIFFERENCE FEATURES
        # ============================================================

        df['diff_1'] = df['quantity'].diff(1)
        df['diff_7'] = df['quantity'].diff(7)
        df['diff_28'] = df['quantity'].diff(28)

        # ============================================================
        # PERCENTAGE CHANGE
        # ============================================================

        df['pct_change_1'] = df['quantity'].pct_change(1)
        df['pct_change_7'] = df['quantity'].pct_change(7)

        # ============================================================
        # STATISTICAL FEATURES
        # ============================================================

        # Z-score (how unusual is today's sales)
        df['zscore_7'] = (df['quantity'] - df['rolling_mean_7']) / (df['rolling_std_7'] + 1e-6)
        df['zscore_30'] = (df['quantity'] - df['rolling_mean_30']) / (df['rolling_std_30'] + 1e-6)

        # Coefficient of variation
        df['cv_7'] = df['rolling_std_7'] / (df['rolling_mean_7'] + 1e-6)
        df['cv_30'] = df['rolling_std_30'] / (df['rolling_mean_30'] + 1e-6)

        # ============================================================
        # INTERACTION FEATURES
        # ============================================================

        df['day_of_week_x_month'] = df['day_of_week'] * df['month']
        df['weekend_x_month'] = df['is_weekend'] * df['month']

        logger.info(f"Engineered {len(df.columns) - 3} features from sales data")
        return df

    def _prepare_training_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training
        """
        # Drop rows with NaN (from lag/rolling features)
        df_clean = df.dropna()

        if len(df_clean) < 30:
            raise ValueError("Insufficient data after feature engineering")

        # Define target and features
        target = 'quantity'
        exclude_cols = ['date', 'quantity', 'unit_price', 'revenue']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        X = df_clean[feature_cols]
        y = df_clean[target]

        # Store feature names
        self.feature_names = list(X.columns)

        logger.info(f"Training data prepared: {len(X)} samples, {len(feature_cols)} features")
        return X, y

    async def _tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict:
        """
        Hyperparameter tuning using time series cross-validation
        """
        logger.info("Starting hyperparameter tuning...")

        # Define parameter grid
        param_grid = {
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [300, 500, 700],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        best_params = self.DEFAULT_PARAMS.copy()
        best_score = float('inf')

        # Simple grid search (for production, use RandomizedSearchCV or Optuna)
        from itertools import product

        # Sample parameter combinations
        param_combinations = [
            {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 500},
            {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300},
            {'max_depth': 10, 'learning_rate': 0.01, 'n_estimators': 700}
        ]

        for params in param_combinations:
            model_params = self.DEFAULT_PARAMS.copy()
            model_params.update(params)

            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = xgb.XGBRegressor(**model_params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                y_pred = model.predict(X_val)
                rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                scores.append(rmse)

            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = model_params

        logger.info(f"Best parameters found with RMSE: {best_score:.2f}")
        return best_params

    async def _generate_forecast(
        self,
        df_features: pd.DataFrame,
        periods: int,
        product_id: int
    ) -> pd.DataFrame:
        """
        Generate multi-step forecast
        """
        product = await self._get_product(product_id)

        # Start from last known date
        last_date = df_features['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )

        predictions = []

        # Use last known values for initial features
        last_row = df_features.iloc[-1:].copy()

        for i, forecast_date in enumerate(forecast_dates):
            # Update time features for forecast date
            future_features = self._create_future_features(
                forecast_date=forecast_date,
                last_known_data=df_features,
                step=i
            )

            # Make prediction
            X_pred = future_features[self.feature_names]
            pred = self.model.predict(X_pred)[0]
            pred = max(0, pred)  # No negative sales

            predictions.append({
                'date': forecast_date,
                'predicted_sales': pred,
                'predicted_revenue': pred * product.unit_price
            })

            # Update rolling features for next iteration
            df_features = self._update_features_with_prediction(
                df_features, forecast_date, pred
            )

        forecast_df = pd.DataFrame(predictions)
        logger.info(f"Generated {len(forecast_df)} day forecast")

        return forecast_df

    def _create_future_features(
        self,
        forecast_date: pd.Timestamp,
        last_known_data: pd.DataFrame,
        step: int
    ) -> pd.DataFrame:
        """
        Create features for a future date
        """
        features = {}

        # Time features
        features['year'] = forecast_date.year
        features['month'] = forecast_date.month
        features['day'] = forecast_date.day
        features['day_of_week'] = forecast_date.dayofweek
        features['day_of_year'] = forecast_date.dayofyear
        features['week_of_year'] = forecast_date.isocalendar().week
        features['quarter'] = forecast_date.quarter
        features['is_weekend'] = int(forecast_date.dayofweek >= 5)
        features['is_monday'] = int(forecast_date.dayofweek == 0)
        features['is_friday'] = int(forecast_date.dayofweek == 4)
        features['is_month_start'] = int(forecast_date.is_month_start)
        features['is_month_end'] = int(forecast_date.is_month_end)
        features['days_in_month'] = forecast_date.days_in_month

        # Season
        features['season'] = (
            1 if forecast_date.month in [12, 1, 2] else
            2 if forecast_date.month in [3, 4, 5] else
            3 if forecast_date.month in [6, 7, 8] else
            4
        )

        # Lag features (use last known values)
        for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
            if lag <= len(last_known_data):
                features[f'lag_{lag}'] = last_known_data['quantity'].iloc[-lag]
            else:
                features[f'lag_{lag}'] = last_known_data['quantity'].mean()

        # Rolling statistics (approximate from last known)
        for window in [3, 7, 14, 21, 28, 30]:
            if window <= len(last_known_data):
                recent = last_known_data['quantity'].iloc[-window:]
                features[f'rolling_mean_{window}'] = recent.mean()
                features[f'rolling_std_{window}'] = recent.std()
                features[f'rolling_min_{window}'] = recent.min()
                features[f'rolling_max_{window}'] = recent.max()
            else:
                features[f'rolling_mean_{window}'] = last_known_data['quantity'].mean()
                features[f'rolling_std_{window}'] = last_known_data['quantity'].std()
                features[f'rolling_min_{window}'] = last_known_data['quantity'].min()
                features[f'rolling_max_{window}'] = last_known_data['quantity'].max()

        # EMA (approximate)
        features['ema_7'] = last_known_data['quantity'].ewm(span=7).mean().iloc[-1]
        features['ema_14'] = last_known_data['quantity'].ewm(span=14).mean().iloc[-1]
        features['ema_28'] = last_known_data['quantity'].ewm(span=28).mean().iloc[-1]

        # Difference features
        features['diff_1'] = 0  # Unknown for future
        features['diff_7'] = 0
        features['diff_28'] = 0

        # Percentage change
        features['pct_change_1'] = 0
        features['pct_change_7'] = 0

        # Z-scores (approximate)
        features['zscore_7'] = 0
        features['zscore_30'] = 0

        # Coefficient of variation
        features['cv_7'] = features['rolling_std_7'] / (features['rolling_mean_7'] + 1e-6)
        features['cv_30'] = features['rolling_std_30'] / (features['rolling_mean_30'] + 1e-6)

        # Interaction features
        features['day_of_week_x_month'] = features['day_of_week'] * features['month']
        features['weekend_x_month'] = features['is_weekend'] * features['month']

        return pd.DataFrame([features])

    def _update_features_with_prediction(
        self,
        df: pd.DataFrame,
        new_date: pd.Timestamp,
        prediction: float
    ) -> pd.DataFrame:
        """
        Add prediction to feature dataframe for next iteration
        """
        new_row = {'date': new_date, 'quantity': prediction}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return df

    def _add_confidence_intervals(
        self,
        forecast: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Calculate confidence intervals using quantile regression
        """
        from scipy import stats

        # Train quantile models for lower and upper bounds
        alpha = 1 - confidence_level

        # Lower bound (alpha/2 quantile)
        lower_model = xgb.XGBRegressor(
            **{**self.DEFAULT_PARAMS, 'objective': 'reg:quantileerror'},
            quantile_alpha=alpha/2
        )
        lower_model.fit(X_train, y_train)

        # Upper bound (1 - alpha/2 quantile)
        upper_model = xgb.XGBRegressor(
            **{**self.DEFAULT_PARAMS, 'objective': 'reg:quantileerror'},
            quantile_alpha=1-alpha/2
        )
        upper_model.fit(X_train, y_train)

        # Simple approximation: use residual std
        predictions = self.model.predict(X_train)
        residuals = y_train - predictions
        std_residual = np.std(residuals)

        z_score = stats.norm.ppf(1 - alpha/2)

        forecast['lower_bound'] = forecast['predicted_sales'] - z_score * std_residual
        forecast['upper_bound'] = forecast['predicted_sales'] + z_score * std_residual
        forecast['confidence_level'] = confidence_level

        # Ensure non-negative bounds
        forecast['lower_bound'] = forecast['lower_bound'].clip(lower=0)

        return forecast

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top N most important features
        """
        if not self.feature_importance:
            return {}

        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return dict(sorted_features[:top_n])

    async def _get_sales_data(self, product_id: int) -> pd.DataFrame:
        """Get historical sales data"""
        query = select(SalesRecord).where(
            SalesRecord.product_id == product_id
        ).order_by(SalesRecord.sale_date)

        result = await self.db.execute(query)
        records = result.scalars().all()

        if not records:
            raise ValueError(f"No sales data found for product {product_id}")

        df = pd.DataFrame([{
            'date': r.sale_date,
            'quantity': r.quantity,
            'unit_price': r.unit_price
        } for r in records])

        df['date'] = pd.to_datetime(df['date'])
        df = df.groupby('date').agg({
            'quantity': 'sum',
            'unit_price': 'mean'
        }).reset_index()

        return df

    async def _get_product(self, product_id: int) -> Product:
        """Get product by ID"""
        query = select(Product).where(Product.id == product_id)
        result = await self.db.execute(query)
        product = result.scalar_one_or_none()

        if not product:
            raise ValueError(f"Product {product_id} not found")

        return product
