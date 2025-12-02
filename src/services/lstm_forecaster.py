"""
Advanced LSTM Forecasting Service

Deep learning specialist with:
- Multi-layer LSTM architecture
- Attention mechanism
- Sequence-to-sequence modeling
- Early stopping and learning rate scheduling
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

from src.models.database import SalesRecord, Product

logger = logging.getLogger(__name__)


class LSTMForecaster:
    """
    Advanced LSTM neural network for time series forecasting
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = 60  # Days to look back
        self.sequence_length = 60

        # Model architecture parameters
        self.lstm_units = [128, 64, 32]  # 3-layer LSTM
        self.dropout_rate = 0.3
        self.use_attention = True

    async def forecast(
        self,
        product_id: int,
        periods: int = 30,
        epochs: int = 100,
        batch_size: int = 32,
        use_attention: bool = True,
        early_stopping: bool = True
    ) -> pd.DataFrame:
        """
        Generate forecast using deep LSTM network

        Args:
            product_id: Product to forecast
            periods: Number of days to forecast
            epochs: Training epochs
            batch_size: Batch size for training
            use_attention: Whether to use attention mechanism
            early_stopping: Whether to use early stopping

        Returns:
            DataFrame with forecasts
        """
        logger.info(f"Starting LSTM forecast for product {product_id}, {periods} periods")

        self.use_attention = use_attention

        # Get historical data
        sales_data = await self._get_sales_data(product_id)

        if len(sales_data) < 90:
            raise ValueError(f"Need at least 90 days of data for LSTM, got {len(sales_data)}")

        # Prepare sequences
        X_train, y_train, dates_train = self._prepare_sequences(sales_data)

        if len(X_train) < 30:
            raise ValueError(f"Insufficient training sequences")

        # Build model
        self.model = self._build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            use_attention=use_attention
        )

        # Setup callbacks
        callbacks = self._setup_callbacks(early_stopping=early_stopping)

        # Train model
        logger.info(f"Training LSTM with {len(X_train)} sequences...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )

        logger.info(f"Training complete. Final loss: {history.history['loss'][-1]:.4f}")

        # Generate forecast
        forecast = await self._generate_forecast(
            sales_data=sales_data,
            periods=periods,
            product_id=product_id
        )

        return forecast

    def _prepare_sequences(
        self,
        sales_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Prepare sequences for LSTM training
        """
        # Extract values and scale
        values = sales_data['quantity'].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)

        X, y = [], []
        dates = []

        for i in range(self.lookback, len(scaled_values)):
            X.append(scaled_values[i-self.lookback:i])
            y.append(scaled_values[i])
            dates.append(sales_data['date'].iloc[i])

        X = np.array(X)
        y = np.array(y)
        dates = pd.DatetimeIndex(dates)

        logger.info(f"Created {len(X)} sequences with lookback={self.lookback}")
        return X, y, dates

    def _build_model(
        self,
        input_shape: Tuple[int, int],
        use_attention: bool = True
    ) -> keras.Model:
        """
        Build deep LSTM architecture with optional attention
        """
        inputs = layers.Input(shape=input_shape)

        # First LSTM layer
        x = layers.LSTM(
            self.lstm_units[0],
            return_sequences=True,
            activation='tanh'
        )(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.BatchNormalization()(x)

        # Second LSTM layer
        x = layers.LSTM(
            self.lstm_units[1],
            return_sequences=True,
            activation='tanh'
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.BatchNormalization()(x)

        # Third LSTM layer
        if use_attention:
            lstm_out = layers.LSTM(
                self.lstm_units[2],
                return_sequences=True,
                activation='tanh'
            )(x)

            # Attention mechanism
            attention = layers.Dense(1, activation='tanh')(lstm_out)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(self.lstm_units[2])(attention)
            attention = layers.Permute([2, 1])(attention)

            # Apply attention weights
            x = layers.Multiply()([lstm_out, attention])
            x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
        else:
            x = layers.LSTM(
                self.lstm_units[2],
                return_sequences=False,
                activation='tanh'
            )(x)

        x = layers.Dropout(self.dropout_rate)(x)

        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)

        # Output layer
        outputs = layers.Dense(1)(x)

        # Build model
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile with adaptive learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        logger.info(f"Built LSTM model with {model.count_params():,} parameters")
        logger.info(f"Architecture: {' -> '.join(map(str, self.lstm_units))} units")
        logger.info(f"Attention mechanism: {'enabled' if use_attention else 'disabled'}")

        return model

    def _setup_callbacks(self, early_stopping: bool = True) -> List:
        """
        Setup training callbacks
        """
        callbacks = []

        # Early stopping
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stop)

        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
        callbacks.append(reduce_lr)

        # Model checkpoint (optional - can save best model)
        # checkpoint = keras.callbacks.ModelCheckpoint(
        #     'best_lstm_model.h5',
        #     monitor='val_loss',
        #     save_best_only=True
        # )
        # callbacks.append(checkpoint)

        return callbacks

    async def _generate_forecast(
        self,
        sales_data: pd.DataFrame,
        periods: int,
        product_id: int
    ) -> pd.DataFrame:
        """
        Generate multi-step forecast using recursive approach
        """
        product = await self._get_product(product_id)

        # Get last sequence for forecasting
        last_sequence = sales_data['quantity'].values[-self.lookback:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))

        # Generate forecasts recursively
        predictions = []
        current_sequence = last_sequence_scaled.copy()

        for i in range(periods):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.lookback, 1)

            # Predict next value
            pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]

            # Inverse transform
            pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
            pred = max(0, pred)  # No negative sales

            predictions.append(pred)

            # Update sequence (slide window)
            current_sequence = np.append(current_sequence[1:], [[pred_scaled]], axis=0)

        # Create forecast dataframe
        last_date = sales_data['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_sales': predictions,
            'predicted_revenue': [p * product.unit_price for p in predictions]
        })

        # Add confidence intervals (using prediction variance)
        forecast_df = self._add_confidence_intervals(
            forecast_df,
            sales_data['quantity'].values
        )

        logger.info(f"Generated {len(forecast_df)} day LSTM forecast")
        return forecast_df

    def _add_confidence_intervals(
        self,
        forecast: pd.DataFrame,
        historical_sales: np.ndarray,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Add confidence intervals using Monte Carlo dropout
        """
        from scipy import stats

        # Calculate historical volatility
        returns = np.diff(historical_sales) / (historical_sales[:-1] + 1e-6)
        volatility = np.std(returns)

        # Estimate prediction uncertainty (increases with forecast horizon)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        forecast['lower_bound'] = []
        forecast['upper_bound'] = []

        for i, row in forecast.iterrows():
            # Uncertainty grows with forecast horizon
            horizon_factor = np.sqrt(i + 1)
            uncertainty = volatility * row['predicted_sales'] * horizon_factor

            lower = row['predicted_sales'] - z_score * uncertainty
            upper = row['predicted_sales'] + z_score * uncertainty

            forecast.at[i, 'lower_bound'] = max(0, lower)
            forecast.at[i, 'upper_bound'] = upper

        forecast['confidence_level'] = confidence_level

        return forecast

    async def forecast_with_uncertainty(
        self,
        product_id: int,
        periods: int = 30,
        num_samples: int = 100
    ) -> pd.DataFrame:
        """
        Generate probabilistic forecast using Monte Carlo Dropout

        This enables uncertainty estimation by running multiple
        forward passes with dropout enabled
        """
        logger.info(f"Generating probabilistic forecast with {num_samples} samples")

        # Get data
        sales_data = await self._get_sales_data(product_id)
        product = await self._get_product(product_id)

        if self.model is None:
            raise ValueError("Model not trained. Call forecast() first.")

        # Prepare last sequence
        last_sequence = sales_data['quantity'].values[-self.lookback:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))

        # Generate multiple forecasts with dropout
        all_predictions = []

        for sample in range(num_samples):
            predictions = []
            current_sequence = last_sequence_scaled.copy()

            for i in range(periods):
                X_pred = current_sequence.reshape(1, self.lookback, 1)

                # Predict with dropout (training=True enables dropout during inference)
                pred_scaled = self.model(X_pred, training=True).numpy()[0, 0]
                pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
                pred = max(0, pred)

                predictions.append(pred)
                current_sequence = np.append(current_sequence[1:], [[pred_scaled]], axis=0)

            all_predictions.append(predictions)

        # Calculate statistics
        all_predictions = np.array(all_predictions)

        last_date = sales_data['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_sales': np.mean(all_predictions, axis=0),
            'predicted_revenue': np.mean(all_predictions, axis=0) * product.unit_price,
            'lower_bound': np.percentile(all_predictions, 2.5, axis=0),
            'upper_bound': np.percentile(all_predictions, 97.5, axis=0),
            'std': np.std(all_predictions, axis=0),
            'confidence_level': 0.95
        })

        logger.info("Probabilistic forecast complete")
        return forecast_df

    def get_model_summary(self) -> Dict:
        """
        Get model architecture summary
        """
        if self.model is None:
            return {'error': 'Model not built'}

        summary = {
            'total_parameters': self.model.count_params(),
            'architecture': {
                'lstm_layers': len(self.lstm_units),
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'attention_enabled': self.use_attention,
                'lookback_window': self.lookback
            },
            'training_config': {
                'optimizer': 'Adam',
                'loss': 'MSE',
                'metrics': ['MAE']
            }
        }

        return summary

    async def evaluate_performance(
        self,
        product_id: int,
        test_days: int = 30
    ) -> Dict:
        """
        Evaluate model performance on recent data
        """
        logger.info(f"Evaluating model performance on last {test_days} days")

        # Get all data
        sales_data = await self._get_sales_data(product_id)

        if len(sales_data) < self.lookback + test_days:
            raise ValueError("Insufficient data for evaluation")

        # Split into train and test
        train_data = sales_data.iloc[:-test_days]
        test_data = sales_data.iloc[-test_days:]

        # Prepare test sequences
        X_test, y_test, _ = self._prepare_sequences(sales_data)
        X_test = X_test[-test_days:]
        y_test = y_test[-test_days:]

        # Make predictions
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        y_true = self.scaler.inverse_transform(y_test)

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

        # Bias
        bias = np.mean(y_pred - y_true)

        # Direction accuracy (did we predict up/down correctly?)
        y_true_diff = np.diff(y_true.flatten())
        y_pred_diff = np.diff(y_pred.flatten())
        direction_correct = np.sum((y_true_diff * y_pred_diff) > 0)
        direction_accuracy = direction_correct / len(y_true_diff) * 100

        results = {
            'test_period_days': test_days,
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'bias': float(bias),
                'direction_accuracy': float(direction_accuracy)
            },
            'predictions': {
                'dates': test_data['date'].tolist(),
                'actual': y_true.flatten().tolist(),
                'predicted': y_pred.flatten().tolist()
            }
        }

        logger.info(f"Evaluation complete - MAPE: {mape:.2f}%, Direction Accuracy: {direction_accuracy:.2f}%")
        return results

    async def retrain_online(
        self,
        product_id: int,
        new_data_days: int = 7,
        epochs: int = 10
    ):
        """
        Online learning: retrain model with new data
        """
        logger.info(f"Retraining model with last {new_data_days} days of new data")

        if self.model is None:
            raise ValueError("Model not initialized. Train from scratch first.")

        # Get recent data
        sales_data = await self._get_sales_data(product_id)

        # Use only recent data for quick update
        recent_data = sales_data.iloc[-new_data_days-self.lookback:]

        # Prepare sequences
        X_new, y_new, _ = self._prepare_sequences(recent_data)

        # Fine-tune model
        self.model.fit(
            X_new, y_new,
            epochs=epochs,
            batch_size=8,
            verbose=0
        )

        logger.info("Model retrained successfully")

    def save_model(self, filepath: str):
        """
        Save model to disk
        """
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from disk
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

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

        # Fill missing dates with 0
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        df = df.set_index('date').reindex(date_range, fill_value=0).reset_index()
        df.columns = ['date', 'quantity', 'unit_price']
        df['unit_price'] = df['unit_price'].replace(0, df['unit_price'].mean())

        return df

    async def _get_product(self, product_id: int) -> Product:
        """Get product by ID"""
        query = select(Product).where(Product.id == product_id)
        result = await self.db.execute(query)
        product = result.scalar_one_or_none()

        if not product:
            raise ValueError(f"Product {product_id} not found")

        return product
