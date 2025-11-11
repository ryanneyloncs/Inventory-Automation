import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.services.forecasting.forecasting_service import TimeSeriesForecaster, ForecastingService


@pytest.fixture
def forecaster():
    """Create forecaster instance"""
    return TimeSeriesForecaster()


@pytest.fixture
def sample_sales_data():
    """Generate sample sales data"""
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    quantities = [10 + (5 * (i % 7 == 0)) for i in range(len(dates))]  # Weekly pattern
    
    return pd.DataFrame({
        'sale_date': dates,
        'quantity': quantities,
        'revenue': [q * 20.0 for q in quantities],
        'sales_channel': ['online'] * len(dates),
        'promotion_active': [False] * len(dates)
    })


def test_prepare_prophet_data(forecaster, sample_sales_data):
    """Test Prophet data preparation"""
    prophet_df = forecaster.prepare_prophet_data(sample_sales_data)
    
    assert 'ds' in prophet_df.columns
    assert 'y' in prophet_df.columns
    assert len(prophet_df) > 0
    assert prophet_df['y'].sum() > 0


def test_train_prophet_model(forecaster, sample_sales_data):
    """Test Prophet model training"""
    model = forecaster.train_prophet_model(sample_sales_data)
    
    assert model is not None
    assert forecaster.prophet_model is not None


def test_create_ml_features(forecaster, sample_sales_data):
    """Test feature engineering"""
    features_df = forecaster.create_ml_features(sample_sales_data)
    
    # Check expected columns exist
    assert 'day_of_week' in features_df.columns
    assert 'month' in features_df.columns
    assert 'lag_1' in features_df.columns
    assert 'rolling_mean_7' in features_df.columns
    assert len(features_df) > 0


def test_train_random_forest_model(forecaster, sample_sales_data):
    """Test Random Forest training"""
    model = forecaster.train_random_forest_model(sample_sales_data)
    
    if model:  # Only if enough data
        assert forecaster.rf_model is not None


@pytest.mark.asyncio
async def test_get_sales_data(db_session, sample_product):
    """Test sales data retrieval"""
    from src.models.database import SalesRecord
    from decimal import Decimal
    
    # Add sample sales records
    for i in range(10):
        record = SalesRecord(
            product_id=sample_product.id,
            sale_date=datetime.utcnow() - timedelta(days=i),
            quantity=10,
            revenue=Decimal("200.00"),
            sales_channel='online',
            promotion_active=False
        )
        db_session.add(record)
    
    await db_session.commit()
    
    # Test retrieval
    service = ForecastingService(db_session)
    sales_data = await service.get_sales_data(sample_product.id, days_back=30)
    
    assert len(sales_data) == 10
    assert 'sale_date' in sales_data.columns
    assert 'quantity' in sales_data.columns


@pytest.mark.asyncio
async def test_train_and_forecast_insufficient_data(db_session, sample_product):
    """Test forecast with insufficient data"""
    service = ForecastingService(db_session)
    result = await service.train_and_forecast(sample_product.id)
    
    assert result['success'] == False
    assert 'Insufficient' in result['error']


def test_forecast_prophet(forecaster, sample_sales_data):
    """Test Prophet forecasting"""
    forecaster.train_prophet_model(sample_sales_data)
    forecast = forecaster.forecast_prophet(periods=30)
    
    assert len(forecast) == 30
    assert 'ds' in forecast.columns
    assert 'yhat' in forecast.columns
    assert 'yhat_lower' in forecast.columns
    assert 'yhat_upper' in forecast.columns


def test_ensemble_forecast(forecaster, sample_sales_data):
    """Test ensemble forecasting"""
    forecaster.train_prophet_model(sample_sales_data)
    
    result = forecaster.ensemble_forecast(sample_sales_data, periods=30)
    
    assert 'dates' in result
    assert 'predictions' in result
    assert 'lower_bound' in result
    assert 'upper_bound' in result
    assert 'models_used' in result
    assert len(result['predictions']) == 30


def test_save_and_load_models(forecaster, sample_sales_data, tmp_path):
    """Test model persistence"""
    forecaster.model_dir = tmp_path
    forecaster.train_prophet_model(sample_sales_data)
    
    # Save models
    forecaster.save_models(product_id=1)
    
    # Create new forecaster and load
    new_forecaster = TimeSeriesForecaster()
    new_forecaster.model_dir = tmp_path
    success = new_forecaster.load_models(product_id=1)
    
    assert success
    assert new_forecaster.prophet_model is not None
