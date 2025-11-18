"""
Sales Forecasting Suite - Database Models
"""
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, Boolean,
    Text, ForeignKey, ARRAY, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any, List
from src.models.database import Base


class SalesForecastResult(Base):
    """Multi-horizon sales forecast results"""
    __tablename__ = 'sales_forecast_results'

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)
    forecast_date = Column(Date, nullable=False)
    horizon = Column(String(20), nullable=False)  # hourly, daily, weekly, monthly
    predicted_sales = Column(Float, nullable=False)
    predicted_revenue = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=False)
    upper_bound = Column(Float, nullable=False)
    confidence_level = Column(Float, nullable=False, default=0.95)
    model_used = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    metadata = Column(JSONB, nullable=True)

    # Relationships
    product = relationship("Product", back_populates="sales_forecasts")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'product_id': self.product_id,
            'forecast_date': self.forecast_date.isoformat(),
            'horizon': self.horizon,
            'predicted_sales': self.predicted_sales,
            'predicted_revenue': self.predicted_revenue,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'confidence_level': self.confidence_level,
            'model_used': self.model_used,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


class ForecastScenario(Base):
    """Scenario planning for what-if analysis"""
    __tablename__ = 'forecast_scenarios'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    scenario_type = Column(String(50), nullable=False)  # promotion, new_product, seasonal, custom
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    parameters = Column(JSONB, nullable=False)
    created_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)

    # Relationships
    forecast_results = relationship("ScenarioForecastResult", back_populates="scenario", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'parameters': self.parameters,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }


class ScenarioForecastResult(Base):
    """Forecast results for specific scenarios"""
    __tablename__ = 'scenario_forecast_results'

    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey('forecast_scenarios.id', ondelete='CASCADE'), nullable=False)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)
    forecast_date = Column(Date, nullable=False)
    baseline_sales = Column(Float, nullable=False)
    scenario_sales = Column(Float, nullable=False)
    sales_lift = Column(Float, nullable=False)
    baseline_revenue = Column(Float, nullable=False)
    scenario_revenue = Column(Float, nullable=False)
    incremental_revenue = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    scenario = relationship("ForecastScenario", back_populates="forecast_results")
    product = relationship("Product")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'scenario_id': self.scenario_id,
            'product_id': self.product_id,
            'forecast_date': self.forecast_date.isoformat(),
            'baseline_sales': self.baseline_sales,
            'scenario_sales': self.scenario_sales,
            'sales_lift': self.sales_lift,
            'baseline_revenue': self.baseline_revenue,
            'scenario_revenue': self.scenario_revenue,
            'incremental_revenue': self.incremental_revenue,
            'created_at': self.created_at.isoformat()
        }


class SeasonalDecomposition(Base):
    """Seasonal trend decomposition results"""
    __tablename__ = 'seasonal_decomposition'

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)
    date = Column(Date, nullable=False)
    observed = Column(Float, nullable=False)
    trend = Column(Float, nullable=False)
    seasonal = Column(Float, nullable=False)
    residual = Column(Float, nullable=False)
    decomposition_type = Column(String(20), nullable=False)  # additive, multiplicative
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    product = relationship("Product")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'product_id': self.product_id,
            'date': self.date.isoformat(),
            'observed': self.observed,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'residual': self.residual,
            'decomposition_type': self.decomposition_type,
            'created_at': self.created_at.isoformat()
        }


class EventImpact(Base):
    """Event impact tracking and analysis"""
    __tablename__ = 'event_impacts'

    id = Column(Integer, primary_key=True)
    event_name = Column(String(200), nullable=False)
    event_type = Column(String(50), nullable=False)  # holiday, promotion, external, custom
    event_date = Column(Date, nullable=False)
    impact_start_date = Column(Date, nullable=False)
    impact_end_date = Column(Date, nullable=False)
    affected_products = Column(ARRAY(Integer), nullable=True)
    baseline_sales = Column(Float, nullable=True)
    actual_sales = Column(Float, nullable=True)
    sales_lift_pct = Column(Float, nullable=True)
    revenue_impact = Column(Float, nullable=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'event_name': self.event_name,
            'event_type': self.event_type,
            'event_date': self.event_date.isoformat(),
            'impact_start_date': self.impact_start_date.isoformat(),
            'impact_end_date': self.impact_end_date.isoformat(),
            'affected_products': self.affected_products,
            'baseline_sales': self.baseline_sales,
            'actual_sales': self.actual_sales,
            'sales_lift_pct': self.sales_lift_pct,
            'revenue_impact': self.revenue_impact,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class CannibalizationAnalysis(Base):
    """Product cannibalization analysis"""
    __tablename__ = 'cannibalization_analysis'

    id = Column(Integer, primary_key=True)
    new_product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)
    existing_product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)
    analysis_date = Column(Date, nullable=False)
    similarity_score = Column(Float, nullable=False)
    baseline_sales_existing = Column(Float, nullable=False)
    post_launch_sales_existing = Column(Float, nullable=False)
    cannibalization_rate = Column(Float, nullable=False)
    new_product_sales = Column(Float, nullable=False)
    net_incremental_sales = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    new_product = relationship("Product", foreign_keys=[new_product_id])
    existing_product = relationship("Product", foreign_keys=[existing_product_id])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'new_product_id': self.new_product_id,
            'existing_product_id': self.existing_product_id,
            'analysis_date': self.analysis_date.isoformat(),
            'similarity_score': self.similarity_score,
            'baseline_sales_existing': self.baseline_sales_existing,
            'post_launch_sales_existing': self.post_launch_sales_existing,
            'cannibalization_rate': self.cannibalization_rate,
            'new_product_sales': self.new_product_sales,
            'net_incremental_sales': self.net_incremental_sales,
            'created_at': self.created_at.isoformat()
        }


class ForecastAccuracyMetric(Base):
    """Track forecast accuracy by horizon"""
    __tablename__ = 'forecast_accuracy_metrics'

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=True)
    horizon = Column(String(20), nullable=False)
    metric_date = Column(Date, nullable=False)
    mae = Column(Float, nullable=True)  # Mean Absolute Error
    rmse = Column(Float, nullable=True)  # Root Mean Squared Error
    mape = Column(Float, nullable=True)  # Mean Absolute Percentage Error
    bias = Column(Float, nullable=True)  # Forecast bias
    forecast_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    product = relationship("Product")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'product_id': self.product_id,
            'horizon': self.horizon,
            'metric_date': self.metric_date.isoformat(),
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'bias': self.bias,
            'forecast_count': self.forecast_count,
            'created_at': self.created_at.isoformat()
        }
