"""
Sales Database Model
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, ForeignKey, Date, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, date
from typing import Dict, Any, Optional
from src.models.database import Base


class Sale(Base):
    """Sales transaction history"""
    __tablename__ = 'sales'

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(50), unique=True, nullable=False, index=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)
    
    # Sale details
    sale_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    sale_date_only = Column(Date, nullable=False, index=True)  # For efficient date-based queries
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)  # Price at time of sale
    unit_cost = Column(Float, nullable=False)   # Cost at time of sale
    total_revenue = Column(Float, nullable=False)  # quantity * unit_price
    total_cost = Column(Float, nullable=False)     # quantity * unit_cost
    profit = Column(Float, nullable=False)         # total_revenue - total_cost
    
    # Sale context
    channel = Column(String(50), nullable=True)  # online, in-store, wholesale, etc.
    region = Column(String(100), nullable=True)  # Geographic region
    store_id = Column(Integer, nullable=True)    # If multi-store operation
    
    # Customer information (optional, for analytics)
    customer_id = Column(String(50), nullable=True, index=True)
    customer_segment = Column(String(50), nullable=True)  # new, returning, vip, etc.
    
    # Promotional tracking
    promotion_applied = Column(Boolean, nullable=False, default=False)
    promotion_id = Column(String(50), nullable=True)
    discount_amount = Column(Float, nullable=False, default=0.0)
    discount_pct = Column(Float, nullable=False, default=0.0)
    
    # Temporal attributes (for ML features)
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 6=Sunday
    week_of_year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    quarter = Column(Integer, nullable=False)
    year = Column(Integer, nullable=False)
    is_weekend = Column(Boolean, nullable=False, default=False)
    is_holiday = Column(Boolean, nullable=False, default=False)
    
    # Additional metadata
    metadata = Column(JSONB, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="sales")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sale to dictionary"""
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'product_id': self.product_id,
            'sale_date': self.sale_date.isoformat(),
            'sale_date_only': self.sale_date_only.isoformat(),
            'quantity': self.quantity,
            'unit_price': self.unit_price,
            'unit_cost': self.unit_cost,
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'profit': self.profit,
            'channel': self.channel,
            'region': self.region,
            'store_id': self.store_id,
            'customer_id': self.customer_id,
            'customer_segment': self.customer_segment,
            'promotion_applied': self.promotion_applied,
            'promotion_id': self.promotion_id,
            'discount_amount': self.discount_amount,
            'discount_pct': self.discount_pct,
            'temporal': {
                'day_of_week': self.day_of_week,
                'week_of_year': self.week_of_year,
                'month': self.month,
                'quarter': self.quarter,
                'year': self.year,
                'is_weekend': self.is_weekend,
                'is_holiday': self.is_holiday
            },
            'metadata': self.metadata,
            'notes': self.notes,
            'created_at': self.created_at.isoformat()
        }
    
    @property
    def profit_margin(self) -> float:
        """Calculate profit margin percentage"""
        if self.total_revenue == 0:
            return 0.0
        return (self.profit / self.total_revenue) * 100
    
    @property
    def effective_price(self) -> float:
        """Calculate effective price after discounts"""
        return self.unit_price - (self.discount_amount / self.quantity if self.quantity > 0 else 0)
    
    def __repr__(self) -> str:
        return f"<Sale(id={self.id}, transaction_id='{self.transaction_id}', product_id={self.product_id}, quantity={self.quantity}, revenue=${self.total_revenue:.2f})>"


class SalesAggregate(Base):
    """Pre-aggregated sales data for faster analytics queries"""
    __tablename__ = 'sales_aggregates'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)
    aggregation_level = Column(String(20), nullable=False, index=True)  # daily, weekly, monthly
    period_start = Column(Date, nullable=False, index=True)
    period_end = Column(Date, nullable=False)
    
    # Aggregated metrics
    total_quantity = Column(Integer, nullable=False)
    total_revenue = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)
    total_profit = Column(Float, nullable=False)
    transaction_count = Column(Integer, nullable=False)
    
    # Average metrics
    avg_quantity_per_transaction = Column(Float, nullable=False)
    avg_unit_price = Column(Float, nullable=False)
    avg_profit_margin = Column(Float, nullable=False)
    
    # Channel breakdown (stored as JSONB)
    channel_breakdown = Column(JSONB, nullable=True)
    
    # Promotional metrics
    promo_sales_count = Column(Integer, nullable=False, default=0)
    promo_revenue = Column(Float, nullable=False, default=0.0)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product = relationship("Product")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'product_id': self.product_id,
            'aggregation_level': self.aggregation_level,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_quantity': self.total_quantity,
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'total_profit': self.total_profit,
            'transaction_count': self.transaction_count,
            'avg_quantity_per_transaction': self.avg_quantity_per_transaction,
            'avg_unit_price': self.avg_unit_price,
            'avg_profit_margin': self.avg_profit_margin,
            'channel_breakdown': self.channel_breakdown,
            'promo_sales_count': self.promo_sales_count,
            'promo_revenue': self.promo_revenue,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @property
    def days_in_period(self) -> int:
        """Calculate number of days in the aggregation period"""
        return (self.period_end - self.period_start).days + 1
    
    @property
    def daily_avg_revenue(self) -> float:
        """Calculate average daily revenue"""
        days = self.days_in_period
        return self.total_revenue / days if days > 0 else 0.0
    
    def __repr__(self) -> str:
        return f"<SalesAggregate(product_id={self.product_id}, level='{self.aggregation_level}', period={self.period_start} to {self.period_end})>"


class SalesTrend(Base):
    """Track sales trends and growth metrics"""
    __tablename__ = 'sales_trends'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=True)  # NULL for overall trends
    trend_date = Column(Date, nullable=False, index=True)
    
    # Growth metrics (compared to previous period)
    revenue_growth_pct = Column(Float, nullable=True)
    quantity_growth_pct = Column(Float, nullable=True)
    
    # Moving averages
    ma_7_day_revenue = Column(Float, nullable=True)
    ma_30_day_revenue = Column(Float, nullable=True)
    ma_90_day_revenue = Column(Float, nullable=True)
    
    # Volatility metrics
    revenue_std_dev = Column(Float, nullable=True)
    coefficient_of_variation = Column(Float, nullable=True)
    
    # Trend indicators
    trend_direction = Column(String(20), nullable=True)  # increasing, decreasing, stable
    trend_strength = Column(Float, nullable=True)  # 0-1 scale
    seasonality_index = Column(Float, nullable=True)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'product_id': self.product_id,
            'trend_date': self.trend_date.isoformat(),
            'revenue_growth_pct': self.revenue_growth_pct,
            'quantity_growth_pct': self.quantity_growth_pct,
            'moving_averages': {
                '7_day': self.ma_7_day_revenue,
                '30_day': self.ma_30_day_revenue,
                '90_day': self.ma_90_day_revenue
            },
            'volatility': {
                'std_dev': self.revenue_std_dev,
                'coefficient_of_variation': self.coefficient_of_variation
            },
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'seasonality_index': self.seasonality_index,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self) -> str:
        product_info = f"product_id={self.product_id}" if self.product_id else "overall"
        return f"<SalesTrend({product_info}, date={self.trend_date}, direction='{self.trend_direction}')>"
