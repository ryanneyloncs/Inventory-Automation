"""
Product Database Model
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any, Optional
from src.models.database import Base


class Product(Base):
    """Product catalog with inventory management"""
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True)
    sku = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=False, index=True)
    subcategory = Column(String(100), nullable=True)
    
    # Pricing
    unit_cost = Column(Float, nullable=False)  # Cost from supplier
    unit_price = Column(Float, nullable=False)  # Selling price
    
    # Inventory levels
    current_stock = Column(Integer, nullable=False, default=0)
    reserved_stock = Column(Integer, nullable=False, default=0)  # Reserved for orders
    available_stock = Column(Integer, nullable=False, default=0)  # current - reserved
    reorder_point = Column(Integer, nullable=False)  # Trigger reorder at this level
    reorder_quantity = Column(Integer, nullable=False)  # How much to order
    safety_stock = Column(Integer, nullable=False, default=0)  # Buffer stock
    max_stock = Column(Integer, nullable=True)  # Maximum inventory level
    
    # Supplier relationship
    supplier_id = Column(Integer, ForeignKey('suppliers.id', ondelete='SET NULL'), nullable=True)
    lead_time_days = Column(Integer, nullable=False, default=7)  # Delivery time
    
    # Product attributes
    weight = Column(Float, nullable=True)  # For shipping calculations
    dimensions = Column(String(50), nullable=True)  # L x W x H
    is_active = Column(Boolean, nullable=False, default=True)
    is_perishable = Column(Boolean, nullable=False, default=False)
    shelf_life_days = Column(Integer, nullable=True)  # For perishables
    
    # Lifecycle tracking
    launch_date = Column(DateTime, nullable=True)
    discontinue_date = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, nullable=True)  # Flexible storage for custom attributes
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    supplier = relationship("Supplier", back_populates="products")
    sales = relationship("Sale", back_populates="product", cascade="all, delete-orphan")
    sales_forecasts = relationship("SalesForecastResult", back_populates="product", cascade="all, delete-orphan")
    reorder_recommendations = relationship("ReorderRecommendation", back_populates="product", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert product to dictionary"""
        return {
            'id': self.id,
            'sku': self.sku,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'subcategory': self.subcategory,
            'unit_cost': self.unit_cost,
            'unit_price': self.unit_price,
            'current_stock': self.current_stock,
            'reserved_stock': self.reserved_stock,
            'available_stock': self.available_stock,
            'reorder_point': self.reorder_point,
            'reorder_quantity': self.reorder_quantity,
            'safety_stock': self.safety_stock,
            'max_stock': self.max_stock,
            'supplier_id': self.supplier_id,
            'lead_time_days': self.lead_time_days,
            'weight': self.weight,
            'dimensions': self.dimensions,
            'is_active': self.is_active,
            'is_perishable': self.is_perishable,
            'shelf_life_days': self.shelf_life_days,
            'launch_date': self.launch_date.isoformat() if self.launch_date else None,
            'discontinue_date': self.discontinue_date.isoformat() if self.discontinue_date else None,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @property
    def margin(self) -> float:
        """Calculate profit margin"""
        if self.unit_price == 0:
            return 0.0
        return ((self.unit_price - self.unit_cost) / self.unit_price) * 100
    
    @property
    def stock_value(self) -> float:
        """Calculate total inventory value at cost"""
        return self.current_stock * self.unit_cost
    
    @property
    def needs_reorder(self) -> bool:
        """Check if product needs reordering"""
        return self.available_stock <= self.reorder_point
    
    @property
    def stock_status(self) -> str:
        """Get human-readable stock status"""
        if self.available_stock == 0:
            return "out_of_stock"
        elif self.available_stock <= self.safety_stock:
            return "critical"
        elif self.available_stock <= self.reorder_point:
            return "low"
        elif self.max_stock and self.available_stock >= self.max_stock * 0.9:
            return "overstocked"
        else:
            return "normal"
    
    def update_available_stock(self) -> None:
        """Update available stock calculation"""
        self.available_stock = max(0, self.current_stock - self.reserved_stock)
    
    def __repr__(self) -> str:
        return f"<Product(id={self.id}, sku='{self.sku}', name='{self.name}', stock={self.current_stock})>"


class ReorderRecommendation(Base):
    """Automated reorder recommendations generated by ML system"""
    __tablename__ = 'reorder_recommendations'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)
    recommended_quantity = Column(Integer, nullable=False)
    recommended_date = Column(DateTime, nullable=False)
    priority = Column(String(20), nullable=False)  # critical, high, medium, low
    reason = Column(Text, nullable=True)
    expected_stockout_date = Column(DateTime, nullable=True)
    confidence_score = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default='pending')  # pending, approved, ordered, cancelled
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(String(100), nullable=True)
    
    # Relationships
    product = relationship("Product", back_populates="reorder_recommendations")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'product_id': self.product_id,
            'recommended_quantity': self.recommended_quantity,
            'recommended_date': self.recommended_date.isoformat(),
            'priority': self.priority,
            'reason': self.reason,
            'expected_stockout_date': self.expected_stockout_date.isoformat() if self.expected_stockout_date else None,
            'confidence_score': self.confidence_score,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'approved_by': self.approved_by
        }
    
    def __repr__(self) -> str:
        return f"<ReorderRecommendation(id={self.id}, product_id={self.product_id}, qty={self.recommended_quantity}, priority='{self.priority}')>"
