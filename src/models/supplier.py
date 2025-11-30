"""
Supplier Database Model
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any, List
from src.models.database import Base


class Supplier(Base):
    """Supplier information and performance tracking"""
    __tablename__ = 'suppliers'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    code = Column(String(50), unique=True, nullable=False, index=True)
    
    # Contact information
    contact_person = Column(String(100), nullable=True)
    email = Column(String(100), nullable=True)
    phone = Column(String(50), nullable=True)
    website = Column(String(200), nullable=True)
    
    # Address
    address_line1 = Column(String(200), nullable=True)
    address_line2 = Column(String(200), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)
    country = Column(String(100), nullable=False, default='USA')
    postal_code = Column(String(20), nullable=True)
    
    # Business terms
    payment_terms = Column(String(100), nullable=True)  # Net 30, Net 60, etc.
    minimum_order_value = Column(Float, nullable=True)
    currency = Column(String(10), nullable=False, default='USD')
    
    # Performance metrics
    average_lead_time_days = Column(Integer, nullable=True)
    on_time_delivery_rate = Column(Float, nullable=True)  # Percentage
    quality_rating = Column(Float, nullable=True)  # 0-5 scale
    reliability_score = Column(Float, nullable=True)  # 0-100 scale
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    is_preferred = Column(Boolean, nullable=False, default=False)
    risk_level = Column(String(20), nullable=True)  # low, medium, high
    
    # Metadata
    notes = Column(Text, nullable=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_order_date = Column(DateTime, nullable=True)
    
    # Relationships
    products = relationship("Product", back_populates="supplier")
    purchase_orders = relationship("PurchaseOrder", back_populates="supplier", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert supplier to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'code': self.code,
            'contact_person': self.contact_person,
            'email': self.email,
            'phone': self.phone,
            'website': self.website,
            'address': {
                'line1': self.address_line1,
                'line2': self.address_line2,
                'city': self.city,
                'state': self.state,
                'country': self.country,
                'postal_code': self.postal_code
            },
            'payment_terms': self.payment_terms,
            'minimum_order_value': self.minimum_order_value,
            'currency': self.currency,
            'performance': {
                'average_lead_time_days': self.average_lead_time_days,
                'on_time_delivery_rate': self.on_time_delivery_rate,
                'quality_rating': self.quality_rating,
                'reliability_score': self.reliability_score
            },
            'is_active': self.is_active,
            'is_preferred': self.is_preferred,
            'risk_level': self.risk_level,
            'notes': self.notes,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_order_date': self.last_order_date.isoformat() if self.last_order_date else None
        }
    
    @property
    def full_address(self) -> str:
        """Get formatted full address"""
        parts = [
            self.address_line1,
            self.address_line2,
            f"{self.city}, {self.state} {self.postal_code}" if self.city and self.state else None,
            self.country
        ]
        return ", ".join([p for p in parts if p])
    
    @property
    def performance_summary(self) -> str:
        """Get human-readable performance summary"""
        if not self.reliability_score:
            return "unrated"
        if self.reliability_score >= 90:
            return "excellent"
        elif self.reliability_score >= 75:
            return "good"
        elif self.reliability_score >= 60:
            return "fair"
        else:
            return "poor"
    
    def __repr__(self) -> str:
        return f"<Supplier(id={self.id}, name='{self.name}', code='{self.code}')>"


class PurchaseOrder(Base):
    """Purchase orders to suppliers"""
    __tablename__ = 'purchase_orders'
    
    id = Column(Integer, primary_key=True)
    order_number = Column(String(50), unique=True, nullable=False, index=True)
    supplier_id = Column(Integer, ForeignKey('suppliers.id', ondelete='CASCADE'), nullable=False)
    
    # Order details
    order_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    expected_delivery_date = Column(DateTime, nullable=True)
    actual_delivery_date = Column(DateTime, nullable=True)
    
    # Financial
    total_amount = Column(Float, nullable=False)
    currency = Column(String(10), nullable=False, default='USD')
    
    # Status
    status = Column(String(20), nullable=False, default='pending')  # pending, confirmed, shipped, delivered, cancelled
    payment_status = Column(String(20), nullable=False, default='unpaid')  # unpaid, partial, paid
    
    # Line items stored as JSONB for flexibility
    # Each item: {product_id, quantity, unit_cost, total_cost}
    line_items = Column(JSONB, nullable=False)
    
    # Tracking
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100), nullable=True)
    
    # Relationships
    supplier = relationship("Supplier", back_populates="purchase_orders")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'order_number': self.order_number,
            'supplier_id': self.supplier_id,
            'order_date': self.order_date.isoformat(),
            'expected_delivery_date': self.expected_delivery_date.isoformat() if self.expected_delivery_date else None,
            'actual_delivery_date': self.actual_delivery_date.isoformat() if self.actual_delivery_date else None,
            'total_amount': self.total_amount,
            'currency': self.currency,
            'status': self.status,
            'payment_status': self.payment_status,
            'line_items': self.line_items,
            'notes': self.notes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': self.created_by
        }
    
    @property
    def is_delayed(self) -> bool:
        """Check if order is delayed"""
        if not self.expected_delivery_date or self.status == 'delivered':
            return False
        return datetime.utcnow() > self.expected_delivery_date
    
    @property
    def days_until_delivery(self) -> int:
        """Calculate days until expected delivery"""
        if not self.expected_delivery_date or self.status == 'delivered':
            return 0
        delta = self.expected_delivery_date - datetime.utcnow()
        return max(0, delta.days)
    
    def __repr__(self) -> str:
        return f"<PurchaseOrder(id={self.id}, order_number='{self.order_number}', supplier_id={self.supplier_id}, status='{self.status}')>"
