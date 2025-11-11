from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON, Text, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.sql import func

Base = declarative_base()


class Supplier(Base):
    """Supplier information and contact details"""
    __tablename__ = "suppliers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True)
    contact_email = Column(String(200))
    contact_phone = Column(String(50))
    address = Column(Text)
    lead_time_days = Column(Integer, default=7)
    minimum_order_quantity = Column(Integer, default=1)
    api_endpoint = Column(String(500))
    api_key = Column(String(500))
    reliability_score = Column(Float, default=1.0)  # 0-1 scale
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    products = relationship("Product", back_populates="supplier")
    purchase_orders = relationship("PurchaseOrder", back_populates="supplier")


class Product(Base):
    """Product catalog with supplier and cost information"""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(100), index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    unit_cost = Column(Numeric(10, 2), nullable=False)
    selling_price = Column(Numeric(10, 2))
    weight = Column(Float)  # in kg
    dimensions = Column(JSON)  # {"length": x, "width": y, "height": z}
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    supplier = relationship("Supplier", back_populates="products")
    inventory = relationship("InventoryLevel", back_populates="product", uselist=False)
    sales = relationship("SalesRecord", back_populates="product")
    forecasts = relationship("ForecastResult", back_populates="product")
    optimization_params = relationship("OptimizationParameters", back_populates="product", uselist=False)
    purchase_order_items = relationship("PurchaseOrderItem", back_populates="product")


class InventoryLevel(Base):
    """Current inventory levels and locations"""
    __tablename__ = "inventory_levels"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, unique=True)
    quantity_on_hand = Column(Integer, default=0, nullable=False)
    quantity_reserved = Column(Integer, default=0)
    quantity_available = Column(Integer, default=0)
    warehouse_location = Column(String(100))
    last_counted_at = Column(DateTime)
    last_reorder_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="inventory")


class SalesRecord(Base):
    """Historical sales data for forecasting"""
    __tablename__ = "sales_records"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    sale_date = Column(DateTime, nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    revenue = Column(Numeric(10, 2))
    customer_segment = Column(String(50))
    sales_channel = Column(String(50))  # online, retail, wholesale
    promotion_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="sales")


class ForecastResult(Base):
    """ML model forecast outputs"""
    __tablename__ = "forecast_results"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    forecast_date = Column(DateTime, nullable=False, index=True)
    predicted_demand = Column(Float, nullable=False)
    lower_bound = Column(Float)
    upper_bound = Column(Float)
    confidence = Column(Float)
    model_used = Column(String(50))  # prophet, lstm, arima, ensemble
    model_version = Column(String(50))
    features_used = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="forecasts")


class OptimizationParameters(Base):
    """Calculated optimization parameters for each product"""
    __tablename__ = "optimization_parameters"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, unique=True)
    economic_order_quantity = Column(Integer)
    reorder_point = Column(Integer)
    safety_stock = Column(Integer)
    max_stock_level = Column(Integer)
    holding_cost_per_unit = Column(Numeric(10, 2))
    ordering_cost = Column(Numeric(10, 2))
    stockout_cost = Column(Numeric(10, 2))
    service_level = Column(Float, default=0.95)
    lead_time_days = Column(Integer)
    demand_std_dev = Column(Float)
    last_calculated_at = Column(DateTime, default=datetime.utcnow)
    calculation_method = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="optimization_params")


class PurchaseOrder(Base):
    """Purchase orders to suppliers"""
    __tablename__ = "purchase_orders"
    
    id = Column(Integer, primary_key=True, index=True)
    order_number = Column(String(100), unique=True, nullable=False, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    order_date = Column(DateTime, default=datetime.utcnow)
    expected_delivery_date = Column(DateTime)
    actual_delivery_date = Column(DateTime)
    status = Column(String(50), default="pending")  # pending, sent, confirmed, received, cancelled
    total_amount = Column(Numeric(10, 2))
    notes = Column(Text)
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    supplier = relationship("Supplier", back_populates="purchase_orders")
    items = relationship("PurchaseOrderItem", back_populates="purchase_order", cascade="all, delete-orphan")


class PurchaseOrderItem(Base):
    """Individual items in a purchase order"""
    __tablename__ = "purchase_order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    purchase_order_id = Column(Integer, ForeignKey("purchase_orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity_ordered = Column(Integer, nullable=False)
    quantity_received = Column(Integer, default=0)
    unit_price = Column(Numeric(10, 2), nullable=False)
    line_total = Column(Numeric(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    purchase_order = relationship("PurchaseOrder", back_populates="items")
    product = relationship("Product", back_populates="purchase_order_items")


class AlertLog(Base):
    """System alerts and notifications"""
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(50), nullable=False)  # low_stock, stockout, reorder_triggered, delivery_delay
    severity = Column(String(20), default="info")  # info, warning, error, critical
    product_id = Column(Integer, ForeignKey("products.id"))
    message = Column(Text, nullable=False)
    additional_data = Column(JSON)  # Renamed from 'metadata' to avoid reserved keyword conflict
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelPerformanceMetric(Base):
    """Track ML model performance over time"""
    __tablename__ = "model_performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(50))
    metric_name = Column(String(50), nullable=False)  # mae, rmse, mape, accuracy
    metric_value = Column(Float, nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"))
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    training_samples = Column(Integer)
    test_samples = Column(Integer)
    additional_data = Column(JSON)  # Renamed from 'metadata' to avoid reserved keyword conflict
    created_at = Column(DateTime, default=datetime.utcnow)
