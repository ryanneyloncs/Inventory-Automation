"""Initial schema with all tables

Revision ID: 03b47965adc7
Revises:
Create Date: 2025-11-11 12:48:25.316041

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '03b47965adc7'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create suppliers table
    op.create_table('suppliers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('contact_email', sa.String(length=200), nullable=True),
        sa.Column('contact_phone', sa.String(length=50), nullable=True),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('lead_time_days', sa.Integer(), nullable=True),
        sa.Column('minimum_order_quantity', sa.Integer(), nullable=True),
        sa.Column('api_endpoint', sa.String(length=500), nullable=True),
        sa.Column('api_key', sa.String(length=500), nullable=True),
        sa.Column('reliability_score', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_suppliers_id'), 'suppliers', ['id'], unique=False)

    # Create products table
    op.create_table('products',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sku', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('supplier_id', sa.Integer(), nullable=False),
        sa.Column('unit_cost', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('selling_price', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('weight', sa.Float(), nullable=True),
        sa.Column('dimensions', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['supplier_id'], ['suppliers.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('sku')
    )
    op.create_index(op.f('ix_products_category'), 'products', ['category'], unique=False)
    op.create_index(op.f('ix_products_id'), 'products', ['id'], unique=False)
    op.create_index(op.f('ix_products_sku'), 'products', ['sku'], unique=False)

    # Create inventory_levels table
    op.create_table('inventory_levels',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('quantity_on_hand', sa.Integer(), nullable=False),
        sa.Column('quantity_reserved', sa.Integer(), nullable=True),
        sa.Column('quantity_available', sa.Integer(), nullable=True),
        sa.Column('warehouse_location', sa.String(length=100), nullable=True),
        sa.Column('last_counted_at', sa.DateTime(), nullable=True),
        sa.Column('last_reorder_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('product_id')
    )
    op.create_index(op.f('ix_inventory_levels_id'), 'inventory_levels', ['id'], unique=False)

    # Create sales_records table
    op.create_table('sales_records',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('sale_date', sa.DateTime(), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('revenue', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('customer_segment', sa.String(length=50), nullable=True),
        sa.Column('sales_channel', sa.String(length=50), nullable=True),
        sa.Column('promotion_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_sales_records_id'), 'sales_records', ['id'], unique=False)
    op.create_index(op.f('ix_sales_records_product_id'), 'sales_records', ['product_id'], unique=False)
    op.create_index(op.f('ix_sales_records_sale_date'), 'sales_records', ['sale_date'], unique=False)

    # Create forecast_results table
    op.create_table('forecast_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('forecast_date', sa.DateTime(), nullable=False),
        sa.Column('forecasted_quantity', sa.Float(), nullable=False),
        sa.Column('lower_bound', sa.Float(), nullable=True),
        sa.Column('upper_bound', sa.Float(), nullable=True),
        sa.Column('confidence_level', sa.Float(), nullable=True),
        sa.Column('model_used', sa.String(length=50), nullable=True),
        sa.Column('model_accuracy', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_forecast_results_forecast_date'), 'forecast_results', ['forecast_date'], unique=False)
    op.create_index(op.f('ix_forecast_results_id'), 'forecast_results', ['id'], unique=False)
    op.create_index(op.f('ix_forecast_results_product_id'), 'forecast_results', ['product_id'], unique=False)

    # Create optimization_parameters table
    op.create_table('optimization_parameters',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('economic_order_quantity', sa.Integer(), nullable=True),
        sa.Column('reorder_point', sa.Integer(), nullable=True),
        sa.Column('safety_stock', sa.Integer(), nullable=True),
        sa.Column('max_stock_level', sa.Integer(), nullable=True),
        sa.Column('holding_cost_per_unit', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('ordering_cost', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('stockout_cost', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('service_level', sa.Float(), nullable=True),
        sa.Column('lead_time_days', sa.Integer(), nullable=True),
        sa.Column('demand_std_dev', sa.Float(), nullable=True),
        sa.Column('calculation_method', sa.String(length=50), nullable=True),
        sa.Column('last_optimized_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('product_id')
    )
    op.create_index(op.f('ix_optimization_parameters_id'), 'optimization_parameters', ['id'], unique=False)

    # Create purchase_orders table
    op.create_table('purchase_orders',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('order_number', sa.String(length=100), nullable=False),
        sa.Column('supplier_id', sa.Integer(), nullable=False),
        sa.Column('order_date', sa.DateTime(), nullable=True),
        sa.Column('expected_delivery_date', sa.DateTime(), nullable=True),
        sa.Column('actual_delivery_date', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('total_cost', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('total_items', sa.Integer(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['supplier_id'], ['suppliers.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order_number')
    )
    op.create_index(op.f('ix_purchase_orders_id'), 'purchase_orders', ['id'], unique=False)
    op.create_index(op.f('ix_purchase_orders_order_number'), 'purchase_orders', ['order_number'], unique=False)

    # Create purchase_order_items table
    op.create_table('purchase_order_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('purchase_order_id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('quantity_ordered', sa.Integer(), nullable=False),
        sa.Column('quantity_received', sa.Integer(), nullable=True),
        sa.Column('unit_price', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('line_total', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('received_date', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ),
        sa.ForeignKeyConstraint(['purchase_order_id'], ['purchase_orders.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_purchase_order_items_id'), 'purchase_order_items', ['id'], unique=False)

    # Create alert_logs table
    op.create_table('alert_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('alert_type', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=True),
        sa.Column('product_id', sa.Integer(), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('additional_data', sa.JSON(), nullable=True),
        sa.Column('acknowledged', sa.Boolean(), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(), nullable=True),
        sa.Column('acknowledged_by', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_alert_logs_id'), 'alert_logs', ['id'], unique=False)

    # Create model_performance_metrics table
    op.create_table('model_performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(length=50), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('metric_name', sa.String(length=50), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=True),
        sa.Column('evaluation_date', sa.DateTime(), nullable=True),
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('test_samples', sa.Integer(), nullable=True),
        sa.Column('additional_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_performance_metrics_id'), 'model_performance_metrics', ['id'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index(op.f('ix_model_performance_metrics_id'), table_name='model_performance_metrics')
    op.drop_table('model_performance_metrics')

    op.drop_index(op.f('ix_alert_logs_id'), table_name='alert_logs')
    op.drop_table('alert_logs')

    op.drop_index(op.f('ix_purchase_order_items_id'), table_name='purchase_order_items')
    op.drop_table('purchase_order_items')

    op.drop_index(op.f('ix_purchase_orders_order_number'), table_name='purchase_orders')
    op.drop_index(op.f('ix_purchase_orders_id'), table_name='purchase_orders')
    op.drop_table('purchase_orders')

    op.drop_index(op.f('ix_optimization_parameters_id'), table_name='optimization_parameters')
    op.drop_table('optimization_parameters')

    op.drop_index(op.f('ix_forecast_results_product_id'), table_name='forecast_results')
    op.drop_index(op.f('ix_forecast_results_id'), table_name='forecast_results')
    op.drop_index(op.f('ix_forecast_results_forecast_date'), table_name='forecast_results')
    op.drop_table('forecast_results')

    op.drop_index(op.f('ix_sales_records_sale_date'), table_name='sales_records')
    op.drop_index(op.f('ix_sales_records_product_id'), table_name='sales_records')
    op.drop_index(op.f('ix_sales_records_id'), table_name='sales_records')
    op.drop_table('sales_records')

    op.drop_index(op.f('ix_inventory_levels_id'), table_name='inventory_levels')
    op.drop_table('inventory_levels')

    op.drop_index(op.f('ix_products_sku'), table_name='products')
    op.drop_index(op.f('ix_products_id'), table_name='products')
    op.drop_index(op.f('ix_products_category'), table_name='products')
    op.drop_table('products')

    op.drop_index(op.f('ix_suppliers_id'), table_name='suppliers')
    op.drop_table('suppliers')
