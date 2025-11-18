"""Add sales forecasting suite tables

Revision ID: 002_sales_forecasting
Revises: 001_initial
Create Date: 2025-11-12

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '002_sales_forecasting'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade():
    # Sales Forecast Results - Multi-horizon forecasts
    op.create_table(
        'sales_forecast_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('forecast_date', sa.Date(), nullable=False),
        sa.Column('horizon', sa.String(20), nullable=False),  # hourly, daily, weekly, monthly
        sa.Column('predicted_sales', sa.Float(), nullable=False),
        sa.Column('predicted_revenue', sa.Float(), nullable=False),
        sa.Column('lower_bound', sa.Float(), nullable=False),
        sa.Column('upper_bound', sa.Float(), nullable=False),
        sa.Column('confidence_level', sa.Float(), nullable=False, server_default='0.95'),
        sa.Column('model_used', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_sales_forecast_product_date', 'sales_forecast_results', ['product_id', 'forecast_date'])
    op.create_index('idx_sales_forecast_horizon', 'sales_forecast_results', ['horizon'])

    # Forecast Scenarios - What-if analysis
    op.create_table(
        'forecast_scenarios',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('scenario_type', sa.String(50), nullable=False),  # promotion, new_product, seasonal, custom
        sa.Column('start_date', sa.Date(), nullable=False),
        sa.Column('end_date', sa.Date(), nullable=False),
        sa.Column('parameters', postgresql.JSONB(), nullable=False),  # scenario-specific params
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_scenario_type', 'forecast_scenarios', ['scenario_type'])
    op.create_index('idx_scenario_active', 'forecast_scenarios', ['is_active'])

    # Scenario Forecast Results
    op.create_table(
        'scenario_forecast_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('scenario_id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('forecast_date', sa.Date(), nullable=False),
        sa.Column('baseline_sales', sa.Float(), nullable=False),
        sa.Column('scenario_sales', sa.Float(), nullable=False),
        sa.Column('sales_lift', sa.Float(), nullable=False),  # percentage change
        sa.Column('baseline_revenue', sa.Float(), nullable=False),
        sa.Column('scenario_revenue', sa.Float(), nullable=False),
        sa.Column('incremental_revenue', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['scenario_id'], ['forecast_scenarios.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_scenario_forecast_scenario', 'scenario_forecast_results', ['scenario_id'])
    op.create_index('idx_scenario_forecast_product', 'scenario_forecast_results', ['product_id'])

    # Seasonal Decomposition Results
    op.create_table(
        'seasonal_decomposition',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('observed', sa.Float(), nullable=False),
        sa.Column('trend', sa.Float(), nullable=False),
        sa.Column('seasonal', sa.Float(), nullable=False),
        sa.Column('residual', sa.Float(), nullable=False),
        sa.Column('decomposition_type', sa.String(20), nullable=False),  # additive, multiplicative
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_seasonal_product_date', 'seasonal_decomposition', ['product_id', 'date'])

    # Event Impact Tracking
    op.create_table(
        'event_impacts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('event_name', sa.String(200), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),  # holiday, promotion, external, custom
        sa.Column('event_date', sa.Date(), nullable=False),
        sa.Column('impact_start_date', sa.Date(), nullable=False),
        sa.Column('impact_end_date', sa.Date(), nullable=False),
        sa.Column('affected_products', postgresql.ARRAY(sa.Integer()), nullable=True),  # array of product IDs
        sa.Column('baseline_sales', sa.Float(), nullable=True),
        sa.Column('actual_sales', sa.Float(), nullable=True),
        sa.Column('sales_lift_pct', sa.Float(), nullable=True),
        sa.Column('revenue_impact', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_event_date', 'event_impacts', ['event_date'])
    op.create_index('idx_event_type', 'event_impacts', ['event_type'])

    # Cannibalization Analysis
    op.create_table(
        'cannibalization_analysis',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('new_product_id', sa.Integer(), nullable=False),
        sa.Column('existing_product_id', sa.Integer(), nullable=False),
        sa.Column('analysis_date', sa.Date(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),  # 0-1, product similarity
        sa.Column('baseline_sales_existing', sa.Float(), nullable=False),
        sa.Column('post_launch_sales_existing', sa.Float(), nullable=False),
        sa.Column('cannibalization_rate', sa.Float(), nullable=False),  # percentage cannibalized
        sa.Column('new_product_sales', sa.Float(), nullable=False),
        sa.Column('net_incremental_sales', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['new_product_id'], ['products.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['existing_product_id'], ['products.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_cannibalization_new_product', 'cannibalization_analysis', ['new_product_id'])
    op.create_index('idx_cannibalization_existing_product', 'cannibalization_analysis', ['existing_product_id'])

    # Forecast Accuracy Metrics - Track performance by horizon
    op.create_table(
        'forecast_accuracy_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=True),  # NULL for aggregate metrics
        sa.Column('horizon', sa.String(20), nullable=False),
        sa.Column('metric_date', sa.Date(), nullable=False),
        sa.Column('mae', sa.Float(), nullable=True),  # Mean Absolute Error
        sa.Column('rmse', sa.Float(), nullable=True),  # Root Mean Squared Error
        sa.Column('mape', sa.Float(), nullable=True),  # Mean Absolute Percentage Error
        sa.Column('bias', sa.Float(), nullable=True),  # Forecast bias
        sa.Column('forecast_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_accuracy_product_horizon', 'forecast_accuracy_metrics', ['product_id', 'horizon'])


def downgrade():
    op.drop_table('forecast_accuracy_metrics')
    op.drop_table('cannibalization_analysis')
    op.drop_table('event_impacts')
    op.drop_table('seasonal_decomposition')
    op.drop_table('scenario_forecast_results')
    op.drop_table('forecast_scenarios')
    op.drop_table('sales_forecast_results')
