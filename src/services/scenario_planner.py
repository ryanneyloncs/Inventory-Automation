"""
Scenario Planning & What-If Analysis Service

Enables business users to model different scenarios:
- Promotions and discounts
- New product launches
- Seasonal events
- Custom business scenarios
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging

from src.models.database import SalesRecord, Product
from src.models.sales_forecasting_models import (
    ForecastScenario, ScenarioForecastResult, SalesForecastResult
)
from .multi_horizon_forecaster import MultiHorizonForecaster

logger = logging.getLogger(__name__)


class ScenarioPlanner:
    """
    Scenario planning and what-if analysis engine
    """

    SCENARIO_TYPES = {
        'promotion': 'Promotional Campaign',
        'new_product': 'New Product Launch',
        'seasonal': 'Seasonal Event',
        'price_change': 'Price Adjustment',
        'custom': 'Custom Scenario'
    }

    def __init__(self, db: AsyncSession):
        self.db = db
        self.forecaster = MultiHorizonForecaster(db)

    async def create_scenario(
        self,
        name: str,
        scenario_type: str,
        start_date: date,
        end_date: date,
        parameters: Dict,
        description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> ForecastScenario:
        """
        Create a new scenario for analysis

        Args:
            name: Scenario name
            scenario_type: Type of scenario (promotion, new_product, seasonal, price_change, custom)
            start_date: Scenario start date
            end_date: Scenario end date
            parameters: Scenario-specific parameters
            description: Optional description
            created_by: User creating the scenario

        Returns:
            Created ForecastScenario object
        """
        if scenario_type not in self.SCENARIO_TYPES:
            raise ValueError(f"Invalid scenario type. Must be one of: {list(self.SCENARIO_TYPES.keys())}")

        scenario = ForecastScenario(
            name=name,
            description=description,
            scenario_type=scenario_type,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters,
            created_by=created_by,
            is_active=True
        )

        self.db.add(scenario)
        await self.db.commit()
        await self.db.refresh(scenario)

        logger.info(f"Created scenario: {name} (ID: {scenario.id})")
        return scenario

    async def analyze_promotion_scenario(
        self,
        scenario_id: int,
        product_ids: List[int],
        discount_pct: float,
        expected_lift_pct: Optional[float] = None
    ) -> Dict:
        """
        Analyze impact of a promotional campaign

        Args:
            scenario_id: Scenario ID
            product_ids: Products affected by promotion
            discount_pct: Discount percentage (e.g., 20 for 20% off)
            expected_lift_pct: Expected sales lift (if None, estimated based on discount)

        Returns:
            Dict with baseline vs scenario comparison
        """
        scenario = await self._get_scenario(scenario_id)

        # Estimate sales lift if not provided
        if expected_lift_pct is None:
            # Rule of thumb: 1% discount = 0.5-2% sales lift
            # Using conservative 1.5x multiplier
            expected_lift_pct = discount_pct * 1.5

        results = {
            'scenario_id': scenario_id,
            'scenario_name': scenario.name,
            'products': [],
            'aggregate': {
                'baseline_sales': 0,
                'scenario_sales': 0,
                'sales_lift': 0,
                'baseline_revenue': 0,
                'scenario_revenue': 0,
                'incremental_revenue': 0
            }
        }

        for product_id in product_ids:
            product_result = await self._analyze_product_promotion(
                scenario=scenario,
                product_id=product_id,
                discount_pct=discount_pct,
                lift_pct=expected_lift_pct
            )
            results['products'].append(product_result)

            # Aggregate
            results['aggregate']['baseline_sales'] += product_result['baseline_sales']
            results['aggregate']['scenario_sales'] += product_result['scenario_sales']
            results['aggregate']['baseline_revenue'] += product_result['baseline_revenue']
            results['aggregate']['scenario_revenue'] += product_result['scenario_revenue']

        results['aggregate']['sales_lift'] = (
            (results['aggregate']['scenario_sales'] / results['aggregate']['baseline_sales'] - 1) * 100
            if results['aggregate']['baseline_sales'] > 0 else 0
        )
        results['aggregate']['incremental_revenue'] = (
            results['aggregate']['scenario_revenue'] - results['aggregate']['baseline_revenue']
        )

        logger.info(f"Promotion scenario {scenario_id}: {results['aggregate']['incremental_revenue']:.2f} incremental revenue")
        return results

    async def _analyze_product_promotion(
        self,
        scenario: ForecastScenario,
        product_id: int,
        discount_pct: float,
        lift_pct: float
    ) -> Dict:
        """
        Analyze promotion impact for a single product
        """
        # Get baseline forecast
        baseline_forecast = await self._get_baseline_forecast(
            product_id=product_id,
            start_date=scenario.start_date,
            end_date=scenario.end_date,
            horizon='daily'
        )

        product = await self._get_product(product_id)

        # Calculate scenario sales (baseline + lift)
        scenario_sales = baseline_forecast['predicted_sales'] * (1 + lift_pct / 100)

        # Calculate revenue impact
        baseline_revenue = baseline_forecast['predicted_sales'] * product.unit_price
        discounted_price = product.unit_price * (1 - discount_pct / 100)
        scenario_revenue = scenario_sales * discounted_price

        total_baseline_sales = baseline_forecast['predicted_sales'].sum()
        total_scenario_sales = scenario_sales.sum()
        total_baseline_revenue = baseline_revenue.sum()
        total_scenario_revenue = scenario_revenue.sum()

        # Save scenario results
        for i, row in baseline_forecast.iterrows():
            result = ScenarioForecastResult(
                scenario_id=scenario.id,
                product_id=product_id,
                forecast_date=row['date'].date(),
                baseline_sales=float(row['predicted_sales']),
                scenario_sales=float(scenario_sales.iloc[i]),
                sales_lift=lift_pct,
                baseline_revenue=float(baseline_revenue.iloc[i]),
                scenario_revenue=float(scenario_revenue.iloc[i]),
                incremental_revenue=float(scenario_revenue.iloc[i] - baseline_revenue.iloc[i])
            )
            self.db.add(result)

        await self.db.commit()

        return {
            'product_id': product_id,
            'product_name': product.name,
            'baseline_sales': float(total_baseline_sales),
            'scenario_sales': float(total_scenario_sales),
            'sales_lift_pct': lift_pct,
            'baseline_revenue': float(total_baseline_revenue),
            'scenario_revenue': float(total_scenario_revenue),
            'incremental_revenue': float(total_scenario_revenue - total_baseline_revenue),
            'roi': float((total_scenario_revenue - total_baseline_revenue) / total_baseline_revenue * 100) if total_baseline_revenue > 0 else 0
        }

    async def analyze_new_product_launch(
        self,
        scenario_id: int,
        new_product_id: int,
        similar_product_ids: List[int],
        expected_sales_multiplier: float = 0.8
    ) -> Dict:
        """
        Analyze new product launch scenario using similar products as reference

        Args:
            scenario_id: Scenario ID
            new_product_id: New product being launched
            similar_product_ids: Existing similar products for benchmarking
            expected_sales_multiplier: Expected sales as % of similar products (0.8 = 80%)

        Returns:
            Dict with launch projections
        """
        scenario = await self._get_scenario(scenario_id)
        new_product = await self._get_product(new_product_id)

        # Get baseline forecasts for similar products
        similar_forecasts = []
        for prod_id in similar_product_ids:
            forecast = await self._get_baseline_forecast(
                product_id=prod_id,
                start_date=scenario.start_date,
                end_date=scenario.end_date,
                horizon='daily'
            )
            similar_forecasts.append(forecast['predicted_sales'])

        # Average sales of similar products
        avg_similar_sales = pd.concat(similar_forecasts, axis=1).mean(axis=1)

        # Projected sales for new product
        projected_sales = avg_similar_sales * expected_sales_multiplier
        projected_revenue = projected_sales * new_product.unit_price

        # Ramp-up curve (sales grow over first 30 days)
        days = len(projected_sales)
        if days > 30:
            ramp_up_curve = np.concatenate([
                np.linspace(0.3, 1.0, 30),  # 30-day ramp up
                np.ones(days - 30)  # Full sales after ramp
            ])
        else:
            ramp_up_curve = np.linspace(0.3, 1.0, days)

        projected_sales = projected_sales * ramp_up_curve
        projected_revenue = projected_revenue * ramp_up_curve

        result = {
            'scenario_id': scenario_id,
            'new_product_id': new_product_id,
            'new_product_name': new_product.name,
            'similar_products': similar_product_ids,
            'projected_total_sales': float(projected_sales.sum()),
            'projected_total_revenue': float(projected_revenue.sum()),
            'average_daily_sales': float(projected_sales.mean()),
            'peak_daily_sales': float(projected_sales.max()),
            'by_date': [
                {
                    'date': date.isoformat(),
                    'projected_sales': float(sales),
                    'projected_revenue': float(revenue)
                }
                for date, sales, revenue in zip(
                    pd.date_range(scenario.start_date, scenario.end_date),
                    projected_sales,
                    projected_revenue
                )
            ]
        }

        logger.info(f"New product launch scenario: {result['projected_total_revenue']:.2f} projected revenue")
        return result

    async def analyze_price_change_scenario(
        self,
        scenario_id: int,
        product_id: int,
        new_price: float,
        price_elasticity: Optional[float] = None
    ) -> Dict:
        """
        Analyze impact of price changes on sales and revenue

        Args:
            scenario_id: Scenario ID
            product_id: Product to analyze
            new_price: New price point
            price_elasticity: Price elasticity of demand (if None, estimated)

        Returns:
            Dict with price change analysis
        """
        scenario = await self._get_scenario(scenario_id)
        product = await self._get_product(product_id)

        # Estimate price elasticity if not provided
        if price_elasticity is None:
            # Default elasticity based on product category
            # Luxury: -2.5, Standard: -1.5, Essential: -0.8
            price_elasticity = -1.5  # Moderate elasticity

        # Get baseline forecast
        baseline_forecast = await self._get_baseline_forecast(
            product_id=product_id,
            start_date=scenario.start_date,
            end_date=scenario.end_date,
            horizon='daily'
        )

        # Calculate price change impact
        price_change_pct = (new_price - product.unit_price) / product.unit_price * 100
        demand_change_pct = price_elasticity * price_change_pct

        # Scenario sales and revenue
        scenario_sales = baseline_forecast['predicted_sales'] * (1 + demand_change_pct / 100)
        baseline_revenue = baseline_forecast['predicted_sales'] * product.unit_price
        scenario_revenue = scenario_sales * new_price

        result = {
            'scenario_id': scenario_id,
            'product_id': product_id,
            'product_name': product.name,
            'current_price': float(product.unit_price),
            'new_price': float(new_price),
            'price_change_pct': float(price_change_pct),
            'price_elasticity': float(price_elasticity),
            'baseline_sales': float(baseline_forecast['predicted_sales'].sum()),
            'scenario_sales': float(scenario_sales.sum()),
            'demand_change_pct': float(demand_change_pct),
            'baseline_revenue': float(baseline_revenue.sum()),
            'scenario_revenue': float(scenario_revenue.sum()),
            'revenue_change': float(scenario_revenue.sum() - baseline_revenue.sum()),
            'revenue_change_pct': float((scenario_revenue.sum() / baseline_revenue.sum() - 1) * 100) if baseline_revenue.sum() > 0 else 0
        }

        logger.info(f"Price change scenario: {result['revenue_change']:.2f} revenue impact")
        return result

    async def compare_scenarios(
        self,
        scenario_ids: List[int]
    ) -> Dict:
        """
        Compare multiple scenarios side-by-side

        Args:
            scenario_ids: List of scenario IDs to compare

        Returns:
            Dict with comparative analysis
        """
        comparison = {
            'scenarios': [],
            'comparison_metrics': {}
        }

        for scenario_id in scenario_ids:
            # Get scenario results
            query = select(ScenarioForecastResult).where(
                ScenarioForecastResult.scenario_id == scenario_id
            )
            result = await self.db.execute(query)
            results = result.scalars().all()

            if not results:
                continue

            scenario = await self._get_scenario(scenario_id)

            total_baseline_revenue = sum(r.baseline_revenue for r in results)
            total_scenario_revenue = sum(r.scenario_revenue for r in results)
            total_incremental_revenue = sum(r.incremental_revenue for r in results)

            comparison['scenarios'].append({
                'scenario_id': scenario_id,
                'scenario_name': scenario.name,
                'scenario_type': scenario.scenario_type,
                'baseline_revenue': float(total_baseline_revenue),
                'scenario_revenue': float(total_scenario_revenue),
                'incremental_revenue': float(total_incremental_revenue),
                'roi_pct': float((total_incremental_revenue / total_baseline_revenue * 100)) if total_baseline_revenue > 0 else 0
            })

        # Find best scenario
        if comparison['scenarios']:
            best_scenario = max(comparison['scenarios'], key=lambda x: x['incremental_revenue'])
            comparison['comparison_metrics']['best_scenario'] = best_scenario['scenario_name']
            comparison['comparison_metrics']['best_incremental_revenue'] = best_scenario['incremental_revenue']

        return comparison

    async def _get_baseline_forecast(
        self,
        product_id: int,
        start_date: date,
        end_date: date,
        horizon: str
    ) -> pd.DataFrame:
        """
        Get baseline forecast for comparison
        """
        # Try to get existing forecast from database
        query = select(SalesForecastResult).where(
            and_(
                SalesForecastResult.product_id == product_id,
                SalesForecastResult.forecast_date >= start_date,
                SalesForecastResult.forecast_date <= end_date,
                SalesForecastResult.horizon == horizon
            )
        )
        result = await self.db.execute(query)
        existing_forecasts = result.scalars().all()

        if existing_forecasts and len(existing_forecasts) > 0:
            # Use existing forecast
            df = pd.DataFrame([{
                'date': pd.to_datetime(f.forecast_date),
                'predicted_sales': f.predicted_sales,
                'predicted_revenue': f.predicted_revenue,
                'lower_bound': f.lower_bound,
                'upper_bound': f.upper_bound
            } for f in existing_forecasts])
            return df

        # Generate new forecast
        days = (end_date - start_date).days + 1
        forecast = await self.forecaster.forecast_single_horizon(
            product_id=product_id,
            horizon=horizon,
            periods=days
        )

        # Filter to date range
        forecast = forecast[
            (forecast['date'].dt.date >= start_date) &
            (forecast['date'].dt.date <= end_date)
        ]

        return forecast

    async def _get_scenario(self, scenario_id: int) -> ForecastScenario:
        """Get scenario by ID"""
        query = select(ForecastScenario).where(ForecastScenario.id == scenario_id)
        result = await self.db.execute(query)
        scenario = result.scalar_one_or_none()

        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")

        return scenario

    async def _get_product(self, product_id: int) -> Product:
        """Get product by ID"""
        query = select(Product).where(Product.id == product_id)
        result = await self.db.execute(query)
        product = result.scalar_one_or_none()

        if not product:
            raise ValueError(f"Product {product_id} not found")

        return product
