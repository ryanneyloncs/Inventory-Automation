import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from src.models.database import (
    Product, SalesRecord, InventoryLevel, OptimizationParameters,
    ForecastResult, Supplier
)
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """Complete inventory optimization algorithms"""
    
    def __init__(self):
        self.holding_cost_rate = settings.HOLDING_COST_PERCENTAGE
        self.ordering_cost = settings.ORDERING_COST
        self.service_level = settings.SERVICE_LEVEL
        self.lead_time_days = settings.LEAD_TIME_DAYS
        
    def calculate_economic_order_quantity(
        self,
        annual_demand: float,
        ordering_cost: float,
        holding_cost_per_unit: float
    ) -> int:
        """
        Calculate Economic Order Quantity (EOQ)
        
        EOQ = sqrt((2 * D * S) / H)
        where:
        D = Annual demand
        S = Ordering cost per order
        H = Holding cost per unit per year
        """
        if holding_cost_per_unit <= 0:
            return 0
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        return max(1, int(np.ceil(eoq)))
    
    def calculate_reorder_point(
        self,
        avg_daily_demand: float,
        lead_time_days: int,
        safety_stock: int
    ) -> int:
        """
        Calculate Reorder Point (ROP)
        
        ROP = (Average Daily Demand × Lead Time) + Safety Stock
        """
        lead_time_demand = avg_daily_demand * lead_time_days
        rop = lead_time_demand + safety_stock
        return max(0, int(np.ceil(rop)))
    
    def calculate_safety_stock(
        self,
        demand_std_dev: float,
        lead_time_days: int,
        service_level: float = 0.95
    ) -> int:
        """
        Calculate Safety Stock using normal distribution
        
        Safety Stock = Z × σ_LT
        where:
        Z = Z-score for desired service level
        σ_LT = Standard deviation of demand during lead time
        """
        # Get Z-score for service level
        z_score = stats.norm.ppf(service_level)
        
        # Standard deviation during lead time
        lead_time_std = demand_std_dev * np.sqrt(lead_time_days)
        
        safety_stock = z_score * lead_time_std
        return max(0, int(np.ceil(safety_stock)))
    
    def calculate_max_stock_level(
        self,
        reorder_point: int,
        eoq: int
    ) -> int:
        """Calculate maximum stock level"""
        return reorder_point + eoq
    
    def calculate_optimal_service_level(
        self,
        stockout_cost: float,
        holding_cost_per_unit: float,
        avg_daily_demand: float
    ) -> float:
        """
        Calculate optimal service level based on costs
        
        This balances the cost of stockouts vs. carrying inventory
        """
        if holding_cost_per_unit <= 0:
            return 0.95
        
        # Cost ratio
        critical_ratio = stockout_cost / (stockout_cost + holding_cost_per_unit)
        
        # Ensure it's within reasonable bounds
        service_level = min(0.99, max(0.80, critical_ratio))
        
        return service_level
    
    def calculate_order_frequency(
        self,
        eoq: int,
        annual_demand: float
    ) -> Dict:
        """Calculate optimal ordering frequency"""
        if eoq <= 0:
            return {'orders_per_year': 0, 'days_between_orders': 0}
        
        orders_per_year = annual_demand / eoq
        days_between_orders = 365 / orders_per_year if orders_per_year > 0 else 0
        
        return {
            'orders_per_year': round(orders_per_year, 2),
            'days_between_orders': round(days_between_orders, 2)
        }
    
    def calculate_total_inventory_cost(
        self,
        annual_demand: float,
        order_quantity: int,
        ordering_cost: float,
        holding_cost_per_unit: float
    ) -> float:
        """Calculate total annual inventory cost"""
        if order_quantity <= 0:
            return float('inf')
        
        # Ordering costs
        num_orders = annual_demand / order_quantity
        annual_ordering_cost = num_orders * ordering_cost
        
        # Holding costs
        avg_inventory = order_quantity / 2
        annual_holding_cost = avg_inventory * holding_cost_per_unit
        
        total_cost = annual_ordering_cost + annual_holding_cost
        return total_cost
    
    def optimize_newsvendor_model(
        self,
        demand_mean: float,
        demand_std: float,
        unit_cost: float,
        selling_price: float,
        salvage_value: float = 0
    ) -> int:
        """
        Newsvendor model for single-period optimization
        Useful for perishable or seasonal items
        """
        if selling_price <= unit_cost:
            return 0
        
        # Critical ratio
        critical_ratio = (selling_price - unit_cost) / (selling_price - salvage_value)
        
        # Find optimal order quantity
        z = stats.norm.ppf(critical_ratio)
        optimal_quantity = demand_mean + (z * demand_std)
        
        return max(0, int(np.ceil(optimal_quantity)))
    
    def calculate_fill_rate(
        self,
        safety_stock: int,
        demand_std: float,
        eoq: int
    ) -> float:
        """Calculate expected fill rate (percentage of demand met from stock)"""
        if demand_std <= 0 or eoq <= 0:
            return 1.0
        
        # Simplified calculation
        z = safety_stock / (demand_std + 1e-6)
        fill_rate = stats.norm.cdf(z)
        
        return min(1.0, max(0.0, fill_rate))
    
    def abc_classification(
        self,
        annual_revenue: float,
        total_revenue: float
    ) -> str:
        """
        ABC classification based on revenue contribution
        A items: Top 80% of revenue (usually 20% of items)
        B items: Next 15% of revenue
        C items: Last 5% of revenue
        """
        revenue_percentage = (annual_revenue / total_revenue * 100) if total_revenue > 0 else 0
        
        if revenue_percentage >= 80:
            return 'A'
        elif revenue_percentage >= 15:
            return 'B'
        else:
            return 'C'
    
    def optimize_multi_product_constraint(
        self,
        products_data: list,
        budget_constraint: float = None,
        space_constraint: float = None
    ) -> Dict:
        """
        Optimize ordering for multiple products with constraints
        Uses linear programming approach
        """
        # This is a simplified version
        # In production, you'd use scipy.optimize.linprog or similar
        
        optimized_orders = []
        total_cost = 0
        total_space = 0
        
        # Sort by profitability
        products_sorted = sorted(
            products_data,
            key=lambda x: x.get('profit_margin', 0),
            reverse=True
        )
        
        for product in products_sorted:
            eoq = product['eoq']
            unit_cost = product['unit_cost']
            space_per_unit = product.get('space_per_unit', 1)
            
            order_cost = eoq * unit_cost
            order_space = eoq * space_per_unit
            
            # Check constraints
            if budget_constraint and (total_cost + order_cost) > budget_constraint:
                continue
            if space_constraint and (total_space + order_space) > space_constraint:
                continue
            
            optimized_orders.append({
                'product_id': product['product_id'],
                'order_quantity': eoq
            })
            
            total_cost += order_cost
            total_space += order_space
        
        return {
            'orders': optimized_orders,
            'total_cost': total_cost,
            'total_space': total_space
        }


class OptimizationService:
    """Main optimization service with database integration"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.optimizer = InventoryOptimizer()
    
    async def get_demand_statistics(
        self,
        product_id: int,
        days_back: int = 365
    ) -> Dict:
        """Calculate demand statistics from historical data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        query = select(SalesRecord).where(
            and_(
                SalesRecord.product_id == product_id,
                SalesRecord.sale_date >= cutoff_date
            )
        )
        
        result = await self.db.execute(query)
        sales_records = result.scalars().all()
        
        if not sales_records:
            return {
                'annual_demand': 0,
                'avg_daily_demand': 0,
                'demand_std_dev': 0,
                'total_revenue': 0
            }
        
        # Calculate statistics
        daily_demand = {}
        total_quantity = 0
        total_revenue = 0
        
        for record in sales_records:
            date_key = record.sale_date.date()
            daily_demand[date_key] = daily_demand.get(date_key, 0) + record.quantity
            total_quantity += record.quantity
            total_revenue += float(record.revenue) if record.revenue else 0
        
        # Daily statistics
        daily_quantities = list(daily_demand.values())
        avg_daily_demand = np.mean(daily_quantities) if daily_quantities else 0
        demand_std_dev = np.std(daily_quantities) if len(daily_quantities) > 1 else 0
        
        # Annualize
        days_of_data = len(daily_quantities)
        if days_of_data > 0:
            annual_demand = (total_quantity / days_of_data) * 365
        else:
            annual_demand = 0
        
        return {
            'annual_demand': annual_demand,
            'avg_daily_demand': avg_daily_demand,
            'demand_std_dev': demand_std_dev,
            'total_revenue': total_revenue,
            'days_of_data': days_of_data
        }
    
    async def get_product_costs(self, product_id: int) -> Dict:
        """Get cost information for a product"""
        query = select(Product).where(Product.id == product_id)
        result = await self.db.execute(query)
        product = result.scalar_one_or_none()
        
        if not product:
            return {}
        
        unit_cost = float(product.unit_cost)
        holding_cost_per_unit = unit_cost * self.optimizer.holding_cost_rate
        
        # Stockout cost (estimated as 2x the profit margin)
        selling_price = float(product.selling_price) if product.selling_price else unit_cost * 1.5
        profit_margin = selling_price - unit_cost
        stockout_cost = profit_margin * settings.STOCKOUT_COST_MULTIPLIER
        
        return {
            'unit_cost': unit_cost,
            'selling_price': selling_price,
            'holding_cost_per_unit': holding_cost_per_unit,
            'stockout_cost': stockout_cost,
            'ordering_cost': self.optimizer.ordering_cost
        }
    
    async def get_supplier_lead_time(self, product_id: int) -> int:
        """Get lead time from supplier"""
        query = select(Product, Supplier).join(
            Supplier, Product.supplier_id == Supplier.id
        ).where(Product.id == product_id)
        
        result = await self.db.execute(query)
        row = result.first()
        
        if row and row.Supplier:
            return row.Supplier.lead_time_days
        
        return self.optimizer.lead_time_days
    
    async def optimize_product(self, product_id: int) -> Dict:
        """Complete optimization for a single product"""
        logger.info(f"Optimizing parameters for product {product_id}")
        
        # Get demand statistics
        demand_stats = await self.get_demand_statistics(product_id)
        
        if demand_stats['annual_demand'] == 0:
            return {
                'success': False,
                'error': 'No demand data available'
            }
        
        # Get cost information
        costs = await self.get_product_costs(product_id)
        
        if not costs:
            return {
                'success': False,
                'error': 'Product not found'
            }
        
        # Get lead time
        lead_time = await self.get_supplier_lead_time(product_id)
        
        # Calculate EOQ
        eoq = self.optimizer.calculate_economic_order_quantity(
            annual_demand=demand_stats['annual_demand'],
            ordering_cost=costs['ordering_cost'],
            holding_cost_per_unit=costs['holding_cost_per_unit']
        )
        
        # Calculate optimal service level
        service_level = self.optimizer.calculate_optimal_service_level(
            stockout_cost=costs['stockout_cost'],
            holding_cost_per_unit=costs['holding_cost_per_unit'],
            avg_daily_demand=demand_stats['avg_daily_demand']
        )
        
        # Calculate safety stock
        safety_stock = self.optimizer.calculate_safety_stock(
            demand_std_dev=demand_stats['demand_std_dev'],
            lead_time_days=lead_time,
            service_level=service_level
        )
        
        # Calculate reorder point
        reorder_point = self.optimizer.calculate_reorder_point(
            avg_daily_demand=demand_stats['avg_daily_demand'],
            lead_time_days=lead_time,
            safety_stock=safety_stock
        )
        
        # Calculate max stock level
        max_stock = self.optimizer.calculate_max_stock_level(
            reorder_point=reorder_point,
            eoq=eoq
        )
        
        # Calculate order frequency
        order_frequency = self.optimizer.calculate_order_frequency(
            eoq=eoq,
            annual_demand=demand_stats['annual_demand']
        )
        
        # Calculate total cost
        total_cost = self.optimizer.calculate_total_inventory_cost(
            annual_demand=demand_stats['annual_demand'],
            order_quantity=eoq,
            ordering_cost=costs['ordering_cost'],
            holding_cost_per_unit=costs['holding_cost_per_unit']
        )
        
        # Calculate fill rate
        fill_rate = self.optimizer.calculate_fill_rate(
            safety_stock=safety_stock,
            demand_std=demand_stats['demand_std_dev'],
            eoq=eoq
        )
        
        optimization_result = {
            'product_id': product_id,
            'economic_order_quantity': eoq,
            'reorder_point': reorder_point,
            'safety_stock': safety_stock,
            'max_stock_level': max_stock,
            'service_level': service_level,
            'lead_time_days': lead_time,
            'order_frequency': order_frequency,
            'annual_inventory_cost': round(total_cost, 2),
            'expected_fill_rate': round(fill_rate * 100, 2),
            'demand_statistics': demand_stats,
            'cost_parameters': costs
        }
        
        # Save to database
        await self.save_optimization_parameters(product_id, optimization_result)
        
        logger.info(f"Optimization completed for product {product_id}")
        
        return {
            'success': True,
            'result': optimization_result
        }
    
    async def save_optimization_parameters(self, product_id: int, result: Dict):
        """Save optimization parameters to database"""
        # Check if parameters exist
        query = select(OptimizationParameters).where(
            OptimizationParameters.product_id == product_id
        )
        db_result = await self.db.execute(query)
        existing = db_result.scalar_one_or_none()
        
        if existing:
            # Update existing
            existing.economic_order_quantity = result['economic_order_quantity']
            existing.reorder_point = result['reorder_point']
            existing.safety_stock = result['safety_stock']
            existing.max_stock_level = result['max_stock_level']
            existing.service_level = result['service_level']
            existing.lead_time_days = result['lead_time_days']
            existing.demand_std_dev = result['demand_statistics']['demand_std_dev']
            existing.holding_cost_per_unit = result['cost_parameters']['holding_cost_per_unit']
            existing.ordering_cost = result['cost_parameters']['ordering_cost']
            existing.stockout_cost = result['cost_parameters']['stockout_cost']
            existing.last_calculated_at = datetime.utcnow()
            existing.calculation_method = 'eoq_safety_stock'
        else:
            # Create new
            params = OptimizationParameters(
                product_id=product_id,
                economic_order_quantity=result['economic_order_quantity'],
                reorder_point=result['reorder_point'],
                safety_stock=result['safety_stock'],
                max_stock_level=result['max_stock_level'],
                service_level=result['service_level'],
                lead_time_days=result['lead_time_days'],
                demand_std_dev=result['demand_statistics']['demand_std_dev'],
                holding_cost_per_unit=result['cost_parameters']['holding_cost_per_unit'],
                ordering_cost=result['cost_parameters']['ordering_cost'],
                stockout_cost=result['cost_parameters']['stockout_cost'],
                calculation_method='eoq_safety_stock'
            )
            self.db.add(params)
        
        await self.db.commit()
    
    async def optimize_all_products(self) -> Dict:
        """Run optimization for all active products"""
        logger.info("Starting optimization for all products")
        
        query = select(Product).where(Product.is_active == True)
        result = await self.db.execute(query)
        products = result.scalars().all()
        
        results = []
        successful = 0
        failed = 0
        
        for product in products:
            try:
                result = await self.optimize_product(product.id)
                if result['success']:
                    successful += 1
                else:
                    failed += 1
                results.append(result)
            except Exception as e:
                logger.error(f"Error optimizing product {product.id}: {e}")
                failed += 1
                results.append({
                    'success': False,
                    'product_id': product.id,
                    'error': str(e)
                })
        
        logger.info(f"Optimization completed: {successful} successful, {failed} failed")
        
        return {
            'total': len(products),
            'successful': successful,
            'failed': failed,
            'results': results
        }
    
    async def get_optimization_parameters(self, product_id: int) -> Optional[Dict]:
        """Retrieve optimization parameters for a product"""
        query = select(OptimizationParameters).where(
            OptimizationParameters.product_id == product_id
        )
        result = await self.db.execute(query)
        params = result.scalar_one_or_none()
        
        if not params:
            return None
        
        return {
            'economic_order_quantity': params.economic_order_quantity,
            'reorder_point': params.reorder_point,
            'safety_stock': params.safety_stock,
            'max_stock_level': params.max_stock_level,
            'service_level': params.service_level,
            'lead_time_days': params.lead_time_days,
            'last_calculated': params.last_calculated_at.isoformat()
        }
