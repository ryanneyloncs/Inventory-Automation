import asyncio
import schedule
import time
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import AsyncSessionLocal
from src.services.optimization.optimization_service import OptimizationService
from src.services.reorder.reorder_service import ReorderService
from src.services.forecasting.forecasting_service import ForecastingService
from sqlalchemy import select
from src.models.database import Product

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_daily_optimization():
    """Run optimization for all products"""
    logger.info("Starting daily optimization...")
    
    try:
        async with AsyncSessionLocal() as db:
            service = OptimizationService(db)
            result = await service.optimize_all_products()
            
            logger.info(
                f"Optimization completed: {result['successful']} successful, "
                f"{result['failed']} failed"
            )
    
    except Exception as e:
        logger.error(f"Error in daily optimization: {e}")


async def run_reorder_check():
    """Check and process reorders"""
    logger.info("Checking reorder needs...")
    
    try:
        async with AsyncSessionLocal() as db:
            service = ReorderService(db)
            result = await service.process_automatic_reorders()
            
            logger.info(
                f"Reorder check completed: {result['successful']} POs created, "
                f"{result['failed']} failed"
            )
    
    except Exception as e:
        logger.error(f"Error in reorder check: {e}")


async def run_weekly_forecasting():
    """Generate forecasts for all products"""
    logger.info("Starting weekly forecasting...")
    
    try:
        async with AsyncSessionLocal() as db:
            # Get all active products
            query = select(Product).where(Product.is_active == True)
            result = await db.execute(query)
            products = result.scalars().all()
            
            successful = 0
            failed = 0
            
            for product in products:
                try:
                    service = ForecastingService(db)
                    forecast_result = await service.train_and_forecast(
                        product_id=product.id,
                        forecast_horizon=90
                    )
                    
                    if forecast_result['success']:
                        successful += 1
                    else:
                        failed += 1
                
                except Exception as e:
                    logger.error(f"Error forecasting product {product.id}: {e}")
                    failed += 1
            
            logger.info(
                f"Forecasting completed: {successful} successful, {failed} failed"
            )
    
    except Exception as e:
        logger.error(f"Error in weekly forecasting: {e}")


async def check_overdue_orders():
    """Check for overdue purchase orders"""
    logger.info("Checking for overdue orders...")
    
    try:
        async with AsyncSessionLocal() as db:
            service = ReorderService(db)
            overdue = await service.check_overdue_orders()
            
            if overdue:
                logger.warning(f"Found {len(overdue)} overdue orders")
                for order in overdue:
                    logger.warning(
                        f"PO {order['order_number']} from {order['supplier']} "
                        f"is {order['days_overdue']} days overdue"
                    )
            else:
                logger.info("No overdue orders")
    
    except Exception as e:
        logger.error(f"Error checking overdue orders: {e}")


async def check_low_stock():
    """Check for low stock alerts"""
    logger.info("Checking for low stock...")
    
    try:
        async with AsyncSessionLocal() as db:
            service = ReorderService(db)
            alerts = await service.get_low_stock_alerts()
            
            if alerts:
                logger.warning(f"Found {len(alerts)} products with low stock")
                for alert in alerts[:5]:  # Log first 5
                    logger.warning(
                        f"Low stock: {alert['name']} (SKU: {alert['sku']}) - "
                        f"Current: {alert['current_quantity']}, "
                        f"Safety Stock: {alert['safety_stock']}"
                    )
            else:
                logger.info("No low stock alerts")
    
    except Exception as e:
        logger.error(f"Error checking low stock: {e}")


def schedule_tasks():
    """Schedule all periodic tasks"""
    
    # Daily optimization at 2 AM
    schedule.every().day.at("02:00").do(
        lambda: asyncio.run(run_daily_optimization())
    )
    
    # Check reorders every 6 hours
    schedule.every(6).hours.do(
        lambda: asyncio.run(run_reorder_check())
    )
    
    # Weekly forecasting on Sunday at 3 AM
    schedule.every().sunday.at("03:00").do(
        lambda: asyncio.run(run_weekly_forecasting())
    )
    
    # Check overdue orders daily at 9 AM
    schedule.every().day.at("09:00").do(
        lambda: asyncio.run(check_overdue_orders())
    )
    
    # Check low stock every 4 hours
    schedule.every(4).hours.do(
        lambda: asyncio.run(check_low_stock())
    )
    
    logger.info("Scheduler tasks configured:")
    logger.info("- Daily optimization: 2 AM")
    logger.info("- Reorder check: Every 6 hours")
    logger.info("- Weekly forecasting: Sunday 3 AM")
    logger.info("- Overdue orders check: Daily 9 AM")
    logger.info("- Low stock check: Every 4 hours")


def main():
    """Main scheduler loop"""
    logger.info("Starting inventory automation scheduler...")
    
    schedule_tasks()
    
    # Run initial checks
    logger.info("Running initial checks...")
    asyncio.run(check_low_stock())
    asyncio.run(run_reorder_check())
    
    # Main loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
