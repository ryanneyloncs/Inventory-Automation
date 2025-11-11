"""
Celery tasks for inventory automation background processing
"""
import asyncio
from celery import Celery
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "inventory_automation",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)


def run_async(coro):
    """Helper to run async functions in Celery tasks"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@celery_app.task(name="tasks.train_forecasting_models")
def train_forecasting_models(product_id: int = None):
    """
    Train forecasting models for products

    Args:
        product_id: Specific product ID to train, or None for all products
    """
    logger.info(f"Starting forecasting model training for product_id={product_id}")

    async def _train():
        from src.database import AsyncSessionLocal
        from src.services.forecasting.forecasting_service import ForecastingService

        async with AsyncSessionLocal() as session:
            service = ForecastingService(session)

            if product_id:
                result = await service.train_and_forecast(product_id)
                return result
            else:
                # Train for all products
                from src.models.database import Product
                from sqlalchemy import select

                stmt = select(Product).where(Product.is_active == True)
                result = await session.execute(stmt)
                products = result.scalars().all()

                results = []
                for product in products:
                    try:
                        result = await service.train_and_forecast(product.id)
                        results.append({
                            "product_id": product.id,
                            "success": result.get("success", False)
                        })
                    except Exception as e:
                        logger.error(f"Error training model for product {product.id}: {e}")
                        results.append({
                            "product_id": product.id,
                            "success": False,
                            "error": str(e)
                        })

                return {
                    "total_products": len(products),
                    "results": results
                }

    result = run_async(_train())
    logger.info(f"Completed forecasting model training for product_id={product_id}")
    return result


@celery_app.task(name="tasks.optimize_inventory_parameters")
def optimize_inventory_parameters(product_id: int = None):
    """
    Optimize inventory parameters for products

    Args:
        product_id: Specific product ID to optimize, or None for all products
    """
    logger.info(f"Starting inventory optimization for product_id={product_id}")

    async def _optimize():
        from src.database import AsyncSessionLocal
        from src.services.optimization.optimization_service import OptimizationService

        async with AsyncSessionLocal() as session:
            service = OptimizationService(session)

            if product_id:
                result = await service.optimize_product(product_id)
                return result
            else:
                result = await service.optimize_all_products()
                return result

    result = run_async(_optimize())
    logger.info(f"Completed inventory optimization for product_id={product_id}")
    return result


@celery_app.task(name="tasks.process_automatic_reorders")
def process_automatic_reorders():
    """
    Check inventory levels and process automatic reorders
    """
    logger.info("Starting automatic reorder processing")

    async def _reorder():
        from src.database import AsyncSessionLocal
        from src.services.reorder.reorder_service import ReorderService

        async with AsyncSessionLocal() as session:
            service = ReorderService(session)
            result = await service.process_automatic_reorders()
            return result

    result = run_async(_reorder())
    logger.info(f"Completed automatic reorder processing: {result}")
    return result


@celery_app.task(name="tasks.check_low_stock_alerts")
def check_low_stock_alerts():
    """
    Check for low stock levels and generate alerts
    """
    logger.info("Checking for low stock alerts")

    async def _check_alerts():
        from src.database import AsyncSessionLocal
        from src.services.reorder.reorder_service import ReorderService

        async with AsyncSessionLocal() as session:
            service = ReorderService(session)
            alerts = await service.get_low_stock_alerts()

            # Log alerts
            if alerts:
                logger.warning(f"Found {len(alerts)} low stock alerts")
                for alert in alerts:
                    logger.warning(
                        f"Low stock: {alert.get('sku')} - "
                        f"{alert.get('quantity_on_hand')} units "
                        f"(safety stock: {alert.get('safety_stock')})"
                    )

            return {
                "total_alerts": len(alerts),
                "alerts": alerts
            }

    result = run_async(_check_alerts())
    logger.info(f"Completed low stock alert check: {result.get('total_alerts', 0)} alerts")
    return result


@celery_app.task(name="tasks.check_overdue_orders")
def check_overdue_orders():
    """
    Check for overdue purchase orders
    """
    logger.info("Checking for overdue orders")

    async def _check_overdue():
        from src.database import AsyncSessionLocal
        from src.services.reorder.reorder_service import ReorderService

        async with AsyncSessionLocal() as session:
            service = ReorderService(session)
            overdue = await service.check_overdue_orders()

            # Log overdue orders
            if overdue:
                logger.warning(f"Found {len(overdue)} overdue orders")
                for order in overdue:
                    logger.warning(
                        f"Overdue order: {order.get('order_number')} - "
                        f"Expected: {order.get('expected_delivery_date')}"
                    )

            return {
                "total_overdue": len(overdue),
                "orders": overdue
            }

    result = run_async(_check_overdue())
    logger.info(f"Completed overdue order check: {result.get('total_overdue', 0)} overdue")
    return result


@celery_app.task(name="tasks.cleanup_old_forecasts")
def cleanup_old_forecasts(days_old: int = 90):
    """
    Clean up old forecast data

    Args:
        days_old: Remove forecasts older than this many days
    """
    logger.info(f"Cleaning up forecasts older than {days_old} days")

    async def _cleanup():
        from src.database import AsyncSessionLocal
        from src.models.database import ForecastResult
        from sqlalchemy import delete
        from datetime import datetime, timedelta

        async with AsyncSessionLocal() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            stmt = delete(ForecastResult).where(
                ForecastResult.created_at < cutoff_date
            )

            result = await session.execute(stmt)
            await session.commit()

            return {
                "deleted_count": result.rowcount,
                "cutoff_date": cutoff_date.isoformat()
            }

    result = run_async(_cleanup())
    logger.info(f"Cleaned up {result.get('deleted_count', 0)} old forecasts")
    return result


# Periodic task schedule (to be configured in Celery beat)
celery_app.conf.beat_schedule = {
    'daily-optimization': {
        'task': 'tasks.optimize_inventory_parameters',
        'schedule': 86400.0,  # 24 hours
        'args': (None,)  # Optimize all products
    },
    'six-hourly-reorder-check': {
        'task': 'tasks.process_automatic_reorders',
        'schedule': 21600.0,  # 6 hours
    },
    'hourly-low-stock-check': {
        'task': 'tasks.check_low_stock_alerts',
        'schedule': 3600.0,  # 1 hour
    },
    'daily-overdue-check': {
        'task': 'tasks.check_overdue_orders',
        'schedule': 86400.0,  # 24 hours
    },
    'weekly-forecast-training': {
        'task': 'tasks.train_forecasting_models',
        'schedule': 604800.0,  # 7 days
        'args': (None,)  # Train all products
    },
    'monthly-forecast-cleanup': {
        'task': 'tasks.cleanup_old_forecasts',
        'schedule': 2592000.0,  # 30 days
        'args': (90,)  # Clean up forecasts older than 90 days
    },
}

if __name__ == "__main__":
    # For testing individual tasks
    celery_app.start()
