from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import logging

from config.settings import settings
from src.database import get_db, engine
from src.models.database import Base
from src.services.forecasting.forecasting_service import ForecastingService
from src.services.optimization.optimization_service import OptimizationService
from src.services.reorder.reorder_service import ReorderService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.APP_VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for requests/responses
class ForecastRequest(BaseModel):
    product_id: int
    forecast_horizon: int = 90


class ForecastResponse(BaseModel):
    success: bool
    product_id: int
    forecast: Optional[dict] = None
    error: Optional[str] = None


class OptimizationRequest(BaseModel):
    product_id: Optional[int] = None


class OptimizationResponse(BaseModel):
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None


class ReorderRequest(BaseModel):
    auto_send: bool = True


class PurchaseOrderCreate(BaseModel):
    supplier_id: int
    items: List[dict]
    auto_send: bool = True


class ReceivePORequest(BaseModel):
    items_received: List[dict]


# Health check
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "timestamp": datetime.utcnow().isoformat()
    }


# Forecasting endpoints
@app.post("/api/v1/forecast", response_model=ForecastResponse)
async def create_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate demand forecast for a product
    
    - **product_id**: Product ID to forecast
    - **forecast_horizon**: Number of days to forecast (default: 90)
    """
    try:
        service = ForecastingService(db)
        result = await service.train_and_forecast(
            product_id=request.product_id,
            forecast_horizon=request.forecast_horizon
        )
        
        return ForecastResponse(**result)
    
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/forecast/{product_id}")
async def get_forecast(
    product_id: int,
    days_ahead: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    Get existing forecast for a product
    
    - **product_id**: Product ID
    - **days_ahead**: Number of days to retrieve (default: 30)
    """
    try:
        service = ForecastingService(db)
        forecast = await service.get_forecast(product_id, days_ahead)
        
        return {
            "product_id": product_id,
            "forecast": forecast,
            "days_ahead": days_ahead
        }
    
    except Exception as e:
        logger.error(f"Error retrieving forecast: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Optimization endpoints
@app.post("/api/v1/optimize", response_model=OptimizationResponse)
async def optimize_product(
    request: OptimizationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Optimize inventory parameters for product(s)
    
    - **product_id**: Specific product ID (optional, if not provided optimizes all)
    """
    try:
        service = OptimizationService(db)
        
        if request.product_id:
            result = await service.optimize_product(request.product_id)
        else:
            result = await service.optimize_all_products()
        
        return OptimizationResponse(**result)
    
    except Exception as e:
        logger.error(f"Error optimizing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/optimize/{product_id}")
async def get_optimization_params(
    product_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get optimization parameters for a product
    
    - **product_id**: Product ID
    """
    try:
        service = OptimizationService(db)
        params = await service.get_optimization_parameters(product_id)
        
        if params is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Optimization parameters not found"
            )
        
        return {
            "product_id": product_id,
            "parameters": params
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving optimization params: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Reorder endpoints
@app.get("/api/v1/reorder/check")
async def check_reorder_needs(
    db: AsyncSession = Depends(get_db)
):
    """
    Check which products need reordering
    
    Returns list of products below reorder point
    """
    try:
        service = ReorderService(db)
        candidates = await service.check_reorder_needs()
        
        return {
            "total": len(candidates),
            "products": candidates
        }
    
    except Exception as e:
        logger.error(f"Error checking reorder needs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/reorder/automatic")
async def process_automatic_reorders(
    db: AsyncSession = Depends(get_db)
):
    """
    Automatically process reorders for all products
    
    Creates purchase orders for products below reorder point
    """
    try:
        service = ReorderService(db)
        result = await service.process_automatic_reorders()
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing automatic reorders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/purchase-orders")
async def create_purchase_order(
    request: PurchaseOrderCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a purchase order
    
    - **supplier_id**: Supplier ID
    - **items**: List of items [{product_id, quantity}, ...]
    - **auto_send**: Automatically send to supplier (default: true)
    """
    try:
        service = ReorderService(db)
        result = await service.create_purchase_order(
            supplier_id=request.supplier_id,
            items=request.items,
            auto_send=request.auto_send
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error creating purchase order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/purchase-orders/{po_id}/receive")
async def receive_purchase_order(
    po_id: int,
    request: ReceivePORequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Process received purchase order
    
    Updates inventory levels based on received quantities
    
    - **po_id**: Purchase order ID
    - **items_received**: List of items [{product_id, quantity}, ...]
    """
    try:
        service = ReorderService(db)
        result = await service.receive_purchase_order(
            po_id=po_id,
            items_received=request.items_received
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error receiving purchase order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/alerts/low-stock")
async def get_low_stock_alerts(
    db: AsyncSession = Depends(get_db)
):
    """
    Get products with low stock levels
    
    Returns products below safety stock
    """
    try:
        service = ReorderService(db)
        alerts = await service.get_low_stock_alerts()
        
        return {
            "total": len(alerts),
            "alerts": alerts
        }
    
    except Exception as e:
        logger.error(f"Error retrieving low stock alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/alerts/overdue-orders")
async def get_overdue_orders(
    db: AsyncSession = Depends(get_db)
):
    """
    Get overdue purchase orders
    
    Returns orders past expected delivery date
    """
    try:
        service = ReorderService(db)
        overdue = await service.check_overdue_orders()
        
        return {
            "total": len(overdue),
            "orders": overdue
        }
    
    except Exception as e:
        logger.error(f"Error retrieving overdue orders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created/verified")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")
    await engine.dispose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
