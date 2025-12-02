"""
Sales & Revenue Forecasting Suite - API Routes

Provides REST endpoints for:
- Multi-horizon forecasting
- Scenario planning
- Seasonal analysis
- Cannibalization analysis
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import date, datetime
from pydantic import BaseModel, Field
import logging

from src.database import get_db
from src.services.multi_horizon_forecaster import MultiHorizonForecaster
from src.services.scenario_planner import ScenarioPlanner
from src.services.seasonal_analyzer import SeasonalAnalyzer
from src.services.cannibalization_analyzer import CannibalizationAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sales-forecasting", tags=["Sales Forecasting"])


# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

class ForecastRequest(BaseModel):
    product_id: int = Field(..., description="Product ID to forecast")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level for intervals")


class SingleHorizonRequest(BaseModel):
    product_id: int
    horizon: str = Field(..., description="One of: hourly, daily, weekly, monthly")
    periods: Optional[int] = Field(None, description="Number of periods to forecast")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)


class CreateScenarioRequest(BaseModel):
    name: str = Field(..., max_length=100)
    scenario_type: str = Field(..., description="One of: promotion, new_product, seasonal, price_change, custom")
    start_date: date
    end_date: date
    parameters: dict
    description: Optional[str] = None
    created_by: Optional[str] = None


class PromotionScenarioRequest(BaseModel):
    scenario_id: int
    product_ids: List[int]
    discount_pct: float = Field(..., ge=0, le=100, description="Discount percentage")
    expected_lift_pct: Optional[float] = Field(None, ge=0, description="Expected sales lift percentage")


class NewProductLaunchRequest(BaseModel):
    scenario_id: int
    new_product_id: int
    similar_product_ids: List[int]
    expected_sales_multiplier: float = Field(0.8, ge=0, le=2.0)


class PriceChangeRequest(BaseModel):
    scenario_id: int
    product_id: int
    new_price: float = Field(..., gt=0)
    price_elasticity: Optional[float] = Field(None, description="Price elasticity of demand")


class CompareRequest(BaseModel):
    scenario_ids: List[int]


class DecomposeRequest(BaseModel):
    product_id: int
    decomposition_type: str = Field("multiplicative", description="additive or multiplicative")
    period: Optional[int] = Field(None, description="Seasonality period (auto-detected if None)")


class EventImpactRequest(BaseModel):
    event_name: str = Field(..., max_length=200)
    event_type: str = Field(..., description="holiday, promotion, external, custom")
    event_date: date
    product_ids: List[int]
    impact_window_days: int = Field(14, ge=1, le=90)


class CannibalizationRequest(BaseModel):
    new_product_id: int
    launch_date: date
    analysis_window_days: int = Field(90, ge=7, le=365)
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0)


class AnomalyDetectionRequest(BaseModel):
    product_id: int
    threshold: float = Field(3.0, ge=1.0, le=5.0, description="Z-score threshold")


# ============================================================================
# Forecasting Endpoints
# ============================================================================

@router.post("/forecast/all-horizons")
async def forecast_all_horizons(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate forecasts for all time horizons (hourly, daily, weekly, monthly)
    """
    try:
        forecaster = MultiHorizonForecaster(db)
        results = await forecaster.forecast_all_horizons(
            product_id=request.product_id,
            confidence_level=request.confidence_level
        )

        # Format response
        response = {
            'product_id': request.product_id,
            'confidence_level': request.confidence_level,
            'horizons': {}
        }

        for horizon, df in results.items():
            if not df.empty:
                response['horizons'][horizon] = {
                    'forecasts': df.to_dict('records'),
                    'summary': {
                        'total_predicted_sales': float(df['predicted_sales'].sum()),
                        'total_predicted_revenue': float(df['predicted_revenue'].sum()),
                        'average_daily_sales': float(df['predicted_sales'].mean()),
                        'periods': len(df)
                    }
                }

        return response

    except Exception as e:
        logger.error(f"Error in forecast_all_horizons: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast/single-horizon")
async def forecast_single_horizon(
    request: SingleHorizonRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate forecast for a specific time horizon
    """
    try:
        forecaster = MultiHorizonForecaster(db)
        result = await forecaster.forecast_single_horizon(
            product_id=request.product_id,
            horizon=request.horizon,
            periods=request.periods,
            confidence_level=request.confidence_level
        )

        return {
            'product_id': request.product_id,
            'horizon': request.horizon,
            'confidence_level': request.confidence_level,
            'forecast': result.to_dict('records'),
            'summary': {
                'total_predicted_sales': float(result['predicted_sales'].sum()),
                'total_predicted_revenue': float(result['predicted_revenue'].sum()),
                'average_sales': float(result['predicted_sales'].mean()),
                'periods': len(result)
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in forecast_single_horizon: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Scenario Planning Endpoints
# ============================================================================

@router.post("/scenarios/create")
async def create_scenario(
    request: CreateScenarioRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new scenario for what-if analysis
    """
    try:
        planner = ScenarioPlanner(db)
        scenario = await planner.create_scenario(
            name=request.name,
            scenario_type=request.scenario_type,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.parameters,
            description=request.description,
            created_by=request.created_by
        )

        return scenario.to_dict()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in create_scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenarios/analyze-promotion")
async def analyze_promotion(
    request: PromotionScenarioRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze impact of a promotional campaign
    """
    try:
        planner = ScenarioPlanner(db)
        results = await planner.analyze_promotion_scenario(
            scenario_id=request.scenario_id,
            product_ids=request.product_ids,
            discount_pct=request.discount_pct,
            expected_lift_pct=request.expected_lift_pct
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_promotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenarios/analyze-new-product")
async def analyze_new_product(
    request: NewProductLaunchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze new product launch scenario
    """
    try:
        planner = ScenarioPlanner(db)
        results = await planner.analyze_new_product_launch(
            scenario_id=request.scenario_id,
            new_product_id=request.new_product_id,
            similar_product_ids=request.similar_product_ids,
            expected_sales_multiplier=request.expected_sales_multiplier
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_new_product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenarios/analyze-price-change")
async def analyze_price_change(
    request: PriceChangeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze impact of price changes
    """
    try:
        planner = ScenarioPlanner(db)
        results = await planner.analyze_price_change_scenario(
            scenario_id=request.scenario_id,
            product_id=request.product_id,
            new_price=request.new_price,
            price_elasticity=request.price_elasticity
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_price_change: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenarios/compare")
async def compare_scenarios(
    request: CompareRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Compare multiple scenarios side-by-side
    """
    try:
        planner = ScenarioPlanner(db)
        comparison = await planner.compare_scenarios(
            scenario_ids=request.scenario_ids
        )

        return comparison

    except Exception as e:
        logger.error(f"Error in compare_scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Seasonal Analysis Endpoints
# ============================================================================

@router.post("/seasonal/decompose")
async def decompose_time_series(
    request: DecomposeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Decompose time series into trend, seasonal, and residual components
    """
    try:
        analyzer = SeasonalAnalyzer(db)
        results = await analyzer.decompose_time_series(
            product_id=request.product_id,
            decomposition_type=request.decomposition_type,
            period=request.period
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in decompose_time_series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seasonal/detect-anomalies")
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Detect sales anomalies using statistical methods
    """
    try:
        analyzer = SeasonalAnalyzer(db)
        results = await analyzer.detect_anomalies(
            product_id=request.product_id,
            threshold=request.threshold
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Event Impact Endpoints
# ============================================================================

@router.post("/events/measure-impact")
async def measure_event_impact(
    request: EventImpactRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Measure the impact of an event on sales
    """
    try:
        analyzer = SeasonalAnalyzer(db)
        results = await analyzer.measure_event_impact(
            event_name=request.event_name,
            event_type=request.event_type,
            event_date=request.event_date,
            product_ids=request.product_ids,
            impact_window_days=request.impact_window_days
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in measure_event_impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/historical")
async def get_historical_events(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get historical event impacts
    """
    try:
        analyzer = SeasonalAnalyzer(db)
        results = await analyzer.get_historical_event_impacts(
            event_type=event_type,
            limit=limit
        )

        return {
            'count': len(results),
            'events': results
        }

    except Exception as e:
        logger.error(f"Error in get_historical_events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Cannibalization Analysis Endpoints
# ============================================================================

@router.post("/cannibalization/analyze")
async def analyze_cannibalization(
    request: CannibalizationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze product cannibalization for a new product launch
    """
    try:
        analyzer = CannibalizationAnalyzer(db)
        results = await analyzer.analyze_new_product_launch(
            new_product_id=request.new_product_id,
            launch_date=request.launch_date,
            analysis_window_days=request.analysis_window_days,
            similarity_threshold=request.similarity_threshold
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_cannibalization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cannibalization/portfolio-matrix")
async def get_portfolio_matrix(
    product_ids: Optional[List[int]] = Query(None, description="Filter by product IDs"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get cannibalization matrix for product portfolio
    """
    try:
        analyzer = CannibalizationAnalyzer(db)
        results = await analyzer.get_portfolio_cannibalization_matrix(
            product_ids=product_ids
        )

        return results

    except Exception as e:
        logger.error(f"Error in get_portfolio_matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cannibalization/product-history/{product_id}")
async def get_product_cannibalization_history(
    product_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get cannibalization history for a specific product
    """
    try:
        analyzer = CannibalizationAnalyzer(db)
        results = await analyzer.get_product_cannibalization_history(
            product_id=product_id
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_product_cannibalization_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cannibalization/recommend-positioning")
async def recommend_positioning(
    new_product_id: int = Query(..., description="Product ID to position"),
    max_cannibalization_rate: float = Query(30.0, ge=0, le=100, description="Max acceptable cannibalization %"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recommendations for product positioning to minimize cannibalization
    """
    try:
        analyzer = CannibalizationAnalyzer(db)
        results = await analyzer.recommend_product_positioning(
            new_product_id=new_product_id,
            max_cannibalization_rate=max_cannibalization_rate
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in recommend_positioning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        'status': 'healthy',
        'service': 'Sales & Revenue Forecasting Suite',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    }


# ============================================================================
# API Documentation Helper
# ============================================================================

@router.get("/endpoints")
async def list_endpoints():
    """
    List all available endpoints with descriptions
    """
    return {
        'forecasting': {
            'POST /forecast/all-horizons': 'Generate forecasts for all time horizons',
            'POST /forecast/single-horizon': 'Generate forecast for specific horizon'
        },
        'scenarios': {
            'POST /scenarios/create': 'Create a new scenario',
            'POST /scenarios/analyze-promotion': 'Analyze promotion campaign',
            'POST /scenarios/analyze-new-product': 'Analyze new product launch',
            'POST /scenarios/analyze-price-change': 'Analyze price change impact',
            'POST /scenarios/compare': 'Compare multiple scenarios'
        },
        'seasonal': {
            'POST /seasonal/decompose': 'Decompose time series',
            'POST /seasonal/detect-anomalies': 'Detect sales anomalies'
        },
        'events': {
            'POST /events/measure-impact': 'Measure event impact on sales',
            'GET /events/historical': 'Get historical event impacts'
        },
        'cannibalization': {
            'POST /cannibalization/analyze': 'Analyze product cannibalization',
            'GET /cannibalization/portfolio-matrix': 'Get portfolio cannibalization matrix',
            'GET /cannibalization/product-history/{product_id}': 'Get product cannibalization history',
            'POST /cannibalization/recommend-positioning': 'Get positioning recommendations'
        },
        'system': {
            'GET /health': 'Service health check',
            'GET /endpoints': 'List all endpoints'
        }
    }
