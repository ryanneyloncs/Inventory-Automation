"""
Seasonal Decomposition & Event Impact Analysis Service

Provides:
- Time series decomposition (trend, seasonal, residual)
- Event impact measurement
- Anomaly detection
- Seasonal pattern analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

from src.models.database import SalesRecord, Product
from src.models.sales_forecasting_models import (
    SeasonalDecomposition, EventImpact
)

logger = logging.getLogger(__name__)


class SeasonalAnalyzer:
    """
    Seasonal decomposition and event impact analysis
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def decompose_time_series(
        self,
        product_id: int,
        decomposition_type: str = 'multiplicative',
        period: Optional[int] = None
    ) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components

        Args:
            product_id: Product to analyze
            decomposition_type: 'additive' or 'multiplicative'
            period: Seasonality period (auto-detected if None)

        Returns:
            Dict with decomposition results and insights
        """
        if decomposition_type not in ['additive', 'multiplicative']:
            raise ValueError("decomposition_type must be 'additive' or 'multiplicative'")

        # Get sales data
        sales_data = await self._get_sales_data(product_id)

        if len(sales_data) < 60:  # Need at least 2 months of data
            raise ValueError(f"Insufficient data for decomposition. Need at least 60 days, got {len(sales_data)}")

        # Prepare time series
        ts = pd.Series(
            sales_data['quantity'].values,
            index=pd.DatetimeIndex(sales_data['date'])
        )

        # Auto-detect period if not provided
        if period is None:
            period = self._detect_seasonality_period(ts)

        # Perform decomposition
        decomposition = seasonal_decompose(
            ts,
            model=decomposition_type,
            period=period,
            extrapolate_trend='freq'
        )

        # Save to database
        await self._save_decomposition_results(
            product_id=product_id,
            dates=ts.index,
            observed=decomposition.observed,
            trend=decomposition.trend,
            seasonal=decomposition.seasonal,
            residual=decomposition.resid,
            decomposition_type=decomposition_type
        )

        # Analyze patterns
        analysis = self._analyze_decomposition_patterns(
            trend=decomposition.trend,
            seasonal=decomposition.seasonal,
            residual=decomposition.resid,
            period=period
        )

        result = {
            'product_id': product_id,
            'decomposition_type': decomposition_type,
            'period': period,
            'data_points': len(ts),
            'date_range': {
                'start': ts.index[0].isoformat(),
                'end': ts.index[-1].isoformat()
            },
            'components': {
                'trend': decomposition.trend.dropna().tolist(),
                'seasonal': decomposition.seasonal.dropna().tolist(),
                'residual': decomposition.resid.dropna().tolist(),
                'dates': [d.isoformat() for d in ts.index]
            },
            'analysis': analysis
        }

        logger.info(f"Completed seasonal decomposition for product {product_id}")
        return result

    def _detect_seasonality_period(self, ts: pd.Series) -> int:
        """
        Auto-detect seasonality period using autocorrelation
        """
        # Try common periods: weekly (7), monthly (~30), yearly (~365)
        possible_periods = [7, 30, 365]

        # Calculate autocorrelation for each period
        from pandas.plotting import autocorrelation_plot

        # Simple heuristic: use weekly for short series, monthly for medium, yearly for long
        if len(ts) < 90:
            return 7  # Weekly
        elif len(ts) < 365:
            return 30  # Monthly
        else:
            return 365  # Yearly

    def _analyze_decomposition_patterns(
        self,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        period: int
    ) -> Dict:
        """
        Analyze decomposition components for insights
        """
        analysis = {}

        # Trend analysis
        trend_clean = trend.dropna()
        if len(trend_clean) > 1:
            trend_change = (trend_clean.iloc[-1] - trend_clean.iloc[0]) / trend_clean.iloc[0] * 100
            trend_direction = 'increasing' if trend_change > 5 else 'decreasing' if trend_change < -5 else 'stable'

            analysis['trend'] = {
                'direction': trend_direction,
                'change_pct': float(trend_change),
                'average': float(trend_clean.mean()),
                'min': float(trend_clean.min()),
                'max': float(trend_clean.max())
            }

        # Seasonal analysis
        seasonal_clean = seasonal.dropna()
        if len(seasonal_clean) > 0:
            # Calculate seasonal strength
            var_seasonal = np.var(seasonal_clean)
            var_residual = np.var(residual.dropna())
            seasonal_strength = 1 - (var_residual / (var_seasonal + var_residual)) if (var_seasonal + var_residual) > 0 else 0

            analysis['seasonal'] = {
                'strength': float(seasonal_strength),
                'period': period,
                'amplitude': float(seasonal_clean.max() - seasonal_clean.min()),
                'peak_index': int(seasonal_clean.idxmax()) if len(seasonal_clean) > 0 else 0
            }

        # Residual analysis (anomalies)
        residual_clean = residual.dropna()
        if len(residual_clean) > 0:
            # Detect anomalies using z-score
            z_scores = np.abs(stats.zscore(residual_clean))
            anomalies = residual_clean[z_scores > 3]

            analysis['residual'] = {
                'std': float(residual_clean.std()),
                'anomalies_count': len(anomalies),
                'anomaly_dates': [d.isoformat() for d in anomalies.index],
                'noise_level': 'high' if residual_clean.std() > seasonal_clean.std() else 'low'
            }

        return analysis

    async def measure_event_impact(
        self,
        event_name: str,
        event_type: str,
        event_date: date,
        product_ids: List[int],
        impact_window_days: int = 14
    ) -> Dict:
        """
        Measure the impact of an event on sales

        Args:
            event_name: Name of the event (e.g., "Black Friday 2024")
            event_type: Type (holiday, promotion, external, custom)
            event_date: Date of the event
            product_ids: Products affected by the event
            impact_window_days: Days before/after to measure impact

        Returns:
            Dict with event impact analysis
        """
        impact_start = event_date - timedelta(days=impact_window_days)
        impact_end = event_date + timedelta(days=impact_window_days)

        results = {
            'event_name': event_name,
            'event_type': event_type,
            'event_date': event_date.isoformat(),
            'impact_window_days': impact_window_days,
            'products': [],
            'aggregate': {
                'baseline_sales': 0,
                'actual_sales': 0,
                'sales_lift_pct': 0,
                'revenue_impact': 0
            }
        }

        for product_id in product_ids:
            product_impact = await self._measure_product_event_impact(
                product_id=product_id,
                event_date=event_date,
                impact_start=impact_start,
                impact_end=impact_end
            )

            if product_impact:
                results['products'].append(product_impact)
                results['aggregate']['baseline_sales'] += product_impact['baseline_sales']
                results['aggregate']['actual_sales'] += product_impact['actual_sales']
                results['aggregate']['revenue_impact'] += product_impact['revenue_impact']

        # Calculate aggregate lift
        if results['aggregate']['baseline_sales'] > 0:
            results['aggregate']['sales_lift_pct'] = (
                (results['aggregate']['actual_sales'] / results['aggregate']['baseline_sales'] - 1) * 100
            )

        # Save event impact
        event_impact = EventImpact(
            event_name=event_name,
            event_type=event_type,
            event_date=event_date,
            impact_start_date=impact_start,
            impact_end_date=impact_end,
            affected_products=product_ids,
            baseline_sales=results['aggregate']['baseline_sales'],
            actual_sales=results['aggregate']['actual_sales'],
            sales_lift_pct=results['aggregate']['sales_lift_pct'],
            revenue_impact=results['aggregate']['revenue_impact'],
            metadata={'product_count': len(product_ids)}
        )
        self.db.add(event_impact)
        await self.db.commit()

        logger.info(f"Measured event impact: {event_name} - {results['aggregate']['sales_lift_pct']:.1f}% lift")
        return results

    async def _measure_product_event_impact(
        self,
        product_id: int,
        event_date: date,
        impact_start: date,
        impact_end: date
    ) -> Optional[Dict]:
        """
        Measure event impact for a single product
        """
        # Get product
        product = await self._get_product(product_id)

        # Get actual sales during event window
        event_sales = await self._get_sales_in_period(
            product_id=product_id,
            start_date=impact_start,
            end_date=impact_end
        )

        if event_sales.empty:
            return None

        # Get baseline (same period from previous year or 4 weeks earlier)
        baseline_start = impact_start - timedelta(days=28)  # 4 weeks earlier
        baseline_end = impact_end - timedelta(days=28)

        baseline_sales = await self._get_sales_in_period(
            product_id=product_id,
            start_date=baseline_start,
            end_date=baseline_end
        )

        if baseline_sales.empty:
            return None

        # Calculate metrics
        actual_sales_total = event_sales['quantity'].sum()
        baseline_sales_total = baseline_sales['quantity'].sum()

        sales_lift_pct = (
            (actual_sales_total / baseline_sales_total - 1) * 100
            if baseline_sales_total > 0 else 0
        )

        actual_revenue = (event_sales['quantity'] * event_sales['unit_price']).sum()
        baseline_revenue = (baseline_sales['quantity'] * baseline_sales['unit_price']).sum()
        revenue_impact = actual_revenue - baseline_revenue

        # Statistical significance test
        is_significant = self._test_statistical_significance(
            event_sales['quantity'].values,
            baseline_sales['quantity'].values
        )

        return {
            'product_id': product_id,
            'product_name': product.name,
            'baseline_sales': float(baseline_sales_total),
            'actual_sales': float(actual_sales_total),
            'sales_lift_pct': float(sales_lift_pct),
            'revenue_impact': float(revenue_impact),
            'is_statistically_significant': is_significant
        }

    def _test_statistical_significance(
        self,
        event_data: np.ndarray,
        baseline_data: np.ndarray,
        alpha: float = 0.05
    ) -> bool:
        """
        Test if difference between event and baseline is statistically significant
        """
        if len(event_data) < 2 or len(baseline_data) < 2:
            return False

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(event_data, baseline_data)

        return p_value < alpha

    async def get_historical_event_impacts(
        self,
        event_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get historical event impacts for analysis

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of results

        Returns:
            List of event impact records
        """
        query = select(EventImpact).order_by(EventImpact.event_date.desc()).limit(limit)

        if event_type:
            query = query.where(EventImpact.event_type == event_type)

        result = await self.db.execute(query)
        events = result.scalars().all()

        return [event.to_dict() for event in events]

    async def detect_anomalies(
        self,
        product_id: int,
        threshold: float = 3.0
    ) -> Dict:
        """
        Detect sales anomalies using statistical methods

        Args:
            product_id: Product to analyze
            threshold: Z-score threshold for anomaly detection (default: 3.0)

        Returns:
            Dict with detected anomalies
        """
        # Get sales data
        sales_data = await self._get_sales_data(product_id)

        if len(sales_data) < 30:
            raise ValueError(f"Need at least 30 days of data for anomaly detection")

        # Calculate rolling statistics
        window = 7
        sales_data['rolling_mean'] = sales_data['quantity'].rolling(window=window).mean()
        sales_data['rolling_std'] = sales_data['quantity'].rolling(window=window).std()

        # Calculate z-scores
        sales_data['z_score'] = (
            (sales_data['quantity'] - sales_data['rolling_mean']) /
            sales_data['rolling_std']
        )

        # Detect anomalies
        anomalies = sales_data[np.abs(sales_data['z_score']) > threshold].copy()

        result = {
            'product_id': product_id,
            'threshold': threshold,
            'total_data_points': len(sales_data),
            'anomalies_detected': len(anomalies),
            'anomaly_rate_pct': (len(anomalies) / len(sales_data) * 100),
            'anomalies': [
                {
                    'date': row['date'].isoformat(),
                    'quantity': float(row['quantity']),
                    'expected': float(row['rolling_mean']),
                    'z_score': float(row['z_score']),
                    'deviation_pct': float((row['quantity'] / row['rolling_mean'] - 1) * 100) if row['rolling_mean'] > 0 else 0
                }
                for _, row in anomalies.iterrows()
            ]
        }

        logger.info(f"Detected {len(anomalies)} anomalies for product {product_id}")
        return result

    async def _get_sales_data(self, product_id: int) -> pd.DataFrame:
        """
        Get historical sales data for a product
        """
        query = select(SalesRecord).where(
            SalesRecord.product_id == product_id
        ).order_by(SalesRecord.sale_date)

        result = await self.db.execute(query)
        records = result.scalars().all()

        if not records:
            raise ValueError(f"No sales data found for product {product_id}")

        df = pd.DataFrame([{
            'date': r.sale_date,
            'quantity': r.quantity,
            'unit_price': r.unit_price
        } for r in records])

        df['date'] = pd.to_datetime(df['date'])

        # Aggregate by day
        df = df.groupby('date').agg({
            'quantity': 'sum',
            'unit_price': 'mean'
        }).reset_index()

        return df

    async def _get_sales_in_period(
        self,
        product_id: int,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Get sales data for a specific period
        """
        query = select(SalesRecord).where(
            and_(
                SalesRecord.product_id == product_id,
                SalesRecord.sale_date >= start_date,
                SalesRecord.sale_date <= end_date
            )
        ).order_by(SalesRecord.sale_date)

        result = await self.db.execute(query)
        records = result.scalars().all()

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame([{
            'date': r.sale_date,
            'quantity': r.quantity,
            'unit_price': r.unit_price
        } for r in records])

        return df

    async def _save_decomposition_results(
        self,
        product_id: int,
        dates: pd.DatetimeIndex,
        observed: pd.Series,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        decomposition_type: str
    ):
        """
        Save decomposition results to database
        """
        for i, date in enumerate(dates):
            if pd.notna(trend.iloc[i]):  # Skip NaN values
                result = SeasonalDecomposition(
                    product_id=product_id,
                    date=date.date(),
                    observed=float(observed.iloc[i]),
                    trend=float(trend.iloc[i]),
                    seasonal=float(seasonal.iloc[i]),
                    residual=float(residual.iloc[i]),
                    decomposition_type=decomposition_type
                )
                self.db.add(result)

        await self.db.commit()
        logger.info(f"Saved {len(dates)} decomposition results for product {product_id}")

    async def _get_product(self, product_id: int) -> Product:
        """Get product by ID"""
        query = select(Product).where(Product.id == product_id)
        result = await self.db.execute(query)
        product = result.scalar_one_or_none()

        if not product:
            raise ValueError(f"Product {product_id} not found")

        return product
