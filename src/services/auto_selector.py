"""
Automatic Model Selection Service

Intelligently selects the best forecasting model based on:
- Data characteristics (volume, patterns, volatility)
- Product attributes
- Historical performance
- Computational constraints
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging
from scipy import stats
from scipy.fft import fft

from src.models.database import SalesRecord, Product
from src.services.multi_horizon_forecaster import MultiHorizonForecaster
from src.services.xgboost_predictor import XGBoostPredictor
from src.services.lstm_forecaster import LSTMForecaster

logger = logging.getLogger(__name__)


class AutoModelSelector:
    """
    Automatic model selection based on data characteristics
    """

    # Model selection thresholds
    THRESHOLDS = {
        'min_data_lstm': 90,        # Days needed for LSTM
        'min_data_xgboost': 60,     # Days needed for XGBoost
        'min_data_prophet': 30,     # Days needed for Prophet
        'high_volatility': 0.5,     # CV threshold for high volatility
        'strong_seasonality': 0.3,  # Threshold for seasonal strength
        'high_trend': 0.1,          # Threshold for trend strength
        'sparse_data_pct': 0.3      # % of zeros for sparse data
    }

    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_profile = {}
        self.recommendation = {}

    async def select_best_model(
        self,
        product_id: int,
        forecast_horizon: int = 30,
        consider_ensemble: bool = True
    ) -> Dict:
        """
        Analyze data and recommend best model

        Args:
            product_id: Product to analyze
            forecast_horizon: Forecast period
            consider_ensemble: Whether to recommend ensemble

        Returns:
            Dict with model recommendation and reasoning
        """
        logger.info(f"Analyzing data for product {product_id} to select best model")

        # Get sales data
        sales_data = await self._get_sales_data(product_id)

        # Profile the data
        self.data_profile = self._profile_data(sales_data)

        # Analyze patterns
        pattern_analysis = self._analyze_patterns(sales_data)
        self.data_profile.update(pattern_analysis)

        # Select model based on characteristics
        recommendation = self._recommend_model(
            data_profile=self.data_profile,
            forecast_horizon=forecast_horizon,
            consider_ensemble=consider_ensemble
        )

        self.recommendation = recommendation

        logger.info(f"Recommended model: {recommendation['primary_model']}")
        return recommendation

    def _profile_data(self, sales_data: pd.DataFrame) -> Dict:
        """
        Profile data characteristics
        """
        quantities = sales_data['quantity'].values

        profile = {
            'data_points': len(sales_data),
            'date_range': {
                'start': sales_data['date'].min().isoformat(),
                'end': sales_data['date'].max().isoformat(),
                'days': (sales_data['date'].max() - sales_data['date'].min()).days
            },
            'statistics': {
                'mean': float(np.mean(quantities)),
                'std': float(np.std(quantities)),
                'min': float(np.min(quantities)),
                'max': float(np.max(quantities)),
                'median': float(np.median(quantities)),
                'cv': float(np.std(quantities) / (np.mean(quantities) + 1e-6))  # Coefficient of variation
            },
            'data_quality': {
                'missing_pct': float(sales_data['quantity'].isna().sum() / len(sales_data) * 100),
                'zero_pct': float(np.sum(quantities == 0) / len(quantities) * 100),
                'outlier_pct': float(self._count_outliers(quantities) / len(quantities) * 100)
            }
        }

        # Classify volatility
        cv = profile['statistics']['cv']
        if cv < 0.2:
            profile['volatility'] = 'low'
        elif cv < 0.5:
            profile['volatility'] = 'moderate'
        else:
            profile['volatility'] = 'high'

        # Classify sparsity
        if profile['data_quality']['zero_pct'] > 30:
            profile['sparsity'] = 'sparse'
        elif profile['data_quality']['zero_pct'] > 10:
            profile['sparsity'] = 'moderate'
        else:
            profile['sparsity'] = 'dense'

        return profile

    def _analyze_patterns(self, sales_data: pd.DataFrame) -> Dict:
        """
        Analyze time series patterns
        """
        quantities = sales_data['quantity'].values

        patterns = {}

        # Trend analysis
        trend_strength = self._calculate_trend_strength(quantities)
        patterns['trend'] = {
            'strength': float(trend_strength),
            'direction': 'increasing' if trend_strength > 0.1 else 'decreasing' if trend_strength < -0.1 else 'stable'
        }

        # Seasonality analysis
        seasonal_strength = self._calculate_seasonal_strength(quantities)
        patterns['seasonality'] = {
            'strength': float(seasonal_strength),
            'detected': seasonal_strength > self.THRESHOLDS['strong_seasonality']
        }

        # Autocorrelation analysis
        autocorr = self._calculate_autocorrelation(quantities)
        patterns['autocorrelation'] = {
            'lag_1': float(autocorr[0]) if len(autocorr) > 0 else 0,
            'lag_7': float(autocorr[1]) if len(autocorr) > 1 else 0,
            'lag_30': float(autocorr[2]) if len(autocorr) > 2 else 0,
            'has_memory': any(ac > 0.3 for ac in autocorr)
        }

        # Frequency domain analysis
        dominant_frequency = self._detect_dominant_frequency(quantities)
        if dominant_frequency:
            patterns['dominant_cycle'] = {
                'period_days': int(dominant_frequency),
                'type': self._classify_cycle(dominant_frequency)
            }

        return patterns

    def _calculate_trend_strength(self, data: np.ndarray) -> float:
        """
        Calculate trend strength using linear regression
        """
        if len(data) < 10:
            return 0.0

        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data)

        # Normalize by mean to get relative trend
        mean_value = np.mean(data)
        if mean_value > 0:
            normalized_slope = slope / mean_value
        else:
            normalized_slope = 0

        # Weight by R-squared (how well linear model fits)
        trend_strength = normalized_slope * (r_value ** 2)

        return trend_strength

    def _calculate_seasonal_strength(self, data: np.ndarray) -> float:
        """
        Calculate seasonality strength using ACF
        """
        if len(data) < 14:
            return 0.0

        # Check for weekly seasonality (lag 7)
        acf_7 = self._acf_at_lag(data, lag=7)

        # Check for monthly seasonality (lag 30)
        acf_30 = self._acf_at_lag(data, lag=30) if len(data) >= 30 else 0

        # Return strongest seasonality
        return max(abs(acf_7), abs(acf_30))

    def _acf_at_lag(self, data: np.ndarray, lag: int) -> float:
        """
        Calculate autocorrelation at specific lag
        """
        if len(data) <= lag:
            return 0.0

        data_mean = np.mean(data)
        c0 = np.sum((data - data_mean) ** 2) / len(data)
        c_lag = np.sum((data[:-lag] - data_mean) * (data[lag:] - data_mean)) / len(data)

        if c0 == 0:
            return 0.0

        return c_lag / c0

    def _calculate_autocorrelation(self, data: np.ndarray) -> List[float]:
        """
        Calculate autocorrelation for multiple lags
        """
        lags = [1, 7, 30]
        autocorr = []

        for lag in lags:
            if len(data) > lag:
                acf = self._acf_at_lag(data, lag)
                autocorr.append(acf)

        return autocorr

    def _detect_dominant_frequency(self, data: np.ndarray) -> Optional[int]:
        """
        Detect dominant frequency using FFT
        """
        if len(data) < 30:
            return None

        # Detrend data
        detrended = data - np.linspace(data[0], data[-1], len(data))

        # Apply FFT
        fft_vals = fft(detrended)
        fft_power = np.abs(fft_vals) ** 2

        # Find dominant frequency (ignore DC component)
        freqs = np.fft.fftfreq(len(data))
        positive_freqs = freqs[1:len(freqs)//2]
        positive_power = fft_power[1:len(fft_power)//2]

        if len(positive_power) == 0:
            return None

        # Get dominant frequency
        dominant_idx = np.argmax(positive_power)
        dominant_freq = positive_freqs[dominant_idx]

        if dominant_freq > 0:
            period = int(1 / dominant_freq)
            if 3 <= period <= 365:  # Reasonable periods only
                return period

        return None

    def _classify_cycle(self, period: int) -> str:
        """
        Classify cycle type based on period
        """
        if 3 <= period <= 5:
            return 'short_cycle'
        elif 6 <= period <= 8:
            return 'weekly'
        elif 25 <= period <= 35:
            return 'monthly'
        elif 85 <= period <= 95:
            return 'quarterly'
        elif 350 <= period <= 380:
            return 'yearly'
        else:
            return 'custom'

    def _count_outliers(self, data: np.ndarray, threshold: float = 3.0) -> int:
        """
        Count outliers using z-score method
        """
        if len(data) < 3:
            return 0

        z_scores = np.abs(stats.zscore(data))
        outliers = np.sum(z_scores > threshold)

        return int(outliers)

    def _recommend_model(
        self,
        data_profile: Dict,
        forecast_horizon: int,
        consider_ensemble: bool
    ) -> Dict:
        """
        Recommend best model based on data profile
        """
        scores = {
            'prophet': 0,
            'xgboost': 0,
            'lstm': 0,
            'ensemble': 0
        }

        reasons = []

        # Check data availability
        data_days = data_profile['data_points']

        if data_days < self.THRESHOLDS['min_data_prophet']:
            return {
                'primary_model': 'simple_average',
                'confidence': 'low',
                'reasoning': [f"Insufficient data ({data_days} days). Need at least {self.THRESHOLDS['min_data_prophet']} days."],
                'fallback_models': [],
                'data_profile': data_profile
            }

        # Prophet scoring
        if data_days >= self.THRESHOLDS['min_data_prophet']:
            scores['prophet'] += 30

            if data_profile.get('seasonality', {}).get('detected'):
                scores['prophet'] += 30
                reasons.append("Strong seasonality detected → Prophet excels here")

            if data_profile.get('trend', {}).get('direction') != 'stable':
                scores['prophet'] += 20
                reasons.append("Clear trend present → Prophet handles trends well")

            if data_profile['volatility'] == 'low':
                scores['prophet'] += 10

        # XGBoost scoring
        if data_days >= self.THRESHOLDS['min_data_xgboost']:
            scores['xgboost'] += 30

            if data_profile['volatility'] == 'high':
                scores['xgboost'] += 30
                reasons.append("High volatility → XGBoost robust to noise")

            if data_profile.get('autocorrelation', {}).get('has_memory'):
                scores['xgboost'] += 20
                reasons.append("Strong autocorrelation → XGBoost captures lag features")

            if data_profile['sparsity'] == 'dense':
                scores['xgboost'] += 15

            if forecast_horizon <= 30:
                scores['xgboost'] += 10
                reasons.append("Short-term forecast → XGBoost accurate for near-term")

        # LSTM scoring
        if data_days >= self.THRESHOLDS['min_data_lstm']:
            scores['lstm'] += 30

            if data_profile.get('autocorrelation', {}).get('has_memory'):
                scores['lstm'] += 30
                reasons.append("Long-term dependencies → LSTM captures sequences")

            if 'dominant_cycle' in data_profile and data_profile['dominant_cycle']['type'] != 'short_cycle':
                scores['lstm'] += 20
                reasons.append("Complex patterns → LSTM neural network flexibility")

            if data_profile['volatility'] == 'moderate':
                scores['lstm'] += 15

            if forecast_horizon > 30:
                scores['lstm'] += 10
                reasons.append("Long-term forecast → LSTM learns long sequences")

        # Ensemble scoring
        if consider_ensemble and data_days >= self.THRESHOLDS['min_data_lstm']:
            # Ensemble gets average of top 3 scores + bonus
            top_3_avg = np.mean(sorted(scores.values(), reverse=True)[:3])
            scores['ensemble'] = top_3_avg + 20

            if data_profile['volatility'] == 'high':
                scores['ensemble'] += 10
                reasons.append("High volatility → Ensemble reduces risk")

            if len(reasons) >= 3:  # Complex data
                scores['ensemble'] += 10
                reasons.append("Complex patterns → Ensemble combines strengths")

        # Select best model
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        best_score = scores[best_model]

        # Sort remaining models as fallbacks
        fallback_models = sorted(
            [(m, s) for m, s in scores.items() if m != best_model],
            key=lambda x: x[1],
            reverse=True
        )

        # Determine confidence
        if best_score >= 80:
            confidence = 'high'
        elif best_score >= 60:
            confidence = 'medium'
        else:
            confidence = 'low'

        recommendation = {
            'primary_model': best_model,
            'confidence': confidence,
            'scores': scores,
            'reasoning': reasons[:5],  # Top 5 reasons
            'fallback_models': [m for m, _ in fallback_models],
            'data_profile': data_profile,
            'recommendations': self._generate_recommendations(best_model, data_profile)
        }

        return recommendation

    def _generate_recommendations(self, model: str, profile: Dict) -> List[str]:
        """
        Generate actionable recommendations
        """
        recommendations = []

        if model == 'prophet':
            recommendations.append("Use Prophet with automatic seasonality detection")
            if profile.get('seasonality', {}).get('detected'):
                recommendations.append("Enable yearly and weekly seasonality modes")

        elif model == 'xgboost':
            recommendations.append("Use XGBoost with extensive feature engineering")
            recommendations.append("Include lag features up to 30 days")
            if profile['volatility'] == 'high':
                recommendations.append("Use conservative max_depth (6-8) to prevent overfitting")

        elif model == 'lstm':
            recommendations.append("Use LSTM with lookback window of 60 days")
            if profile['data_points'] >= 120:
                recommendations.append("Enable attention mechanism for better performance")
            recommendations.append("Use early stopping to prevent overfitting")

        elif model == 'ensemble':
            recommendations.append("Use ensemble of Prophet + LSTM + XGBoost")
            recommendations.append("Weight models: Prophet 40%, LSTM 30%, XGBoost 30%")
            recommendations.append("Consider adaptive weighting based on recent performance")

        # General recommendations
        if profile['data_quality']['outlier_pct'] > 5:
            recommendations.append("Consider outlier removal or robust scaling")

        if profile['data_quality']['zero_pct'] > 20:
            recommendations.append("Handle sparse data: consider aggregation or specialized models")

        return recommendations

    async def _get_sales_data(self, product_id: int) -> pd.DataFrame:
        """Get historical sales data"""
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
        df = df.groupby('date').agg({
            'quantity': 'sum',
            'unit_price': 'mean'
        }).reset_index()

        return df
