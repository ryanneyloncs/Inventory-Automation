"""
Product Cannibalization Analysis Service

Analyzes how new product launches affect existing product sales:
- Product similarity detection using text analysis
- Sales impact measurement
- Cannibalization rate calculation
- Portfolio optimization insights
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

from src.models.database import SalesRecord, Product
from src.models.sales_forecasting_models import CannibalizationAnalysis

logger = logging.getLogger(__name__)


class CannibalizationAnalyzer:
    """
    Analyze product cannibalization and portfolio effects
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def analyze_new_product_launch(
        self,
        new_product_id: int,
        launch_date: date,
        analysis_window_days: int = 90,
        similarity_threshold: float = 0.3
    ) -> Dict:
        """
        Analyze cannibalization impact of a new product launch

        Args:
            new_product_id: ID of the new product
            launch_date: Date when product was launched
            analysis_window_days: Days after launch to analyze (default: 90)
            similarity_threshold: Minimum similarity score to consider (0-1)

        Returns:
            Dict with cannibalization analysis
        """
        new_product = await self._get_product(new_product_id)

        # Find similar products
        similar_products = await self._find_similar_products(
            new_product=new_product,
            similarity_threshold=similarity_threshold
        )

        if not similar_products:
            logger.info(f"No similar products found for product {new_product_id}")
            return {
                'new_product_id': new_product_id,
                'new_product_name': new_product.name,
                'similar_products_found': 0,
                'cannibalization_detected': False,
                'message': 'No similar products found for comparison'
            }

        # Analyze cannibalization for each similar product
        cannibalization_results = []

        for similar_product_id, similarity_score in similar_products:
            result = await self._analyze_product_pair_cannibalization(
                new_product_id=new_product_id,
                existing_product_id=similar_product_id,
                launch_date=launch_date,
                analysis_window_days=analysis_window_days,
                similarity_score=similarity_score
            )

            if result:
                cannibalization_results.append(result)

        # Calculate aggregate metrics
        if cannibalization_results:
            total_cannibalized_sales = sum(r['cannibalized_sales'] for r in cannibalization_results)
            new_product_sales = cannibalization_results[0]['new_product_sales']  # Same for all

            aggregate = {
                'new_product_id': new_product_id,
                'new_product_name': new_product.name,
                'launch_date': launch_date.isoformat(),
                'analysis_window_days': analysis_window_days,
                'similar_products_analyzed': len(cannibalization_results),
                'new_product_sales': float(new_product_sales),
                'total_cannibalized_sales': float(total_cannibalized_sales),
                'net_incremental_sales': float(new_product_sales - total_cannibalized_sales),
                'cannibalization_rate_pct': float((total_cannibalized_sales / new_product_sales * 100)) if new_product_sales > 0 else 0,
                'products': cannibalization_results
            }
        else:
            aggregate = {
                'new_product_id': new_product_id,
                'new_product_name': new_product.name,
                'similar_products_found': len(similar_products),
                'cannibalization_detected': False,
                'message': 'Insufficient data to measure cannibalization'
            }

        logger.info(f"Cannibalization analysis complete for product {new_product_id}")
        return aggregate

    async def _find_similar_products(
        self,
        new_product: Product,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Find products similar to the new product using text similarity

        Returns:
            List of (product_id, similarity_score) tuples
        """
        # Get all products except the new one
        query = select(Product).where(Product.id != new_product.id)
        result = await self.db.execute(query)
        all_products = result.scalars().all()

        if not all_products:
            return []

        # Prepare text data for similarity analysis
        new_product_text = self._product_to_text(new_product)
        product_texts = [self._product_to_text(p) for p in all_products]
        all_texts = [new_product_text] + product_texts

        # Calculate TF-IDF similarity
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Calculate cosine similarity between new product and all others
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

            # Get products above threshold
            similar_products = [
                (all_products[i].id, float(similarities[i]))
                for i in range(len(similarities))
                if similarities[i] >= similarity_threshold
            ]

            # Sort by similarity (descending)
            similar_products.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"Found {len(similar_products)} similar products for product {new_product.id}")
            return similar_products

        except Exception as e:
            logger.error(f"Error calculating product similarity: {e}")
            return []

    def _product_to_text(self, product: Product) -> str:
        """
        Convert product attributes to text for similarity analysis
        """
        text_parts = [
            product.name or '',
            product.category or '',
            product.description or ''
        ]
        return ' '.join(text_parts)

    async def _analyze_product_pair_cannibalization(
        self,
        new_product_id: int,
        existing_product_id: int,
        launch_date: date,
        analysis_window_days: int,
        similarity_score: float
    ) -> Optional[Dict]:
        """
        Analyze cannibalization between two specific products
        """
        # Get baseline sales (before launch)
        baseline_start = launch_date - timedelta(days=analysis_window_days)
        baseline_end = launch_date - timedelta(days=1)

        baseline_sales = await self._get_product_sales_in_period(
            product_id=existing_product_id,
            start_date=baseline_start,
            end_date=baseline_end
        )

        # Get post-launch sales
        post_launch_start = launch_date
        post_launch_end = launch_date + timedelta(days=analysis_window_days)

        post_launch_sales = await self._get_product_sales_in_period(
            product_id=existing_product_id,
            start_date=post_launch_start,
            end_date=post_launch_end
        )

        # Get new product sales
        new_product_sales_data = await self._get_product_sales_in_period(
            product_id=new_product_id,
            start_date=post_launch_start,
            end_date=post_launch_end
        )

        if baseline_sales.empty or post_launch_sales.empty or new_product_sales_data.empty:
            return None

        # Calculate metrics
        baseline_total = baseline_sales['quantity'].sum()
        post_launch_total = post_launch_sales['quantity'].sum()
        new_product_total = new_product_sales_data['quantity'].sum()

        sales_decline = baseline_total - post_launch_total
        cannibalization_rate = (sales_decline / new_product_total * 100) if new_product_total > 0 else 0

        # Statistical significance test
        baseline_daily = baseline_sales.groupby('date')['quantity'].sum()
        post_launch_daily = post_launch_sales.groupby('date')['quantity'].sum()

        is_significant = self._test_statistical_significance(
            baseline_daily.values,
            post_launch_daily.values
        )

        # Get product details
        existing_product = await self._get_product(existing_product_id)

        # Save to database
        cannibalization_record = CannibalizationAnalysis(
            new_product_id=new_product_id,
            existing_product_id=existing_product_id,
            analysis_date=date.today(),
            similarity_score=similarity_score,
            baseline_sales_existing=float(baseline_total),
            post_launch_sales_existing=float(post_launch_total),
            cannibalization_rate=float(cannibalization_rate),
            new_product_sales=float(new_product_total),
            net_incremental_sales=float(new_product_total - sales_decline)
        )
        self.db.add(cannibalization_record)
        await self.db.commit()

        return {
            'existing_product_id': existing_product_id,
            'existing_product_name': existing_product.name,
            'similarity_score': float(similarity_score),
            'baseline_sales': float(baseline_total),
            'post_launch_sales': float(post_launch_total),
            'sales_decline': float(sales_decline),
            'sales_decline_pct': float((sales_decline / baseline_total * 100)) if baseline_total > 0 else 0,
            'cannibalized_sales': float(max(0, sales_decline)),  # Only positive declines
            'new_product_sales': float(new_product_total),
            'cannibalization_rate_pct': float(max(0, cannibalization_rate)),  # Only positive
            'is_statistically_significant': is_significant
        }

    def _test_statistical_significance(
        self,
        baseline_data: np.ndarray,
        post_launch_data: np.ndarray,
        alpha: float = 0.05
    ) -> bool:
        """
        Test if sales decline is statistically significant
        """
        if len(baseline_data) < 2 or len(post_launch_data) < 2:
            return False

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(baseline_data, post_launch_data)

        return p_value < alpha and baseline_data.mean() > post_launch_data.mean()

    async def get_portfolio_cannibalization_matrix(
        self,
        product_ids: Optional[List[int]] = None
    ) -> Dict:
        """
        Get cannibalization matrix for entire product portfolio

        Args:
            product_ids: Specific products to analyze (or all if None)

        Returns:
            Matrix showing cannibalization relationships
        """
        # Get cannibalization records
        query = select(CannibalizationAnalysis)

        if product_ids:
            query = query.where(
                CannibalizationAnalysis.new_product_id.in_(product_ids)
            )

        result = await self.db.execute(query)
        records = result.scalars().all()

        if not records:
            return {
                'matrix': [],
                'message': 'No cannibalization data available'
            }

        # Build matrix
        matrix_data = []
        for record in records:
            new_product = await self._get_product(record.new_product_id)
            existing_product = await self._get_product(record.existing_product_id)

            matrix_data.append({
                'new_product_id': record.new_product_id,
                'new_product_name': new_product.name,
                'existing_product_id': record.existing_product_id,
                'existing_product_name': existing_product.name,
                'similarity_score': float(record.similarity_score),
                'cannibalization_rate': float(record.cannibalization_rate),
                'net_incremental_sales': float(record.net_incremental_sales),
                'analysis_date': record.analysis_date.isoformat()
            })

        # Sort by cannibalization rate (descending)
        matrix_data.sort(key=lambda x: x['cannibalization_rate'], reverse=True)

        # Calculate summary statistics
        total_records = len(matrix_data)
        high_cannibalization = sum(1 for r in matrix_data if r['cannibalization_rate'] > 50)
        moderate_cannibalization = sum(1 for r in matrix_data if 20 <= r['cannibalization_rate'] <= 50)
        low_cannibalization = sum(1 for r in matrix_data if r['cannibalization_rate'] < 20)

        return {
            'total_relationships': total_records,
            'summary': {
                'high_cannibalization': high_cannibalization,
                'moderate_cannibalization': moderate_cannibalization,
                'low_cannibalization': low_cannibalization
            },
            'matrix': matrix_data
        }

    async def get_product_cannibalization_history(
        self,
        product_id: int
    ) -> Dict:
        """
        Get cannibalization history for a specific product

        Args:
            product_id: Product to analyze

        Returns:
            Historical cannibalization data
        """
        product = await self._get_product(product_id)

        # Get records where this product is the new product
        query_as_new = select(CannibalizationAnalysis).where(
            CannibalizationAnalysis.new_product_id == product_id
        )
        result_new = await self.db.execute(query_as_new)
        as_new_product = result_new.scalars().all()

        # Get records where this product was cannibalized
        query_as_existing = select(CannibalizationAnalysis).where(
            CannibalizationAnalysis.existing_product_id == product_id
        )
        result_existing = await self.db.execute(query_as_existing)
        as_existing_product = result_existing.scalars().all()

        # Format results
        cannibalized_others = []
        for record in as_new_product:
            existing = await self._get_product(record.existing_product_id)
            cannibalized_others.append({
                'affected_product_id': record.existing_product_id,
                'affected_product_name': existing.name,
                'cannibalization_rate': float(record.cannibalization_rate),
                'net_incremental_sales': float(record.net_incremental_sales),
                'analysis_date': record.analysis_date.isoformat()
            })

        cannibalized_by_others = []
        for record in as_existing_product:
            new_prod = await self._get_product(record.new_product_id)
            cannibalized_by_others.append({
                'cannibalizing_product_id': record.new_product_id,
                'cannibalizing_product_name': new_prod.name,
                'cannibalization_rate': float(record.cannibalization_rate),
                'sales_lost': float(record.baseline_sales_existing - record.post_launch_sales_existing),
                'analysis_date': record.analysis_date.isoformat()
            })

        return {
            'product_id': product_id,
            'product_name': product.name,
            'as_new_product': {
                'count': len(cannibalized_others),
                'details': cannibalized_others,
                'total_net_incremental': sum(r['net_incremental_sales'] for r in cannibalized_others)
            },
            'as_existing_product': {
                'count': len(cannibalized_by_others),
                'details': cannibalized_by_others,
                'total_sales_lost': sum(r['sales_lost'] for r in cannibalized_by_others)
            }
        }

    async def recommend_product_positioning(
        self,
        new_product_id: int,
        max_cannibalization_rate: float = 30.0
    ) -> Dict:
        """
        Recommend product positioning to minimize cannibalization

        Args:
            new_product_id: Product to position
            max_cannibalization_rate: Maximum acceptable cannibalization rate

        Returns:
            Recommendations for product positioning
        """
        new_product = await self._get_product(new_product_id)

        # Find similar products
        similar_products = await self._find_similar_products(
            new_product=new_product,
            similarity_threshold=0.2  # Lower threshold for recommendations
        )

        if not similar_products:
            return {
                'product_id': new_product_id,
                'product_name': new_product.name,
                'recommendation': 'No similar products found - low cannibalization risk',
                'similar_products': []
            }

        # Analyze each similar product
        recommendations = []
        high_risk_products = []

        for similar_id, similarity in similar_products[:10]:  # Top 10 most similar
            similar_product = await self._get_product(similar_id)

            # Estimate potential cannibalization
            estimated_cannibalization = similarity * 100  # Simple estimate

            risk_level = (
                'high' if estimated_cannibalization > max_cannibalization_rate
                else 'moderate' if estimated_cannibalization > 15
                else 'low'
            )

            recommendation = {
                'product_id': similar_id,
                'product_name': similar_product.name,
                'similarity_score': float(similarity),
                'estimated_cannibalization_pct': float(estimated_cannibalization),
                'risk_level': risk_level
            }

            recommendations.append(recommendation)

            if risk_level == 'high':
                high_risk_products.append(similar_product.name)

        # Generate overall recommendation
        if high_risk_products:
            overall_recommendation = (
                f"High cannibalization risk with {len(high_risk_products)} products: "
                f"{', '.join(high_risk_products[:3])}. "
                f"Consider: (1) Different pricing strategy, "
                f"(2) Distinct target market positioning, "
                f"(3) Unique feature differentiation."
            )
        elif any(r['risk_level'] == 'moderate' for r in recommendations):
            overall_recommendation = (
                "Moderate cannibalization risk detected. "
                "Recommend clear product differentiation and targeted marketing."
            )
        else:
            overall_recommendation = (
                "Low cannibalization risk. Product appears well-differentiated from portfolio."
            )

        return {
            'product_id': new_product_id,
            'product_name': new_product.name,
            'overall_recommendation': overall_recommendation,
            'max_acceptable_cannibalization': max_cannibalization_rate,
            'similar_products_analyzed': len(recommendations),
            'high_risk_count': len(high_risk_products),
            'details': recommendations
        }

    async def _get_product_sales_in_period(
        self,
        product_id: int,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Get sales data for a product in a specific period
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

    async def _get_product(self, product_id: int) -> Product:
        """Get product by ID"""
        query = select(Product).where(Product.id == product_id)
        result = await self.db.execute(query)
        product = result.scalar_one_or_none()

        if not product:
            raise ValueError(f"Product {product_id} not found")

        return product
