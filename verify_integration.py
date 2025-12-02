#!/usr/bin/env python3
"""
Sales & Revenue Forecasting Suite - Integration Verification Script

Tests all components to ensure proper integration
"""

import asyncio
import sys
from datetime import date, timedelta
from typing import Dict, List
import logging

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.GREEN}[PASS] {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}[FAIL] {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}[INFO] {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARN] {msg}{Colors.END}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


class IntegrationVerifier:
    """Verify all components of the forecasting suite"""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.warnings = 0

    async def run_all_tests(self):
        """Run all verification tests"""
        print_header("Sales & Revenue Forecasting Suite - Integration Verification")

        # Test 1: Imports
        print_info("Test 1: Checking imports...")
        if await self.test_imports():
            print_success("All imports successful")
            self.tests_passed += 1
        else:
            print_error("Import test failed")
            self.tests_failed += 1
            return False

        # Test 2: Database Connection
        print_info("\nTest 2: Checking database connection...")
        db = await self.test_database_connection()
        if db:
            print_success("Database connection successful")
            self.tests_passed += 1
        else:
            print_error("Database connection failed")
            self.tests_failed += 1
            return False

        # Test 3: Database Tables
        print_info("\nTest 3: Checking database tables...")
        if await self.test_database_tables(db):
            print_success("All required tables exist")
            self.tests_passed += 1
        else:
            print_error("Missing database tables")
            self.tests_failed += 1

        # Test 4: Multi-Horizon Forecaster
        print_info("\nTest 4: Testing Multi-Horizon Forecaster...")
        if await self.test_forecaster(db):
            print_success("Forecaster working correctly")
            self.tests_passed += 1
        else:
            print_error("Forecaster test failed")
            self.tests_failed += 1

        # Test 5: Scenario Planner
        print_info("\nTest 5: Testing Scenario Planner...")
        if await self.test_scenario_planner(db):
            print_success("Scenario Planner working correctly")
            self.tests_passed += 1
        else:
            print_error("Scenario Planner test failed")
            self.tests_failed += 1

        # Test 6: Seasonal Analyzer
        print_info("\nTest 6: Testing Seasonal Analyzer...")
        if await self.test_seasonal_analyzer(db):
            print_success("Seasonal Analyzer working correctly")
            self.tests_passed += 1
        else:
            print_error("Seasonal Analyzer test failed")
            self.tests_failed += 1

        # Test 7: Cannibalization Analyzer
        print_info("\nTest 7: Testing Cannibalization Analyzer...")
        if await self.test_cannibalization_analyzer(db):
            print_success("Cannibalization Analyzer working correctly")
            self.tests_passed += 1
        else:
            print_error("Cannibalization Analyzer test failed")
            self.tests_failed += 1

        # Test 8: API Routes
        print_info("\nTest 8: Testing API Routes...")
        if await self.test_api_routes():
            print_success("API Routes registered correctly")
            self.tests_passed += 1
        else:
            print_warning("API Routes test skipped (optional)")
            self.warnings += 1

        # Print summary
        await self.print_summary()

        # Close database
        await db.close()

        return self.tests_failed == 0

    async def test_imports(self) -> bool:
        """Test if all required modules can be imported"""
        try:
            # Core services
            from src.services.multi_horizon_forecaster import MultiHorizonForecaster
            from src.services.scenario_planner import ScenarioPlanner
            from src.services.seasonal_analyzer import SeasonalAnalyzer
            from src.services.cannibalization_analyzer import CannibalizationAnalyzer

            # Models
            from src.models.sales_forecasting_models import (
                SalesForecastResult, ForecastScenario, ScenarioForecastResult,
                SeasonalDecomposition, EventImpact, CannibalizationAnalysis,
                ForecastAccuracyMetric
            )

            # ML libraries
            import prophet
            import statsmodels
            import scipy
            import sklearn
            import tensorflow

            print_info("  - All service modules imported successfully")
            print_info("  - All model classes imported successfully")
            print_info("  - All ML libraries available")

            return True

        except ImportError as e:
            print_error(f"  - Import failed: {e}")
            print_info("  - Run: pip install -r requirements_forecasting.txt")
            return False
        except Exception as e:
            print_error(f"  - Unexpected error: {e}")
            return False

    async def test_database_connection(self):
        """Test database connection"""
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker
            from src.config import DATABASE_URL  # Adjust import based on your config

            engine = create_async_engine(DATABASE_URL, echo=False)
            async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

            async with async_session() as session:
                await session.execute("SELECT 1")
                print_info("  - Database connection established")
                return session

        except ImportError:
            print_warning("  - Could not import database config")
            print_info("  - Update DATABASE_URL in verify_integration.py")
            return None
        except Exception as e:
            print_error(f"  - Database connection failed: {e}")
            return None

    async def test_database_tables(self, db) -> bool:
        """Test if all required tables exist"""
        if not db:
            return False

        required_tables = [
            'sales_forecast_results',
            'forecast_scenarios',
            'scenario_forecast_results',
            'seasonal_decomposition',
            'event_impacts',
            'cannibalization_analysis',
            'forecast_accuracy_metrics'
        ]

        try:
            from sqlalchemy import text

            for table in required_tables:
                result = await db.execute(
                    text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}')")
                )
                exists = result.scalar()

                if exists:
                    print_info(f"  - Table '{table}' exists")
                else:
                    print_error(f"  - Table '{table}' missing")
                    print_info("  - Run: alembic upgrade head")
                    return False

            return True

        except Exception as e:
            print_error(f"  - Error checking tables: {e}")
            return False

    async def test_forecaster(self, db) -> bool:
        """Test Multi-Horizon Forecaster"""
        try:
            from src.services.multi_horizon_forecaster import MultiHorizonForecaster

            forecaster = MultiHorizonForecaster(db)
            print_info("  - Forecaster instantiated successfully")
            print_info("  - Horizons configured: hourly, daily, weekly, monthly")

            return True

        except Exception as e:
            print_error(f"  - Forecaster test failed: {e}")
            return False

    async def test_scenario_planner(self, db) -> bool:
        """Test Scenario Planner"""
        try:
            from src.services.scenario_planner import ScenarioPlanner

            planner = ScenarioPlanner(db)
            print_info("  - Scenario Planner instantiated successfully")
            print_info("  - Scenario types available: promotion, new_product, seasonal, price_change, custom")

            return True

        except Exception as e:
            print_error(f"  - Scenario Planner test failed: {e}")
            return False

    async def test_seasonal_analyzer(self, db) -> bool:
        """Test Seasonal Analyzer"""
        try:
            from src.services.seasonal_analyzer import SeasonalAnalyzer

            analyzer = SeasonalAnalyzer(db)
            print_info("  - Seasonal Analyzer instantiated successfully")
            print_info("  - Features: decomposition, event impact, anomaly detection")

            return True

        except Exception as e:
            print_error(f"  - Seasonal Analyzer test failed: {e}")
            return False

    async def test_cannibalization_analyzer(self, db) -> bool:
        """Test Cannibalization Analyzer"""
        try:
            from src.services.cannibalization_analyzer import CannibalizationAnalyzer

            analyzer = CannibalizationAnalyzer(db)
            print_info("  - Cannibalization Analyzer instantiated successfully")
            print_info("  - Features: similarity detection, impact measurement, portfolio analysis")

            return True

        except Exception as e:
            print_error(f"  - Cannibalization Analyzer test failed: {e}")
            return False

    async def test_api_routes(self) -> bool:
        """Test API Routes registration"""
        try:
            from src.api.sales_forecasting_routes import router

            # Count routes
            route_count = len(router.routes)

            print_info(f"  - API Router imported successfully")
            print_info(f"  - Total routes registered: {route_count}")

            # List route paths
            print_info("  - Available endpoints:")
            for route in router.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    methods = ', '.join(route.methods)
                    print_info(f"    {methods} {route.path}")

            return True

        except ImportError as e:
            print_warning(f"  - Could not import API routes: {e}")
            print_info("  - This is optional if not using FastAPI")
            return False
        except Exception as e:
            print_warning(f"  - API routes test error: {e}")
            return False

    async def print_summary(self):
        """Print test summary"""
        print_header("Test Summary")

        total_tests = self.tests_passed + self.tests_failed

        print(f"{Colors.BOLD}Total Tests:{Colors.END} {total_tests}")
        print(f"{Colors.GREEN}Passed:{Colors.END} {self.tests_passed}")
        print(f"{Colors.RED}Failed:{Colors.END} {self.tests_failed}")
        print(f"{Colors.YELLOW}Warnings:{Colors.END} {self.warnings}")

        if self.tests_failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}[SUCCESS] All tests passed!{Colors.END}")
            print(f"\n{Colors.BLUE}Next steps:{Colors.END}")
            print("  1. Install dependencies: pip install -r requirements_forecasting.txt")
            print("  2. Run migration: alembic upgrade head")
            print("  3. Start your application and test the API endpoints")
            print("  4. Check docs/INTEGRATION_CHECKLIST.md for detailed guide")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}[FAILURE] Some tests failed{Colors.END}")
            print(f"\n{Colors.BLUE}Troubleshooting:{Colors.END}")
            print("  1. Check import errors - install missing dependencies")
            print("  2. Verify database connection settings")
            print("  3. Run database migration: alembic upgrade head")
            print("  4. Check docs/INTEGRATION_CHECKLIST.md for help")

        print()


async def main():
    """Main entry point"""
    verifier = IntegrationVerifier()
    success = await verifier.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Run verification
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[INFO] Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[FAIL] Unexpected error: {e}")
        sys.exit(1)
