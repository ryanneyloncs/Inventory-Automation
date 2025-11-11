from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
import secrets


def generate_secret_key() -> str:
    """Generate a secure random secret key"""
    return secrets.token_urlsafe(32)


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application
    APP_NAME: str = "Inventory Management Automation"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/inventory_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 10

    # API
    API_PREFIX: str = "/api/v1"
    API_TITLE: str = "Inventory Management API"
    API_DESCRIPTION: str = "AI-powered inventory management and forecasting"
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]

    # Security
    SECRET_KEY: str = ""  # Must be set via environment variable in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML Models
    FORECASTING_MODELS_PATH: str = "./ml_models/forecasting"
    MODEL_RETRAIN_INTERVAL_DAYS: int = 7
    MIN_TRAINING_SAMPLES: int = 100
    
    # Forecasting
    FORECAST_HORIZON_DAYS: int = 90
    SEASONALITY_PERIOD: int = 7
    CONFIDENCE_INTERVAL: float = 0.95
    
    # Optimization
    HOLDING_COST_PERCENTAGE: float = 0.25
    ORDERING_COST: float = 50.0
    STOCKOUT_COST_MULTIPLIER: float = 2.0
    LEAD_TIME_DAYS: int = 7
    SERVICE_LEVEL: float = 0.95
    
    # Reorder Logic
    SAFETY_STOCK_Z_SCORE: float = 1.65  # 95% service level
    REORDER_CHECK_INTERVAL_HOURS: int = 6
    AUTO_REORDER_ENABLED: bool = True
    MIN_ORDER_QUANTITY: int = 1
    MAX_ORDER_QUANTITY: int = 10000
    
    # Supplier API
    SUPPLIER_API_TIMEOUT: int = 30
    SUPPLIER_API_RETRY_ATTEMPTS: int = 3
    SUPPLIER_API_RETRY_DELAY: int = 2
    
    # Email Configuration
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAIL_FROM: str = "noreply@inventory.com"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Data Processing
    BATCH_SIZE: int = 1000
    MAX_WORKERS: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_production_settings()

    def _validate_production_settings(self):
        """Validate critical settings for production environments"""

        # If no SECRET_KEY is set, generate one for development or fail for production
        if not self.SECRET_KEY or self.SECRET_KEY == "your-secret-key-change-this-in-production":
            if self.ENVIRONMENT == "production":
                raise ValueError(
                    "SECRET_KEY must be set in production environment. "
                    "Generate one using: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )
            else:
                # Auto-generate for development
                self.SECRET_KEY = generate_secret_key()
                print(f"WARNING: Using auto-generated SECRET_KEY for {self.ENVIRONMENT} environment")
                print(f"   For production, set SECRET_KEY in environment variables")

        # Validate production-specific requirements
        if self.ENVIRONMENT == "production":
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production")

            if "localhost" in self.DATABASE_URL:
                print("WARNING: Using localhost database in production environment")

            if "change-this" in self.DATABASE_URL.lower():
                raise ValueError("Database credentials must be updated for production")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
