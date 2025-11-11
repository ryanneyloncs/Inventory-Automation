import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from src.models.database import Base
from config.settings import settings


# Override settings for testing
settings.DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/inventory_db_test"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine):
    """Create database session for tests"""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def sample_supplier(db_session):
    """Create a sample supplier for testing"""
    from src.models.database import Supplier
    
    supplier = Supplier(
        name="Test Supplier",
        contact_email="test@supplier.com",
        lead_time_days=7,
        minimum_order_quantity=10,
        is_active=True
    )
    
    db_session.add(supplier)
    await db_session.commit()
    await db_session.refresh(supplier)
    
    return supplier


@pytest.fixture
async def sample_product(db_session, sample_supplier):
    """Create a sample product for testing"""
    from src.models.database import Product
    from decimal import Decimal
    
    product = Product(
        sku="TEST-001",
        name="Test Product",
        category="Test",
        supplier_id=sample_supplier.id,
        unit_cost=Decimal("10.00"),
        selling_price=Decimal("20.00"),
        is_active=True
    )
    
    db_session.add(product)
    await db_session.commit()
    await db_session.refresh(product)
    
    return product
