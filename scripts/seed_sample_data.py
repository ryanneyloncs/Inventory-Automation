import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
import random
from decimal import Decimal

from src.database import AsyncSessionLocal, init_db
from src.models.database import (
    Supplier, Product, InventoryLevel, SalesRecord, OptimizationParameters
)


async def seed_database():
    """Seed database with sample data"""
    
    print("=" * 70)
    print("SEEDING INVENTORY DATABASE")
    print("=" * 70)
    print()
    
    # Initialize database
    await init_db()
    
    async with AsyncSessionLocal() as session:
        # Create suppliers
        suppliers = [
            Supplier(
                name="ABC Manufacturing Co",
                contact_email="orders@abc-mfg.com",
                contact_phone="555-0101",
                address="123 Industrial Pkwy, Chicago, IL 60601",
                lead_time_days=7,
                minimum_order_quantity=50,
                api_endpoint="https://api.abc-mfg.com/v1",
                api_key="abc_api_key_12345",
                reliability_score=0.95,
                is_active=True
            ),
            Supplier(
                name="Global Distributors Inc",
                contact_email="sales@globaldist.com",
                contact_phone="555-0102",
                address="456 Commerce St, Los Angeles, CA 90001",
                lead_time_days=10,
                minimum_order_quantity=100,
                api_endpoint="https://api.globaldist.com",
                api_key="global_api_key_67890",
                reliability_score=0.88,
                is_active=True
            ),
            Supplier(
                name="FastShip Wholesale",
                contact_email="info@fastship.com",
                contact_phone="555-0103",
                address="789 Distribution Way, Dallas, TX 75201",
                lead_time_days=5,
                minimum_order_quantity=25,
                api_endpoint="https://api.fastship.com",
                api_key="fast_api_key_11111",
                reliability_score=0.92,
                is_active=True
            )
        ]
        
        session.add_all(suppliers)
        await session.commit()
        print(f"Created {len(suppliers)} suppliers")
        
        # Create products
        product_data = [
            {
                "sku": "WDG-001", "name": "Premium Widget A", "category": "Widgets",
                "unit_cost": 15.50, "selling_price": 29.99, "weight": 0.5
            },
            {
                "sku": "WDG-002", "name": "Standard Widget B", "category": "Widgets",
                "unit_cost": 10.25, "selling_price": 19.99, "weight": 0.4
            },
            {
                "sku": "GAD-001", "name": "Ultra Gadget Pro", "category": "Gadgets",
                "unit_cost": 45.00, "selling_price": 89.99, "weight": 1.2
            },
            {
                "sku": "GAD-002", "name": "Basic Gadget Lite", "category": "Gadgets",
                "unit_cost": 22.50, "selling_price": 44.99, "weight": 0.8
            },
            {
                "sku": "ACC-001", "name": "Universal Adapter", "category": "Accessories",
                "unit_cost": 5.75, "selling_price": 12.99, "weight": 0.2
            },
            {
                "sku": "ACC-002", "name": "Premium Cable Pack", "category": "Accessories",
                "unit_cost": 8.50, "selling_price": 17.99, "weight": 0.3
            },
            {
                "sku": "TOL-001", "name": "Professional Tool Set", "category": "Tools",
                "unit_cost": 120.00, "selling_price": 249.99, "weight": 5.0
            },
            {
                "sku": "TOL-002", "name": "Basic Tool Kit", "category": "Tools",
                "unit_cost": 35.00, "selling_price": 69.99, "weight": 2.0
            },
            {
                "sku": "ELC-001", "name": "Smart Controller", "category": "Electronics",
                "unit_cost": 75.00, "selling_price": 149.99, "weight": 1.5
            },
            {
                "sku": "ELC-002", "name": "Power Supply Unit", "category": "Electronics",
                "unit_cost": 28.00, "selling_price": 54.99, "weight": 1.0
            }
        ]
        
        products = []
        for i, pdata in enumerate(product_data):
            supplier = suppliers[i % len(suppliers)]
            
            product = Product(
                sku=pdata["sku"],
                name=pdata["name"],
                description=f"High-quality {pdata['name'].lower()} for professional use",
                category=pdata["category"],
                supplier_id=supplier.id,
                unit_cost=Decimal(str(pdata["unit_cost"])),
                selling_price=Decimal(str(pdata["selling_price"])),
                weight=pdata["weight"],
                dimensions={"length": 10, "width": 5, "height": 3},
                is_active=True
            )
            products.append(product)
        
        session.add_all(products)
        await session.commit()
        print(f"Created {len(products)} products")
        
        # Create inventory levels
        for product in products:
            initial_quantity = random.randint(50, 500)
            inventory = InventoryLevel(
                product_id=product.id,
                quantity_on_hand=initial_quantity,
                quantity_reserved=random.randint(0, 20),
                quantity_available=initial_quantity - random.randint(0, 20),
                warehouse_location=f"WH-{random.choice(['A', 'B', 'C'])}-{random.randint(1, 20):02d}",
                last_counted_at=datetime.utcnow() - timedelta(days=random.randint(1, 30))
            )
            session.add(inventory)
        
        await session.commit()
        print(f"Created inventory levels for all products")
        
        # Create historical sales data (last 2 years)
        print("Creating sales records (this may take a moment)...")
        sales_records = []
        sales_channels = ['online', 'retail', 'wholesale']
        customer_segments = ['enterprise', 'smb', 'individual']
        
        for product in products:
            # Base demand varies by product
            base_demand = random.randint(5, 30)
            demand_volatility = random.uniform(0.1, 0.4)
            
            # Generate 2 years of daily sales
            start_date = datetime.utcnow() - timedelta(days=730)
            
            for day in range(730):
                current_date = start_date + timedelta(days=day)
                
                # Add seasonality (higher sales in Q4)
                seasonal_factor = 1.0
                if current_date.month in [11, 12]:
                    seasonal_factor = 1.5
                elif current_date.month in [1, 2]:
                    seasonal_factor = 0.7
                
                # Add weekly pattern (lower on weekends)
                weekly_factor = 1.0
                if current_date.weekday() in [5, 6]:
                    weekly_factor = 0.6
                
                # Random promotion effect
                promotion_active = random.random() < 0.1  # 10% of days
                promo_factor = 1.3 if promotion_active else 1.0
                
                # Calculate quantity with all factors
                expected_quantity = base_demand * seasonal_factor * weekly_factor * promo_factor
                actual_quantity = max(0, int(random.gauss(expected_quantity, expected_quantity * demand_volatility)))
                
                if actual_quantity > 0:
                    revenue = actual_quantity * float(product.selling_price)
                    
                    record = SalesRecord(
                        product_id=product.id,
                        sale_date=current_date,
                        quantity=actual_quantity,
                        revenue=Decimal(str(revenue)),
                        customer_segment=random.choice(customer_segments),
                        sales_channel=random.choice(sales_channels),
                        promotion_active=promotion_active
                    )
                    sales_records.append(record)
        
        # Batch insert sales records
        batch_size = 1000
        for i in range(0, len(sales_records), batch_size):
            batch = sales_records[i:i+batch_size]
            session.add_all(batch)
            await session.commit()

        print(f"Created {len(sales_records)} sales records")
        
        # Create optimization parameters (placeholders - will be calculated by optimization service)
        for product in products:
            params = OptimizationParameters(
                product_id=product.id,
                economic_order_quantity=100,
                reorder_point=50,
                safety_stock=20,
                max_stock_level=200,
                holding_cost_per_unit=Decimal("3.00"),
                ordering_cost=Decimal("50.00"),
                stockout_cost=Decimal("20.00"),
                service_level=0.95,
                lead_time_days=7,
                demand_std_dev=5.0,
                calculation_method='initial'
            )
            session.add(params)
        
        await session.commit()
        print(f"Created optimization parameters for all products")

    print()
    print("=" * 70)
    print("DATABASE SEEDING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run optimization: POST /api/v1/optimize")
    print("2. Generate forecasts: POST /api/v1/forecast")
    print("3. Check reorder needs: GET /api/v1/reorder/check")
    print()


if __name__ == "__main__":
    asyncio.run(seed_database())
