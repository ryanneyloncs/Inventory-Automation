# AI-Powered Inventory Management System

Complete, production-ready inventory management automation system with ML-powered demand forecasting, intelligent reordering, and supplier integration.

## Features

### Core Capabilities
- **Demand Forecasting**: Multiple ML models (Prophet, LSTM, Random Forest) with ensemble predictions
- **Inventory Optimization**: EOQ calculations, reorder point optimization, safety stock management
- **Automated Reordering**: Intelligent purchase order generation and supplier integration
- **Real-time Monitoring**: Low stock alerts, overdue order tracking, performance metrics
- **Supplier Management**: API integration for automated order placement and status tracking

### Technical Highlights
- **FastAPI** REST API with async/await
- **PostgreSQL** for reliable data storage
- **Redis** for caching and task queues
- **Docker** containerization for easy deployment
- **Prometheus & Grafana** for monitoring
- **Comprehensive Testing** with pytest

## Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional but recommended)

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone <your-repo>
cd inventory_automation

# Create environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Wait for services to be healthy (30 seconds)
sleep 30

# Seed the database
docker-compose exec api python scripts/seed_sample_data.py

# Access the API
open http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your database credentials

# Start PostgreSQL and Redis
# (Install and start separately or use Docker for just these services)
docker-compose up -d postgres redis

# Create database
createdb inventory_db

# Seed the database
python scripts/seed_sample_data.py

# Start the API
uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

## Usage Examples

### 1. Generate Demand Forecast

```bash
curl -X POST "http://localhost:8000/api/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": 1,
    "forecast_horizon": 90
  }'
```

### 2. Optimize Inventory Parameters

```bash
# Optimize specific product
curl -X POST "http://localhost:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -d '{"product_id": 1}'

# Optimize all products
curl -X POST "http://localhost:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 3. Check Reorder Needs

```bash
curl -X GET "http://localhost:8000/api/v1/reorder/check"
```

### 4. Automatic Reordering

```bash
curl -X POST "http://localhost:8000/api/v1/reorder/automatic"
```

### 5. Create Manual Purchase Order

```bash
curl -X POST "http://localhost:8000/api/v1/purchase-orders" \
  -H "Content-Type: application/json" \
  -d '{
    "supplier_id": 1,
    "items": [
      {"product_id": 1, "quantity": 100},
      {"product_id": 2, "quantity": 50}
    ],
    "auto_send": true
  }'
```

### 6. Get Low Stock Alerts

```bash
curl -X GET "http://localhost:8000/api/v1/alerts/low-stock"
```

## Architecture

```
inventory_automation/
├── config/
│   └── settings.py              # Configuration management
├── src/
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints
│   ├── models/
│   │   └── database.py          # SQLAlchemy models
│   ├── services/
│   │   ├── forecasting/
│   │   │   └── forecasting_service.py
│   │   ├── optimization/
│   │   │   └── optimization_service.py
│   │   ├── reorder/
│   │   │   └── reorder_service.py
│   │   └── suppliers/
│   │       └── supplier_client.py
│   └── database.py              # Database connection
├── scripts/
│   └── seed_sample_data.py      # Database seeding
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── deployments/
│   ├── docker/                  # Docker configurations
│   └── kubernetes/              # K8s manifests
├── ml_models/                   # Trained ML models
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_forecasting.py

# Run with verbose output
pytest -v
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

Key environment variables in `.env`:

```env
# Database
DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/inventory_db"

# ML Parameters
FORECAST_HORIZON_DAYS=90
MIN_TRAINING_SAMPLES=100

# Optimization
HOLDING_COST_PERCENTAGE=0.25
ORDERING_COST=50.0
SERVICE_LEVEL=0.95

# Reorder Logic
AUTO_REORDER_ENABLED=True
SAFETY_STOCK_Z_SCORE=1.65
```

## Monitoring

Access monitoring dashboards:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Scheduled Tasks

The system can run automated tasks:

```python
# Example: Daily optimization
# Add to your scheduler (cron, Celery Beat, etc.)

@app.task
async def daily_optimization():
    async with AsyncSessionLocal() as db:
        service = OptimizationService(db)
        await service.optimize_all_products()

@app.task
async def check_reorders():
    async with AsyncSessionLocal() as db:
        service = ReorderService(db)
        await service.process_automatic_reorders()
```

## Integration with ERP/CRM Systems

The system provides REST API endpoints for integration:

```python
# Example integration with your ERP
import httpx

async def sync_inventory_to_erp():
    async with httpx.AsyncClient() as client:
        # Get current inventory levels
        response = await client.get("http://localhost:8000/api/v1/inventory")
        
        # Update your ERP system
        await update_erp_inventory(response.json())
```

## Development

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

## Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Production Deployment

### Kubernetes

```bash
# Apply configurations
kubectl apply -f deployments/kubernetes/

# Check status
kubectl get pods -n inventory

# View logs
kubectl logs -f deployment/inventory-api -n inventory
```

### AWS/GCP/Azure

Deploy using container services:
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances

## Performance

Expected performance metrics:
- **API Response Time**: < 100ms for most endpoints
- **Forecast Generation**: 2-5 seconds per product
- **Optimization**: < 1 second per product
- **Concurrent Users**: 100+ with default settings

## Security

- API authentication via JWT tokens
- Database credentials via environment variables
- HTTPS in production (configure reverse proxy)
- Rate limiting on API endpoints
- Input validation on all endpoints

## Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres
```

### Model Training Fails
```bash
# Check you have enough historical data
# Minimum 100 data points required

# Increase MIN_TRAINING_SAMPLES in .env if needed
MIN_TRAINING_SAMPLES=50
```

### API Returns 500 Errors
```bash
# Check API logs
docker-compose logs api

# Verify database migrations
alembic current
alembic upgrade head
```

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prophet Forecasting](https://facebook.github.io/prophet/)
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/14/orm/extensions/asyncio.html)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Create a GitHub issue
- Email: ryanneyloncs@gmail.com

---

**Built for efficient inventory management**
