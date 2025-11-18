#!/bin/bash

echo "=========================================="
echo "Inventory Management System - Quick Start"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "Docker and Docker Compose are installed"
echo ""

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ".env file created"
else
    echo ".env file already exists"
fi

echo ""
echo "Starting Docker containers..."
docker-compose up -d

echo ""
echo "Waiting for services to be healthy (30 seconds)..."
sleep 30

echo ""
echo "Seeding database with sample data..."
docker-compose exec -T api python scripts/seed_sample_data.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Your inventory management system is ready!"
echo ""
echo "Access Points:"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • API ReDoc:         http://localhost:8000/redoc"
echo "   • Grafana:           http://localhost:3000 (admin/admin)"
echo "   • Prometheus:        http://localhost:9090"
echo ""
echo "Quick Commands:"
echo "   • View logs:         docker-compose logs -f api"
echo "   • Stop services:     docker-compose down"
echo "   • Restart services:  docker-compose restart"
echo ""
echo "Next Steps:"
echo "   1. Open http://localhost:8000/docs"
echo "   2. Try the /api/v1/optimize endpoint to optimize inventory"
echo "   3. Try the /api/v1/forecast endpoint to generate forecasts"
echo "   4. Try the /api/v1/reorder/check to see reorder needs"
echo ""
echo "Full documentation available in README.md"
echo ""
