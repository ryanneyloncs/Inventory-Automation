# Docker Swarm Deployment

Production deployment using Docker Swarm orchestration.

## Files

- **docker-stack.yml** - Complete Docker Swarm stack configuration
- **prometheus.yml** - Prometheus monitoring configuration
- **README.md** - This file

## Quick Deploy

### Prerequisites

```bash
# Initialize Docker Swarm
docker swarm init

# Create secrets
echo "your-strong-postgres-password" | docker secret create db_password -
echo "your-api-secret-key-change-in-prod" | docker secret create api_secret_key -
echo "admin" | docker secret create grafana_password -
```

### Deploy Stack

```bash
# Set registry and version
export REGISTRY=your-registry.io/yourproject
export VERSION=v1.0.0

# Deploy
docker stack deploy -c docker-stack.yml inventory

# Check status
docker stack services inventory
docker service ls
```

## Services

| Service | Replicas | Ports | Resources |
|---------|----------|-------|-----------|
| postgres | 1 | - | 512M-1G |
| redis | 1 | - | 256M-512M |
| api | 3 | 8000 | 512M-2G |
| worker | 2 | - | 1G-4G |
| scheduler | 1 | - | 512M-1G |
| prometheus | 1 | 9090 | Default |
| grafana | 1 | 3000 | Default |

## Updates

```bash
# Update service
docker service update --image your-registry/inventory-api:v2 inventory_api

# Scale service
docker service scale inventory_api=5

# Rollback
docker service rollback inventory_api
```

## Monitoring

```bash
# Service logs
docker service logs -f inventory_api

# Service status
docker service ps inventory_api

# Stack status
docker stack ps inventory
```

## Features

- **High Availability**: 3 API replicas, 2 workers
- **Rolling Updates**: Zero-downtime deployments
- **Secret Management**: Docker secrets for credentials
- **Service Discovery**: Automatic DNS between services
- **Load Balancing**: Built-in load balancer
- **Health Checks**: Automatic restart on failure
- **Resource Limits**: CPU and memory constraints
- **Monitoring**: Prometheus + Grafana

## Security

```bash
# Rotate secrets
echo "new-password" | docker secret create db_password_v2 -
docker service update --secret-rm db_password --secret-add db_password_v2 inventory_postgres
```

## Cleanup

```bash
# Remove stack
docker stack rm inventory

# Remove volumes (⚠️ deletes data)
docker volume rm inventory_postgres_data inventory_redis_data
```

## Notes

- Requires Docker Swarm mode
- Services use overlay network
- Persistent data in named volumes
- Secrets stored encrypted
- Automatic service recovery
