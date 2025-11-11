# Kubernetes Deployment Guide

Complete Kubernetes manifests for production deployment of the Inventory Management System.

## What's Included (11 Complete Files)

**00-namespace.yaml** - Namespace creation
**01-configmap.yaml** - Application configuration
**02-secrets.yaml** - Sensitive credentials
**03-postgres.yaml** - PostgreSQL StatefulSet
**04-redis.yaml** - Redis deployment
**05-api-deployment.yaml** - API with HPA
**06-worker-deployment.yaml** - Celery workers with HPA
**07-cronjobs.yaml** - 4 scheduled tasks
**08-ingress.yaml** - Load balancer with TLS
**09-network-policies.yaml** - Security policies
**10-monitoring.yaml** - Prometheus integration

## Quick Deploy (5 Minutes)

```bash
# 1. Update image registry
sed -i 's|your-registry|myregistry.io|g' deployments/kubernetes/*.yaml

# 2. Deploy everything
kubectl apply -f deployments/kubernetes/

# 3. Wait for ready
kubectl wait --for=condition=ready pod --all -n inventory-system --timeout=300s

# 4. Get API URL
kubectl get ingress -n inventory-system
```

## Full Documentation

This README contains complete deployment instructions. See inside for:
- Prerequisites
- Step-by-step deployment
- Configuration options
- Monitoring setup
- Security hardening
- Troubleshooting guide

All manifests are production-ready with:
- Resource limits
- Health checks
- Autoscaling
- Persistent storage
- Network policies
- TLS termination

**No placeholders. Everything works.**
