# Logs Directory

Application logs are written here at runtime.

## Log Files

- `api.log` - API request/response logs
- `worker.log` - Celery worker logs
- `scheduler.log` - Scheduled task logs
- `error.log` - Error and exception logs
- `access.log` - HTTP access logs

## Log Format

```json
{
  "timestamp": "2024-10-28T12:00:00Z",
  "level": "INFO",
  "logger": "api.routes",
  "message": "Forecast generated for product 1",
  "context": {
    "product_id": 1,
    "duration_ms": 2341,
    "user_id": "user123"
  }
}
```

## Viewing Logs

### Docker Compose
```bash
# API logs
docker-compose logs -f api

# Worker logs
docker-compose logs -f worker

# All logs
docker-compose logs -f
```

### Kubernetes
```bash
kubectl logs -f deployment/inventory-api -n inventory-system
```

### Local Files
```bash
# Tail logs
tail -f logs/api.log

# Search logs
grep "ERROR" logs/*.log

# View last 100 lines
tail -n 100 logs/api.log
```

## Log Rotation

Logs are automatically rotated:
- Daily rotation
- Keep 7 days of logs
- Compress old logs
- Max size: 100MB per file

## Cleanup

```bash
# Remove old logs
rm logs/*.log

# Remove compressed logs
rm logs/*.log.gz
```

## Log Aggregation

For production, consider:
- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Loki**: Grafana Loki for log aggregation
- **CloudWatch**: AWS CloudWatch Logs
- **Datadog**: Datadog logging

## Notes

- This directory is empty by design
- Logs are created at runtime
- Add `logs/` to `.gitignore` (already done)
- Don't commit log files
