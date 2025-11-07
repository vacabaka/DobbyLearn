# ROMA-DSPy Deployment Guide

Production deployment guide for ROMA-DSPy.

## Table of Contents

- [Overview](#overview)
- [Quick Deploy](#quick-deploy)
- [Architecture](#architecture)
- [Environment Configuration](#environment-configuration)
- [Docker Deployment](#docker-deployment)
- [Production Checklist](#production-checklist)
- [Monitoring & Observability](#monitoring--observability)
- [Scaling](#scaling)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

---

## Overview

ROMA-DSPy is designed for production deployment with Docker Compose, providing:

**Infrastructure:**
- PostgreSQL (execution/checkpoint persistence)
- MinIO (S3-compatible object storage for MLflow artifacts)
- MLflow (optional, experiment tracking)
- ROMA API (FastAPI server)

**Features:**
- Health checks and auto-restart
- Volume persistence
- Network isolation
- Multi-stage Docker builds
- Non-root containers

---

## Quick Deploy

### Prerequisites

- Docker 24.0+ and Docker Compose 2.0+
- 4GB RAM minimum (8GB recommended)
- 20GB disk space
- Ports available: 8000 (API), 5432 (Postgres), 9000/9001 (MinIO), 5000 (MLflow)

### 1. Clone Repository

```bash
git clone https://github.com/your-org/ROMA-DSPy.git
cd ROMA-DSPy
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set required values
nano .env
```

**Minimum Required:**
```bash
# LLM Provider
OPENROUTER_API_KEY=your_key_here

# Database
POSTGRES_PASSWORD=secure_password_here

# MinIO/S3
MINIO_ROOT_PASSWORD=secure_password_here
```

### 3. Start Services

```bash
# Basic deployment (API + PostgreSQL + MinIO)
just docker-up

# Full deployment (includes MLflow observability)
just docker-up-full

# Verify health
curl http://localhost:8000/health
```

### 4. Test

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/executions \
  -H "Content-Type: application/json" \
  -d '{"goal": "What is 2+2?", "max_depth": 1}' | jq

# Via CLI (inside container)
docker exec -it roma-dspy-api roma-dspy solve "What is 2+2?"
```

---

## Architecture

### Docker Compose Stack

```
┌──────────────────────────────────────────────────────┐
│                  Docker Network                       │
│                                                       │
│  ┌──────────────┐    ┌──────────────┐               │
│  │   ROMA API   │───▶│  PostgreSQL  │               │
│  │  Port: 8000  │    │  Port: 5432  │               │
│  └──────┬───────┘    └──────────────┘               │
│         │                                             │
│         │            ┌──────────────┐                │
│         └───────────▶│    MinIO     │                │
│                      │ Port: 9000   │                │
│                      │ Console:9001 │                │
│                      └──────┬───────┘                │
│                             │                         │
│                      ┌──────▼───────┐                │
│                      │    MLflow    │ (optional)     │
│                      │  Port: 5000  │                │
│                      └──────────────┘                │
└──────────────────────────────────────────────────────┘
```

### Service Descriptions

**roma-api:**
- FastAPI application server
- Handles execution management
- Exposes REST API
- Health check: `http://localhost:8000/health`

**postgres:**
- PostgreSQL 16 Alpine
- Stores execution metadata, checkpoints, traces
- Persistent volume: `postgres_data`
- Health check: `pg_isready`

**minio:**
- S3-compatible object storage
- Stores MLflow artifacts
- Persistent volume: `minio_data`
- UI: `http://localhost:9001`

**mlflow** (optional):
- Experiment tracking server
- Requires `--profile observability`
- UI: `http://localhost:5000`

---

## Environment Configuration

### Required Variables

```bash
# LLM Provider (at least one required)
OPENROUTER_API_KEY=your_key_here        # Recommended
# OR
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Database
POSTGRES_DB=roma_dspy                  # Database name
POSTGRES_USER=postgres                  # Database user
POSTGRES_PASSWORD=CHANGE_ME_IN_PROD    # Database password
POSTGRES_PORT=5432                      # Host port

# MinIO/S3
MINIO_ROOT_USER=minioadmin             # MinIO access key
MINIO_ROOT_PASSWORD=CHANGE_ME_IN_PROD  # MinIO secret key
MINIO_PORT=9000                         # S3 API port
MINIO_CONSOLE_PORT=9001                 # Console port

# API
API_PORT=8000                           # API port
POSTGRES_ENABLED=true                   # Enable PostgreSQL storage
```

### Optional Variables

```bash
# Toolkit API Keys
E2B_API_KEY=your_key_here              # Code execution
EXA_API_KEY=your_key_here              # Web search via MCP
SERPER_API_KEY=your_key_here           # Web search toolkit
GITHUB_PERSONAL_ACCESS_TOKEN=your_token # GitHub MCP server
COINGECKO_API_KEY=your_key_here        # CoinGecko Pro API

# MLflow (for observability profile)
MLFLOW_PORT=5000
MLFLOW_TRACKING_URI=http://mlflow:5000

# Storage
STORAGE_BASE_PATH=/opt/sentient         # Base path for file storage

# Security
ALLOWED_ORIGINS=https://yourdomain.com  # CORS origins (comma-separated)

# Logging
LOG_LEVEL=INFO                          # DEBUG, INFO, WARNING, ERROR
LOG_DIR=/app/logs                       # Log directory
```

---

## Docker Deployment

### Build and Start

**Build from scratch:**
```bash
# Clean build
just docker-build-clean

# Start services
just docker-up
```

**Start with existing images:**
```bash
# Basic (API + Postgres + MinIO)
just docker-up

# Full (includes MLflow)
just docker-up-full
```

### Verify Deployment

```bash
# Check all services running
just docker-ps

# Check health
curl http://localhost:8000/health

# View logs
just docker-logs

# View specific service logs
just docker-logs-service roma-api
just docker-logs-service postgres
just docker-logs-service mlflow
```

### Stop Services

```bash
# Stop (preserves data)
just docker-down

# Stop and remove volumes (data loss!)
just docker-down-clean
```

---

## Production Checklist

### Security

- [ ] Change default passwords in `.env`:
  - `POSTGRES_PASSWORD`
  - `MINIO_ROOT_PASSWORD`

- [ ] Set `ALLOWED_ORIGINS` for CORS (don't use `*` in production)

- [ ] Use HTTPS reverse proxy (nginx, Caddy, Traefik)

- [ ] Enable authentication on API (add middleware)

- [ ] Restrict network access (firewall rules)

- [ ] Use secrets management (Docker secrets, Vault, AWS Secrets Manager)

- [ ] Regularly update base images:
  ```bash
  docker-compose pull
  docker-compose up -d
  ```

### Reliability

- [ ] Configure automatic backups:
  ```bash
  # PostgreSQL backup
  docker exec roma-dspy-postgres pg_dump -U postgres roma_dspy > backup.sql
  ```

- [ ] Set resource limits in `docker-compose.yaml`:
  ```yaml
  roma-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
  ```

- [ ] Monitor disk usage:
  ```bash
  docker system df
  docker volume ls
  ```

- [ ] Configure log rotation:
  ```yaml
  roma-api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
  ```

### Observability

- [ ] Enable MLflow tracking:
  ```bash
  just docker-up-full
  ```

- [ ] Set up health check monitoring (Prometheus, Datadog, etc.)

- [ ] Configure log aggregation (ELK, Grafana Loki, Datadog)

- [ ] Monitor resource usage (CPU, memory, disk)

- [ ] Set up alerts for service failures

---

## Monitoring & Observability

### Health Checks

**API Health:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600.5,
  "active_executions": 2,
  "storage_connected": true,
  "cache_size": 5,
  "timestamp": "2024-10-21T12:00:00.000Z"
}
```

**PostgreSQL Health:**
```bash
docker exec roma-dspy-postgres pg_isready -U postgres
```

**MinIO Health:**
```bash
curl http://localhost:9000/minio/health/live
```

### MLflow UI

Access at http://localhost:5000

**Features:**
- Experiment tracking
- Run comparison
- Model registry
- Artifact storage

**View Executions:**
1. Navigate to http://localhost:5000
2. Filter by experiment name
3. Click execution ID to view details

### Metrics Endpoints

```bash
# Execution metrics
curl http://localhost:8000/api/v1/executions/<execution_id>/metrics | jq

# Cost summary
curl http://localhost:8000/api/v1/executions/<execution_id>/costs | jq

# Toolkit metrics
curl http://localhost:8000/api/v1/executions/<execution_id>/toolkit-metrics | jq

# LM traces
curl http://localhost:8000/api/v1/executions/<execution_id>/lm-traces | jq
```

### Log Aggregation

**View logs:**
```bash
# All services
just docker-logs

# Specific service
just docker-logs-service roma-api

# Follow logs
docker-compose logs -f roma-api
```

**Export logs:**
```bash
docker-compose logs roma-api > roma-api.log
```

---

## Scaling

### Horizontal Scaling (Multiple API Instances)

**docker-compose.yaml:**
```yaml
roma-api:
  # ... existing config ...
  deploy:
    replicas: 3  # Run 3 instances

  # Load balancer
  labels:
    - "traefik.enable=true"
    - "traefik.http.routers.roma.rule=Host(`api.yourdomain.com`)"
```

**With nginx load balancer:**
```nginx
upstream roma_api {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://roma_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Vertical Scaling (Resource Limits)

**docker-compose.yaml:**
```yaml
roma-api:
  deploy:
    resources:
      limits:
        cpus: '4.0'
        memory: 8G
      reservations:
        cpus: '2.0'
        memory: 4G

postgres:
  deploy:
    resources:
      limits:
        cpus: '2.0'
        memory: 4G
      reservations:
        cpus: '1.0'
        memory: 2G
```

### Database Scaling

**PostgreSQL optimization:**
```bash
# Connect to database
docker exec -it roma-dspy-postgres psql -U postgres -d roma_dspy

# Analyze tables
ANALYZE executions;
ANALYZE checkpoints;
ANALYZE lm_traces;

# Vacuum
VACUUM ANALYZE;

# Check indexes
\di
```

**Connection pooling** (add PgBouncer if needed):
```yaml
pgbouncer:
  image: pgbouncer/pgbouncer:latest
  environment:
    DATABASE_URL: postgres://postgres:password@postgres:5432/roma_dspy
    POOL_MODE: transaction
    MAX_CLIENT_CONN: 1000
    DEFAULT_POOL_SIZE: 20
```

---

## Security

### HTTPS/TLS

**Option 1: nginx reverse proxy**
```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Option 2: Caddy (auto HTTPS)**
```caddy
api.yourdomain.com {
    reverse_proxy localhost:8000
}
```

### Authentication

**Add API key middleware** (example):
```python
# src/roma_dspy/api/middleware.py
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != os.getenv("API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return await call_next(request)
```

**Use:**
```python
# src/roma_dspy/api/main.py
app.add_middleware(APIKeyMiddleware)
```

### Network Security

**Firewall rules:**
```bash
# Allow only specific IPs
sudo ufw allow from 203.0.113.0/24 to any port 8000

# Or use Docker network policies
```

**Internal network only:**
```yaml
# docker-compose.yaml
services:
  postgres:
    ports: []  # Don't expose to host
    networks:
      - roma-network

networks:
  roma-network:
    internal: true  # No external access
```

### Secrets Management

**Using Docker secrets:**
```yaml
secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  openrouter_api_key:
    file: ./secrets/openrouter_api_key.txt

services:
  roma-api:
    secrets:
      - postgres_password
      - openrouter_api_key
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
      OPENROUTER_API_KEY_FILE: /run/secrets/openrouter_api_key
```

---

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
just docker-logs-service roma-api
just docker-logs-service postgres
```

**Common issues:**

1. **Port already in use:**
   ```bash
   # Find process using port
   lsof -i :8000

   # Kill process or change API_PORT in .env
   ```

2. **Database connection failed:**
   ```bash
   # Check postgres health
   docker exec roma-dspy-postgres pg_isready -U postgres

   # Verify DATABASE_URL in .env
   ```

3. **Out of memory:**
   ```bash
   # Check Docker resources
   docker stats

   # Increase Docker memory limit
   # Docker Desktop → Settings → Resources → Memory
   ```

### Data Persistence Issues

**Check volumes:**
```bash
# List volumes
docker volume ls | grep roma

# Inspect volume
docker volume inspect roma-dspy_postgres_data

# Backup volume
docker run --rm -v roma-dspy_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

### Performance Issues

**Monitor resources:**
```bash
# Real-time stats
docker stats

# Check disk usage
docker system df

# Prune unused data
docker system prune -a
```

**Database slow queries:**
```bash
# Enable query logging
docker exec -it roma-dspy-postgres psql -U postgres -d roma_dspy

# Show slow queries
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

### MLflow Not Accessible

**Check service:**
```bash
# Ensure started with observability profile
just docker-up-full

# Check logs
docker-compose logs mlflow

# Verify port
curl http://localhost:5000
```

---

## Additional Resources

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Configuration**: [CONFIGURATION.md](CONFIGURATION.md)
- **API Reference**: http://localhost:8000/docs
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/

---

**Production Ready!** 🚀

For questions or issues, check logs first (`just docker-logs`), then consult the documentation or open an issue.
