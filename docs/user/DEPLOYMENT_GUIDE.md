# Deployment Guide

## Overview

This guide covers deploying the AI CV Generator MVP in various environments, from local development to production cloud deployments.

## Prerequisites

### System Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Stable internet connection for LLM API calls

### Required Services

- **Google Gemini API**: Active API key with sufficient quota
- **Python 3.11+**: Runtime environment
- **Git**: For source code management

## Local Development Deployment

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd aicvgen

# Setup environment
python -m venv .vs_venv
source .vs_venv/bin/activate  # Linux/macOS
# or
.vs_venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run application
python run_app.py
```

### Development Configuration

```bash
# .env for development
GOOGLE_API_KEY=your_development_api_key
LOG_LEVEL=DEBUG
ENABLE_DEBUG_MODE=true
SESSION_TIMEOUT=7200
MAX_CONCURRENT_REQUESTS=5
```

## Docker Deployment

### Building the Image

```bash
# Build the Docker image
docker build -t aicvgen:latest .

# Verify the build
docker images | grep aicvgen
```

### Running with Docker

```bash
# Run with environment file
docker run -d \
  --name aicvgen-app \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  aicvgen:latest

# Check container status
docker ps
docker logs aicvgen-app
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  aicvgen:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - LOG_LEVEL=INFO
      - ENABLE_DEBUG_MODE=false
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - aicvgen
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS

1. **Create ECR Repository**:

```bash
# Create repository
aws ecr create-repository --repository-name aicvgen

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push image
docker tag aicvgen:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/aicvgen:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/aicvgen:latest
```

2. **Create ECS Task Definition**:

```json
{
  "family": "aicvgen-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "aicvgen",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/aicvgen:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "secrets": [
        {
          "name": "GOOGLE_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:aicvgen/api-keys"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/aicvgen",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

3. **Create ECS Service**:

```bash
aws ecs create-service \
  --cluster aicvgen-cluster \
  --service-name aicvgen-service \
  --task-definition aicvgen-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-abcdef],assignPublicIp=ENABLED}"
```

#### Using AWS App Runner

```yaml
# apprunner.yaml
version: 1.0
runtime: python3
build:
  commands:
    build:
      - pip install -r requirements.txt
run:
  runtime-version: 3.11
  command: python run_app.py
  network:
    port: 8501
    env: PORT
  env:
    - name: LOG_LEVEL
      value: INFO
```

### Google Cloud Platform

#### Using Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/aicvgen

# Deploy to Cloud Run
gcloud run deploy aicvgen \
  --image gcr.io/PROJECT_ID/aicvgen \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars LOG_LEVEL=INFO \
  --set-secrets GOOGLE_API_KEY=aicvgen-api-key:latest
```

#### Using GKE

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aicvgen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aicvgen
  template:
    metadata:
      labels:
        app: aicvgen
    spec:
      containers:
      - name: aicvgen
        image: gcr.io/PROJECT_ID/aicvgen:latest
        ports:
        - containerPort: 8501
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: aicvgen-secrets
              key: google-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: aicvgen-service
spec:
  selector:
    app: aicvgen
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

Deploy with:

```bash
kubectl apply -f kubernetes/
```

### Azure Deployment

#### Using Container Instances

```bash
# Create resource group
az group create --name aicvgen-rg --location eastus

# Create container instance
az container create \
  --resource-group aicvgen-rg \
  --name aicvgen-app \
  --image aicvgen:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8501 \
  --environment-variables LOG_LEVEL=INFO \
  --secure-environment-variables GOOGLE_API_KEY=$GOOGLE_API_KEY
```

## Production Configuration

### Environment Variables

```bash
# Production .env
GOOGLE_API_KEY=production_api_key
LOG_LEVEL=WARNING
ENABLE_DEBUG_MODE=false
SESSION_TIMEOUT=3600
MAX_CONCURRENT_REQUESTS=20
ENABLE_METRICS=true
METRICS_PORT=9090
```

### Nginx Configuration

```nginx
# nginx.conf
upstream aicvgen {
    server aicvgen:8501;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://aicvgen;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /healthz {
        proxy_pass http://aicvgen/healthz;
    }
}
```

### SSL/TLS Setup

```bash
# Using Let's Encrypt with Certbot
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Monitoring and Logging

### Health Checks

Add to your application:

```python
# src/api/health.py
from fastapi import APIRouter
from src.core.enhanced_orchestrator import EnhancedOrchestrator

router = APIRouter()

@router.get("/healthz")
async def health_check():
    try:
        # Basic health check
        orchestrator = EnhancedOrchestrator()
        return {"status": "healthy", "timestamp": datetime.utcnow()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@router.get("/readiness")
async def readiness_check():
    # Check external dependencies
    try:
        # Test LLM service
        llm_service = LLMService()
        await llm_service.health_check()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}
```

### Prometheus Metrics

```python
# src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('aicvgen_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('aicvgen_request_duration_seconds', 'Request duration')
ACTIVE_SESSIONS = Gauge('aicvgen_active_sessions', 'Active sessions')
```

### Log Aggregation

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  aicvgen:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  fluentd:
    image: fluent/fluentd:v1.14
    volumes:
      - ./fluentd.conf:/fluentd/etc/fluent.conf
      - /var/log:/var/log
    ports:
      - "24224:24224"
```

## Security Considerations

### API Key Management

- Use environment variables or secret management services
- Rotate keys regularly
- Monitor API usage and set up alerts

### Network Security

```bash
# Firewall rules (UFW example)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 8501/tcp   # Block direct access to Streamlit
sudo ufw enable
```

### Container Security

```dockerfile
# Security-hardened Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r aicvgen && useradd -r -g aicvgen aicvgen

# Install security updates
RUN apt-get update && apt-get upgrade -y && apt-get clean

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership and permissions
RUN chown -R aicvgen:aicvgen /app
USER aicvgen

# Run application
EXPOSE 8501
CMD ["python", "run_app.py"]
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/aicvgen_$DATE"

mkdir -p $BACKUP_DIR

# Backup session data
cp -r /app/data/sessions $BACKUP_DIR/

# Backup logs
cp -r /app/logs $BACKUP_DIR/

# Backup configuration
cp /app/.env $BACKUP_DIR/

# Compress backup
tar -czf "$BACKUP_DIR.tar.gz" $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

### Disaster Recovery

1. **Database Recovery**: Restore session data from backups
2. **Configuration Recovery**: Restore environment variables and configs
3. **Application Recovery**: Redeploy from version control
4. **DNS Failover**: Update DNS to point to backup infrastructure

## Performance Optimization

### Scaling Strategies

1. **Horizontal Scaling**: Multiple container instances behind load balancer
2. **Vertical Scaling**: Increase CPU/memory resources
3. **Caching**: Implement Redis for session and response caching
4. **CDN**: Use CloudFront/CloudFlare for static assets

### Resource Limits

```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   ```bash
   # Check API key configuration
   docker exec aicvgen-app env | grep GOOGLE_API_KEY
   ```

2. **Memory Issues**:
   ```bash
   # Monitor memory usage
   docker stats aicvgen-app
   ```

3. **Network Issues**:
   ```bash
   # Test connectivity
   docker exec aicvgen-app curl -I https://generativelanguage.googleapis.com
   ```

### Log Analysis

```bash
# View application logs
docker logs aicvgen-app --tail 100 -f

# Search for errors
docker logs aicvgen-app 2>&1 | grep ERROR

# Monitor performance
docker logs aicvgen-app 2>&1 | grep "processing_time"
```

For additional support, check the application logs and refer to the API documentation.