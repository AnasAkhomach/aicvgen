# AI CV Generator - Deployment Guide

This guide provides comprehensive instructions for deploying the AI CV Generator application in different environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Configuration](#environment-configuration)
- [Deployment Options](#deployment-options)
- [Production Deployment](#production-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Memory**: Minimum 2GB RAM, recommended 4GB+
- **Storage**: Minimum 5GB free space
- **Network**: Internet connection for API calls

### API Requirements

- **Gemini API Key**: Required for LLM functionality
  - Get your API key from: https://aistudio.google.com/app/apikey
  - Optional: Secondary API key for fallback

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd aicvgen

# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

### 2. Deploy with Script

```bash
# Make deployment script executable (Linux/macOS)
chmod +x scripts/deploy.sh

# Deploy in development mode
./scripts/deploy.sh deploy

# Or deploy in production mode
./scripts/deploy.sh -e production -f production deploy
```

### 3. Access Application

- **Development**: http://localhost:8501
- **Production**: http://localhost (with nginx proxy)

## Environment Configuration

### Required Environment Variables

```bash
# Primary API key (required)
GEMINI_API_KEY=your_api_key_here

# Environment setting
ENVIRONMENT=development  # or staging, production
```

### Optional Configuration

See `.env.example` for comprehensive configuration options including:

- Performance tuning
- Caching settings
- Security options
- Monitoring configuration
- File handling limits

## Deployment Options

### Development Deployment

```bash
# Basic development setup
./scripts/deploy.sh -e development deploy

# With verbose logging
./scripts/deploy.sh -e development -v deploy

# Detached mode
./scripts/deploy.sh -e development -d deploy
```

**Features:**
- Debug mode enabled
- Hot reloading
- Detailed logging
- Single container

### Staging Deployment

```bash
# Staging environment
./scripts/deploy.sh -e staging deploy
```

**Features:**
- Production-like configuration
- Reduced logging
- Performance monitoring
- Health checks

### Production Deployment

```bash
# Full production stack
./scripts/deploy.sh -e production -f production deploy
```

**Features:**
- Nginx reverse proxy
- SSL/TLS support
- Performance optimization
- Security hardening
- Health monitoring

## Production Deployment

### 1. Pre-deployment Checklist

- [ ] Valid SSL certificates (if using HTTPS)
- [ ] Production API keys configured
- [ ] Firewall rules configured
- [ ] Backup strategy in place
- [ ] Monitoring setup verified
- [ ] Resource limits configured

### 2. Production Configuration

```bash
# Update .env for production
ENVIRONMENT=production
ENABLE_DEBUG_MODE=false
LOG_LEVEL=WARNING
SESSION_SECRET_KEY=your_secure_random_key_here
```

### 3. SSL/TLS Setup

```bash
# Create SSL directory
mkdir -p nginx/ssl

# Copy your certificates
cp your-cert.pem nginx/ssl/
cp your-key.pem nginx/ssl/

# Update nginx configuration
# Edit nginx/nginx.conf with your domain and certificate paths
```

### 4. Deploy Production Stack

```bash
# Deploy with production profile
./scripts/deploy.sh -e production -f production -d deploy

# Verify deployment
./scripts/deploy.sh health
```

## Monitoring and Observability

### Enable Monitoring Stack

```bash
# Deploy with monitoring
./scripts/deploy.sh -f monitoring deploy
```

**Includes:**
- **Prometheus**: Metrics collection (http://localhost:9090)
- **Grafana**: Metrics visualization (http://localhost:3000)
- **Application metrics**: Performance and usage data

### Health Monitoring

```bash
# Check application health
./scripts/deploy.sh health

# View application status
./scripts/deploy.sh status

# View logs
./scripts/deploy.sh logs
```

### Key Metrics to Monitor

- **Application Health**: Response time, error rates
- **Resource Usage**: CPU, memory, disk space
- **API Usage**: Request count, token consumption
- **Cache Performance**: Hit rates, memory usage

## Backup and Recovery

### Automated Backup

```bash
# Create backup
./scripts/deploy.sh backup

# Backups are stored in: backups/YYYYMMDD_HHMMSS/
```

### Restore from Backup

```bash
# List and restore backups
./scripts/deploy.sh restore
```

### What Gets Backed Up

- Application data (`data/` directory)
- Configuration files (`.env`)
- Application logs
- User sessions and cache

### Backup Strategy Recommendations

- **Frequency**: Daily automated backups
- **Retention**: Keep 30 days of backups
- **Storage**: Store backups off-site
- **Testing**: Regularly test restore procedures

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check container logs
docker-compose logs aicvgen

# Check system resources
docker system df
free -h
```

#### 2. API Key Issues

```bash
# Verify API key in environment
docker-compose exec aicvgen env | grep GEMINI

# Test API connectivity
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
     "https://generativelanguage.googleapis.com/v1/models"
```

#### 3. Performance Issues

```bash
# Check resource usage
docker stats

# Check application metrics
curl http://localhost:8501/_stcore/health

# Review performance logs
./scripts/deploy.sh logs | grep -i performance
```

#### 4. Network Issues

```bash
# Check port availability
netstat -tlnp | grep 8501

# Test internal connectivity
docker-compose exec aicvgen curl localhost:8501
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
./scripts/deploy.sh restart

# View detailed logs
./scripts/deploy.sh logs
```

### Getting Help

1. Check application logs: `./scripts/deploy.sh logs`
2. Verify configuration: `./scripts/deploy.sh status`
3. Run health check: `./scripts/deploy.sh health`
4. Review documentation: `docs/`

## Security Considerations

### Production Security Checklist

- [ ] **API Keys**: Secure storage, rotation policy
- [ ] **Network**: Firewall rules, VPN access
- [ ] **SSL/TLS**: Valid certificates, HTTPS only
- [ ] **Container**: Non-root user, minimal image
- [ ] **Secrets**: Environment variables, not in code
- [ ] **Updates**: Regular security updates
- [ ] **Monitoring**: Security event logging
- [ ] **Backup**: Encrypted backup storage

### Security Best Practices

1. **Use HTTPS in production**
2. **Rotate API keys regularly**
3. **Limit network access**
4. **Monitor for suspicious activity**
5. **Keep dependencies updated**
6. **Use strong session secrets**
7. **Implement rate limiting**
8. **Regular security audits**

### Network Security

```bash
# Restrict access to specific IPs
# Add to docker-compose.yml:
ports:
  - "127.0.0.1:8501:8501"  # Local access only

# Use nginx for SSL termination and access control
# Configure in nginx/nginx.conf
```

## Advanced Deployment

### Kubernetes Deployment

For Kubernetes deployment, see `docs/architecture.md` for:
- Kubernetes manifests
- Helm charts
- Scaling strategies
- Service mesh integration

### CI/CD Integration

```bash
# Example CI/CD pipeline
# 1. Build and test
docker build -t aicvgen:latest .

# 2. Run tests
docker run --rm aicvgen:latest pytest

# 3. Deploy
./scripts/deploy.sh -e production deploy
```

### Load Balancing

For high-availability deployments:
- Use multiple application instances
- Configure nginx load balancing
- Implement health checks
- Set up session affinity

## Maintenance

### Regular Maintenance Tasks

```bash
# Update application
git pull
./scripts/deploy.sh build
./scripts/deploy.sh restart

# Clean up resources
./scripts/deploy.sh cleanup

# Update dependencies
docker-compose pull
```

### Performance Optimization

1. **Monitor resource usage**
2. **Optimize cache settings**
3. **Tune rate limits**
4. **Scale horizontally if needed**
5. **Regular performance testing**

---

## Support

For additional support:
- Review `docs/developer_guide.md`
- Check `docs/architecture.md`
- See troubleshooting section above
- Contact system administrator