# BIOIMAGIN - Deployment Guide

## Production Deployment Options

### Option 1: Local Server Deployment (Recommended)

#### System Requirements for Production
- **CPU**: 8+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 32GB+ (64GB for heavy workloads)
- **Storage**: 500GB+ SSD storage
- **GPU**: NVIDIA RTX 4090/A6000+ (optional but recommended)
- **OS**: Ubuntu 20.04 LTS or CentOS 8+

#### Production Setup

```bash
# 1. Create dedicated user
sudo useradd -m -s /bin/bash bioimagin
sudo usermod -aG sudo bioimagin

# 2. Switch to production user
sudo su - bioimagin

# 3. Clone repository
git clone https://github.com/your-org/bioimagin.git
cd bioimagin

# 4. Create production environment
python3 -m venv prod_env
source prod_env/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
pip install gunicorn supervisor

# 6. Install production packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install cellpose>=3.0.0
```

#### Production Configuration

**Create production config** (`config/production.py`):
```python
import os

class ProductionConfig:
    # Flask settings
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-super-secret-key'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    UPLOAD_FOLDER = '/var/bioimagin/uploads'
    RESULTS_FOLDER = '/var/bioimagin/results'
    
    # Performance settings
    WORKERS = 4
    WORKER_CONNECTIONS = 1000
    TIMEOUT = 300
    KEEPALIVE = 2
    
    # Security settings
    ALLOWED_HOSTS = ['your-domain.com', 'localhost']
    CORS_ORIGINS = ['https://your-domain.com']
    
    # Database (if needed for logging)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # Monitoring
    LOG_LEVEL = 'INFO'
    LOG_FILE = '/var/log/bioimagin/app.log'
```

#### Production Web Server Setup

**Gunicorn Configuration** (`gunicorn.conf.py`):
```python
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 2

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/var/log/bioimagin/access.log"
errorlog = "/var/log/bioimagin/error.log"
loglevel = "info"

# Process naming
proc_name = 'bioimagin'

# Security
user = "bioimagin"
group = "bioimagin"

# Performance
preload_app = True
worker_tmp_dir = "/dev/shm"
```

**Supervisor Configuration** (`/etc/supervisor/conf.d/bioimagin.conf`):
```ini
[program:bioimagin]
command=/home/bioimagin/bioimagin/prod_env/bin/gunicorn --config gunicorn.conf.py web_integration:app
directory=/home/bioimagin/bioimagin
user=bioimagin
group=bioimagin
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/bioimagin/supervisor.log
environment=PYTHONPATH="/home/bioimagin/bioimagin",FLASK_ENV="production"
```

#### Nginx Reverse Proxy

**Nginx Configuration** (`/etc/nginx/sites-available/bioimagin`):
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/bioimagin.crt;
    ssl_certificate_key /etc/ssl/private/bioimagin.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # File upload size
    client_max_body_size 500M;
    
    # Proxy to Gunicorn
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running analysis
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Static files
    location /static {
        alias /home/bioimagin/bioimagin/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Upload/Results with authentication
    location /uploads {
        alias /var/bioimagin/uploads;
        internal;  # Only accessible through application
    }
    
    location /results {
        alias /var/bioimagin/results;
        internal;
    }
}
```

#### Production Deployment Script

**Deploy Script** (`scripts/deploy.sh`):
```bash
#!/bin/bash
set -e

echo "🚀 Starting BIOIMAGIN Production Deployment"

# Configuration
APP_DIR="/home/bioimagin/bioimagin"
BACKUP_DIR="/var/backups/bioimagin"
LOG_DIR="/var/log/bioimagin"

# Create backup
echo "📦 Creating backup..."
sudo mkdir -p $BACKUP_DIR
sudo cp -r $APP_DIR $BACKUP_DIR/bioimagin_$(date +%Y%m%d_%H%M%S)

# Update application
echo "🔄 Updating application..."
cd $APP_DIR
git pull origin main

# Update dependencies
echo "📚 Updating dependencies..."
source prod_env/bin/activate
pip install -r requirements.txt --upgrade

# Database migrations (if applicable)
echo "🗄️ Running migrations..."
# python manage.py migrate

# Collect static files
echo "📁 Collecting static files..."
mkdir -p static/dist
# npm run build  # If using frontend build process

# Test configuration
echo "🧪 Testing configuration..."
python -c "from web_integration import app; print('✅ Configuration valid')"

# Restart services
echo "🔄 Restarting services..."
sudo supervisorctl restart bioimagin
sudo systemctl reload nginx

# Health check
echo "🏥 Performing health check..."
sleep 5
curl -f http://localhost:5000/api/health || {
    echo "❌ Health check failed"
    exit 1
}

echo "✅ Deployment completed successfully!"
```

### Option 2: Docker Deployment

#### Production Dockerfile

```dockerfile
# Multi-stage build for production
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash bioimagin

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set up application
WORKDIR /app
COPY --chown=bioimagin:bioimagin . .

# Create necessary directories
RUN mkdir -p uploads results models logs && \
    chown -R bioimagin:bioimagin /app

# Switch to non-root user
USER bioimagin

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Expose port
EXPOSE 5000

# Start application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "web_integration:app"]
```

#### Docker Compose for Production

**docker-compose.prod.yml**:
```yaml
version: '3.8'

services:
  bioimagin:
    build:
      context: .
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./uploads:/app/uploads
      - ./results:/app/results
      - ./models:/app/models
      - ./logs:/app/logs
    networks:
      - bioimagin_network
    depends_on:
      - redis
      - postgres
    
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
      - ./static:/static:ro
    networks:
      - bioimagin_network
    depends_on:
      - bioimagin
  
  redis:
    image: redis:alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - bioimagin_network
  
  postgres:
    image: postgres:14-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_DB=bioimagin
      - POSTGRES_USER=bioimagin
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - bioimagin_network

volumes:
  redis_data:
  postgres_data:

networks:
  bioimagin_network:
    driver: bridge
```

### Option 3: Cloud Deployment

#### AWS EC2 Deployment

**EC2 Instance Setup**:
```bash
# Launch EC2 instance (recommended: c5.2xlarge or larger)
# Ubuntu 20.04 LTS AMI
# Security Group: Allow HTTP (80), HTTPS (443), SSH (22)

# Connect and setup
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu

# Clone and deploy
git clone https://github.com/your-org/bioimagin.git
cd bioimagin
cp .env.example .env
# Edit .env with production values

# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

#### AWS ECS Deployment

**ECS Task Definition** (`ecs-task-def.json`):
```json
{
  "family": "bioimagin-prod",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "bioimagin",
      "image": "your-account.dkr.ecr.region.amazonaws.com/bioimagin:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "FLASK_ENV", "value": "production"},
        {"name": "AWS_DEFAULT_REGION", "value": "us-west-2"}
      ],
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:bioimagin-secrets"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/bioimagin",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:5000/api/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### Security Considerations

#### Application Security

**Environment Variables** (`.env.prod`):
```bash
# Flask Configuration
SECRET_KEY=your-super-secret-production-key
FLASK_ENV=production

# Database
DATABASE_URL=postgresql://user:pass@host:5432/bioimagin

# File Upload Security
MAX_UPLOAD_SIZE=500000000  # 500MB
ALLOWED_EXTENSIONS=png,jpg,jpeg,tiff,bmp

# CORS Security
CORS_ORIGINS=https://your-domain.com,https://app.your-domain.com

# Rate Limiting
RATE_LIMIT=100 per hour

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/bioimagin/app.log
```

#### Network Security

**Firewall Configuration**:
```bash
# UFW firewall setup
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable

# Fail2ban for SSH protection
sudo apt install fail2ban
sudo systemctl enable fail2ban
```

#### SSL/TLS Configuration

**Let's Encrypt SSL**:
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Monitoring and Logging

#### System Monitoring

**Prometheus Configuration** (`prometheus.yml`):
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'bioimagin'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

#### Application Monitoring

**Custom Metrics** (add to `web_integration.py`):
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('bioimagin_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('bioimagin_request_duration_seconds', 'Request latency')
ACTIVE_ANALYSES = Gauge('bioimagin_active_analyses', 'Number of active analyses')

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint).inc()
    REQUEST_LATENCY.observe(time.time() - request.start_time)
    return response
```

#### Log Management

**Structured Logging**:
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('/var/log/bioimagin/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('/var/log/bioimagin/app.log'))
```

### Performance Optimization

#### Caching Strategy

**Redis Caching**:
```python
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(timeout=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args)+str(kwargs))}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, timeout, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_result(timeout=1800)  # 30 minutes
def analyze_image_cached(image_path, **kwargs):
    return analyzer.analyze_image(image_path, **kwargs)
```

#### Database Optimization

**Database Setup** (if using database):
```sql
-- PostgreSQL optimizations
CREATE INDEX idx_analysis_created_at ON analyses(created_at);
CREATE INDEX idx_analysis_status ON analyses(status);
CREATE INDEX idx_analysis_user_id ON analyses(user_id);

-- Connection pooling
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
```

### Backup and Recovery

#### Automated Backups

**Backup Script** (`scripts/backup.sh`):
```bash
#!/bin/bash

BACKUP_DIR="/var/backups/bioimagin"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup application files
tar -czf $BACKUP_DIR/$DATE/app_backup.tar.gz \
    /home/bioimagin/bioimagin \
    --exclude=uploads \
    --exclude=results \
    --exclude=__pycache__ \
    --exclude=.git

# Backup models
cp -r /home/bioimagin/bioimagin/models $BACKUP_DIR/$DATE/

# Backup database (if using PostgreSQL)
pg_dump bioimagin > $BACKUP_DIR/$DATE/database.sql

# Backup configuration
cp -r /etc/nginx/sites-available $BACKUP_DIR/$DATE/nginx_config
cp -r /etc/supervisor/conf.d $BACKUP_DIR/$DATE/supervisor_config

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR/$DATE/ s3://your-backup-bucket/bioimagin/$DATE/

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -type d -mtime +30 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR/$DATE"
```

#### Recovery Procedures

**Recovery Script** (`scripts/recover.sh`):
```bash
#!/bin/bash

BACKUP_DATE=$1
BACKUP_DIR="/var/backups/bioimagin/$BACKUP_DATE"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "Starting recovery from backup: $BACKUP_DATE"

# Stop services
sudo supervisorctl stop bioimagin
sudo systemctl stop nginx

# Restore application
tar -xzf $BACKUP_DIR/app_backup.tar.gz -C /

# Restore models
cp -r $BACKUP_DIR/models/* /home/bioimagin/bioimagin/models/

# Restore database
psql bioimagin < $BACKUP_DIR/database.sql

# Restore configuration
cp -r $BACKUP_DIR/nginx_config/* /etc/nginx/sites-available/
cp -r $BACKUP_DIR/supervisor_config/* /etc/supervisor/conf.d/

# Start services
sudo systemctl start nginx
sudo supervisorctl start bioimagin

echo "Recovery completed successfully"
```

### Maintenance Procedures

#### Health Checks

**Automated Health Check** (`scripts/health_check.sh`):
```bash
#!/bin/bash

# Check application health
curl -f http://localhost:5000/api/health || {
    echo "Application health check failed"
    sudo supervisorctl restart bioimagin
}

# Check disk space
DISK_USAGE=$(df /var/bioimagin | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 85 ]; then
    echo "Disk usage critical: ${DISK_USAGE}%"
    # Cleanup old results
    find /var/bioimagin/results -type f -mtime +7 -delete
fi

# Check memory usage
MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
if [ $MEMORY_USAGE -gt 90 ]; then
    echo "Memory usage critical: ${MEMORY_USAGE}%"
    sudo supervisorctl restart bioimagin
fi

# Check log file sizes
find /var/log/bioimagin -name "*.log" -size +100M -exec logrotate -f /etc/logrotate.d/bioimagin {} \;
```

#### Update Procedures

**Update Script** (`scripts/update.sh`):
```bash
#!/bin/bash

echo "Starting BIOIMAGIN update..."

# Backup current version
./scripts/backup.sh

# Update code
cd /home/bioimagin/bioimagin
git fetch origin
git checkout main
git pull origin main

# Update dependencies
source prod_env/bin/activate
pip install -r requirements.txt --upgrade

# Run tests
python -m pytest tests/basic_tests.py

# Rolling restart
sudo supervisorctl restart bioimagin

# Verify deployment
sleep 10
curl -f http://localhost:5000/api/health || {
    echo "Update failed, rolling back..."
    git checkout HEAD~1
    sudo supervisorctl restart bioimagin
    exit 1
}

echo "Update completed successfully"
```

---

**BIOIMAGIN is now ready for production deployment! 🚀🔬**

For additional support:
- Monitor system health with provided scripts
- Review logs regularly for issues
- Keep backups current and tested
- Update security patches promptly