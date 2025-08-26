#!/bin/bash

# Production deployment script with Gunicorn
# Use this for better production performance

echo "Installing production dependencies..."
pip install gunicorn

echo "Creating Gunicorn configuration..."
cat > gunicorn.conf.py << 'EOF'
bind = "0.0.0.0:5000"
workers = 2
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 120
keepalive = 5
preload_app = True
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
EOF

echo "Starting application with Gunicorn..."
gunicorn -c gunicorn.conf.py app:app