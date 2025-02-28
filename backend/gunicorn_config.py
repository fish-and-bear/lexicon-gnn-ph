"""
Gunicorn configuration for the Filipino Dictionary API.
"""

import multiprocessing
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '10000')}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2

# Process naming
proc_name = 'fil-dict-api'
pythonpath = 'backend'

# Logging
accesslog = 'logs/access.log'
errorlog = 'logs/error.log'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# SSL (uncomment if using HTTPS)
# keyfile = 'certs/privkey.pem'
# certfile = 'certs/fullchain.pem'

# Server mechanics
daemon = False
pidfile = 'gunicorn.pid'
user = None
group = None
umask = 0
tmp_upload_dir = None

# Limits
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Debug
reload = os.getenv('FLASK_ENV', 'production') == 'development'
reload_engine = 'auto'
spew = False

# Server hooks
def on_starting(server):
    """Log server startup."""
    server.log.info("Starting Gunicorn server...")

def on_reload(server):
    """Log server reload."""
    server.log.info("Reloading Gunicorn server...")

def post_fork(server, worker):
    """Setup after worker fork."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def pre_fork(server, worker):
    """Setup before worker fork."""
    pass

def pre_exec(server):
    """Setup before exec."""
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    """Log when server is ready."""
    server.log.info("Server is ready. Spawning workers...") 