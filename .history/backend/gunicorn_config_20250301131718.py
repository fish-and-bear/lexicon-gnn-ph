"""
Gunicorn configuration for the Filipino Dictionary API.
This configuration is used in production environments, typically on Linux.
For Windows development, Waitress is used instead (see serve.py).
"""

import multiprocessing
import os
import platform
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine base directory
BASE_DIR = Path(__file__).resolve().parent

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '10000')}"
backlog = int(os.getenv('GUNICORN_BACKLOG', 2048))

# Worker processes - scale based on CPU cores
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv('GUNICORN_WORKER_CLASS', 'sync')
worker_connections = int(os.getenv('GUNICORN_WORKER_CONNECTIONS', 1000))
timeout = int(os.getenv('GUNICORN_TIMEOUT', 30))
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', 2))

# Process naming
proc_name = 'fil-dict-api'
pythonpath = str(BASE_DIR)

# Ensure log directory exists
log_dir = BASE_DIR.parent / 'logs'
log_dir.mkdir(exist_ok=True)

# Logging
accesslog = str(log_dir / 'gunicorn_access.log')
errorlog = str(log_dir / 'gunicorn_error.log')
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# SSL (uncomment and configure if using HTTPS)
if os.getenv('ENABLE_SSL', 'false').lower() == 'true':
    cert_dir = BASE_DIR.parent / 'certs'
    keyfile = str(cert_dir / 'privkey.pem')
    certfile = str(cert_dir / 'fullchain.pem')

# Server mechanics
daemon = os.getenv('GUNICORN_DAEMON', 'false').lower() == 'true'
pidfile = str(BASE_DIR.parent / 'gunicorn.pid')
user = os.getenv('GUNICORN_USER', None)
group = os.getenv('GUNICORN_GROUP', None)
umask = int(os.getenv('GUNICORN_UMASK', 0))
tmp_upload_dir = None

# Limits
limit_request_line = int(os.getenv('GUNICORN_LIMIT_REQUEST_LINE', 4096))
limit_request_fields = int(os.getenv('GUNICORN_LIMIT_REQUEST_FIELDS', 100))
limit_request_field_size = int(os.getenv('GUNICORN_LIMIT_REQUEST_FIELD_SIZE', 8190))

# Debug
reload = os.getenv('FLASK_ENV', 'production') == 'development'
reload_engine = 'auto'
spew = False

# Server hooks
def on_starting(server):
    """Log server startup."""
    server.log.info("Starting Gunicorn server...")
    server.log.info(f"Workers: {workers}, Bind: {bind}")
    server.log.info(f"Environment: {os.getenv('FLASK_ENV', 'production')}")

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
    
    # Print a helpful message about how to use supervisor
    if not daemon:
        print("\nGunicorn server is running.")
        print("For production deployment, consider using supervisor:")
        print("  supervisord -c supervisor.conf")
        print("  supervisorctl -c supervisor.conf status") 