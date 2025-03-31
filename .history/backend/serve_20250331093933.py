"""
Server entry point for the Filipino Dictionary API with enhanced monitoring.
"""

import os
import logging
from app import create_app
from prometheus_client import start_http_server, Counter, Gauge
import structlog
from database import check_db_health
import threading
import time
from typing import Dict, Any

# Configure logging
logger = structlog.get_logger(__name__)

# Metrics
SERVER_UPTIME = Gauge('server_uptime_seconds', 'Server uptime in seconds')
HEALTH_CHECK_STATUS = Gauge('health_check_status', 'Health check status (1=healthy, 0=unhealthy)')
DB_METRICS = Gauge('database_metrics', 'Database metrics', ['metric'])

def health_check_worker():
    """Background worker to monitor database health."""
    start_time = time.time()
    
    while True:
        try:
            # Update uptime
            SERVER_UPTIME.set(time.time() - start_time)
            
            # Check database health
            health_status = check_db_health()
            HEALTH_CHECK_STATUS.set(1 if health_status["status"] == "healthy" else 0)
            
            # Update database metrics
            if "statistics" in health_status:
                stats = health_status["statistics"]
                for metric, value in stats.items():
                    DB_METRICS.labels(metric=metric).set(value)
                    
            if health_status["status"] != "healthy":
                logger.warning("Database health check failed", status=health_status)
        except Exception as e:
            logger.error("Health check error", error=str(e))
            HEALTH_CHECK_STATUS.set(0)
        time.sleep(60)  # Check every minute

def main():
    """Main entry point for the server."""
    try:
        # Start Prometheus metrics server
        metrics_port = int(os.getenv('METRICS_PORT', 9090))
        start_http_server(metrics_port)
        logger.info("Metrics server started", port=metrics_port)
        
        # Create the Flask application
        app = create_app()
        
        # Start health check worker
        health_thread = threading.Thread(target=health_check_worker, daemon=True)
        health_thread.start()
        logger.info("Health check worker started")
        
        # Get port from environment variable or use default
        port = int(os.getenv('PORT', 10000))
        
        # Run the application
        logger.info("Starting server", port=port)
        app.run(
            host='0.0.0.0',
            port=port,
            threaded=True,
            use_reloader=False  # Disable reloader in production
        )
    except Exception as e:
        logger.error("Server startup failed", error=str(e))
        raise

if __name__ == '__main__':
    main()
