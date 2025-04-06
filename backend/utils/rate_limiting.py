"""Rate limiting utilities for Flask routes."""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize the rate limiter
# This will be configured in the application factory
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
) 