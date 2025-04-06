from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Define limiter instance globally but initialize it within the app factory
limiter = Limiter(
    key_func=get_remote_address, 
    strategy="fixed-window" 
    # Default limits and storage_uri will be set via init_app
) 