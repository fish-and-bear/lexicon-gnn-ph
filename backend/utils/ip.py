"""IP utilities for Flask applications."""

from flask import request

def get_remote_address():
    """
    Get the remote address (IP) of the client.
    
    This function checks various HTTP headers for proxied requests
    and falls back to the direct remote address.
    
    Returns:
        str: The remote IP address
    """
    # Check for the X-Forwarded-For header first (common for proxies)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # X-Forwarded-For can be a comma-separated list; the client's IP is the first one
        return forwarded_for.split(',')[0].strip()
    
    # Check for other common proxy headers
    if request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    
    # Fall back to the direct remote address
    return request.remote_addr 