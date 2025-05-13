"""
Filipino Lexical Knowledge Graph Enhancement using Heterogeneous Graph Neural Networks

This module implements a framework for automatically enhancing lexical knowledge graphs
for low-resource Philippine languages using advanced graph neural network techniques.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys

# Configure logging
def setup_logging(log_dir="logs"):
    """Configure logging for the ML module."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = os.path.join(log_dir, 'ml.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    return root_logger

logger = setup_logging()
logger.info("ML module initialization")

# Module version
__version__ = "0.1.0" 