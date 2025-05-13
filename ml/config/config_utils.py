"""
Utility functions for managing configuration settings.

This module provides functions for loading, saving, and merging config files.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'default_config.json')

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file. If None, loads default config.
        
    Returns:
        Dictionary with configuration settings
    """
    # Load default configuration
    default_config = {}
    try:
        with open(DEFAULT_CONFIG_PATH, 'r') as f:
            default_config = json.load(f)
            logger.info(f"Loaded default configuration from {DEFAULT_CONFIG_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load default configuration: {e}")
    
    # If no custom config path provided, return default config
    if config_path is None:
        return default_config
    
    # Load custom configuration
    custom_config = {}
    try:
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            logger.info(f"Loaded custom configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return default_config
    
    # Merge configurations
    merged_config = merge_configs(default_config, custom_config)
    
    return merged_config

def save_config(config: Dict[str, Any], output_path: str) -> bool:
    """
    Save configuration to a file.
    
    Args:
        config: Dictionary with configuration settings
        output_path: Path to save the configuration file
        
    Returns:
        Success status
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to file with pretty formatting
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration to {output_path}: {e}")
        return False

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary to override base values
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        # If the value is a nested dictionary and the key exists in base_config
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(base_config[key], value)
        else:
            # Otherwise just override the value
            merged[key] = value
    
    return merged

def get_experiment_config(
    config_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get configuration for an experiment with optional overrides.
    
    Args:
        config_path: Path to the configuration file
        experiment_name: Name of the experiment
        overrides: Dictionary with configuration overrides
        
    Returns:
        Dictionary with experiment configuration settings
    """
    # Load base configuration
    config = load_config(config_path)
    
    # Add experiment name if provided
    if experiment_name is not None:
        if 'experiment' not in config:
            config['experiment'] = {}
        config['experiment']['name'] = experiment_name
    
    # Apply overrides if provided
    if overrides is not None:
        config = merge_configs(config, overrides)
    
    return config 