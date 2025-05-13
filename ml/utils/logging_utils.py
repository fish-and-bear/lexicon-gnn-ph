"""
Logging utilities for the ML training pipeline.

This module provides functions for setting up logging configurations
and tracking training progress.
"""

import os
import logging
import sys
from typing import Optional
import json
from pathlib import Path
from datetime import datetime

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to the log file
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    # Suppress excessive logging from libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {level}")
    if log_file:
        logger.info(f"Logs will be saved to {log_file}")

class TrainingLogger:
    """
    Logger for tracking and saving training progress.
    """
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize the training logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Optional name for the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set experiment name or generate based on timestamp
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        
        # Create experiment directory
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics = {
            'train': [],
            'valid': [],
            'test': {}
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.experiment_name}")
        self.logger.info(f"Initialized training logger for experiment: {self.experiment_name}")
        self.logger.info(f"Logs will be saved to {self.experiment_dir}")
        
        # Set up file handlers for this experiment
        log_file = self.experiment_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
    
    def log_config(self, config: dict):
        """
        Save configuration to a file.
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Saved configuration to {config_file}")
    
    def log_model_summary(self, model_summary: str):
        """
        Save model summary to a file.
        
        Args:
            model_summary: String representation of model summary
        """
        summary_file = self.experiment_dir / "model_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(model_summary)
        self.logger.info(f"Saved model summary to {summary_file}")
    
    def log_epoch(self, epoch: int, train_metrics: dict, valid_metrics: dict = None):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            valid_metrics: Validation metrics (optional)
        """
        # Add epoch number to metrics
        train_metrics['epoch'] = epoch
        
        # Store metrics
        self.metrics['train'].append(train_metrics)
        
        # Log training metrics
        metric_str = f"Epoch {epoch} - Train: "
        metric_str += ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() 
                               if k != 'epoch' and isinstance(v, (int, float))])
        self.logger.info(metric_str)
        
        # Handle validation metrics if provided
        if valid_metrics:
            valid_metrics['epoch'] = epoch
            self.metrics['valid'].append(valid_metrics)
            
            valid_str = f"Epoch {epoch} - Valid: "
            valid_str += ", ".join([f"{k}: {v:.4f}" for k, v in valid_metrics.items() 
                                  if k != 'epoch' and isinstance(v, (int, float))])
            self.logger.info(valid_str)
        
        # Save updated metrics
        self._save_metrics()
    
    def log_test_results(self, test_metrics: dict):
        """
        Log final test results.
        
        Args:
            test_metrics: Test metrics dictionary
        """
        self.metrics['test'] = test_metrics
        
        # Log test metrics
        test_str = "Test Results: "
        test_str += ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items() 
                             if isinstance(v, (int, float))])
        self.logger.info(test_str)
        
        # Save updated metrics
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to a JSON file."""
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
    def log_prediction_examples(self, examples: list):
        """
        Log prediction examples.
        
        Args:
            examples: List of prediction examples (dict with input, prediction, ground_truth)
        """
        examples_file = self.experiment_dir / "prediction_examples.json"
        with open(examples_file, 'w') as f:
            json.dump(examples, f, indent=2)
        self.logger.info(f"Saved {len(examples)} prediction examples to {examples_file}")
    
    def log_artifact(self, name: str, data, is_json: bool = True):
        """
        Save a custom artifact.
        
        Args:
            name: Artifact name
            data: Data to save
            is_json: Whether the data should be saved as JSON
        """
        artifact_file = self.experiment_dir / f"{name}"
        
        if is_json:
            with open(artifact_file, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(artifact_file, 'w') as f:
                f.write(str(data))
                
        self.logger.info(f"Saved artifact {name} to {artifact_file}")
        
    def log_error(self, error_message: str, exception: Exception = None):
        """
        Log an error.
        
        Args:
            error_message: Error message
            exception: Exception object (optional)
        """
        if exception:
            self.logger.error(f"{error_message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(error_message)
            
        # Store error in a separate file
        error_file = self.experiment_dir / "errors.txt"
        with open(error_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {error_message}")
            if exception:
                f.write(f": {str(exception)}\n")
            else:
                f.write("\n")


def get_tb_writer(log_dir: str, experiment_name: Optional[str] = None):
    """
    Get TensorBoard writer if TensorBoard is available.
    
    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Optional experiment name
        
    Returns:
        TensorBoard SummaryWriter or None if TensorBoard is not available
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        
        # Set experiment name or generate based on timestamp
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
            
        tb_log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir=tb_log_dir)
        logging.getLogger(__name__).info(f"TensorBoard logging enabled at {tb_log_dir}")
        return writer
    except ImportError:
        logging.getLogger(__name__).warning("TensorBoard not available. Skipping TensorBoard logging.")
        return None 