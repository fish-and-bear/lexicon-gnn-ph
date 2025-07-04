"""
Unified configuration system for FilRelex ML components.
Provides simple, robust configuration management.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "fil-dict-db-jessegarfieldscats-becf.h.aivencloud.com"
    port: int = 18251
    database: str = "defaultdb"
    user: str = "public_user"
    password: str = "AVNS_kWlkz-O3MvuC1PQEu3I"
    ssl_mode: str = "require"
    connection_timeout: int = 30
    pool_size: int = 5
    max_overflow: int = 10
    
    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"
        )

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = 256
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "relu"
    use_residual: bool = True
    use_layer_norm: bool = True

@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    early_stopping: bool = True
    patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    save_best_model: bool = True

@dataclass
class DataConfig:
    """Data processing configuration."""
    max_nodes: int = 10000  # Limit for testing/development
    min_degree: int = 1
    include_relations: list = None
    feature_types: list = None
    normalize_features: bool = True
    
    def __post_init__(self):
        if self.include_relations is None:
            self.include_relations = ["synonym", "antonym", "hypernym", "hyponym"]
        if self.feature_types is None:
            self.feature_types = ["lemma", "pos", "definition"]

@dataclass
class MLConfig:
    """Main configuration container."""
    database: DatabaseConfig
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    device: str = "auto"  # auto, cpu, cuda
    output_dir: str = "outputs"
    log_level: str = "INFO"
    random_seed: int = 42
    
    def __post_init__(self):
        """Set device automatically if 'auto'."""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-selected device: {self.device}")
    
    @classmethod
    def from_file(cls, config_path: Optional[str] = None) -> 'MLConfig':
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                # Create configs from nested dictionaries
                database_config = DatabaseConfig(**data.get('database', {}))
                model_config = ModelConfig(**data.get('model', {}))
                training_config = TrainingConfig(**data.get('training', {}))
                data_config = DataConfig(**data.get('data', {}))
                
                # Create main config
                main_config_data = {k: v for k, v in data.items() 
                                  if k not in ['database', 'model', 'training', 'data']}
                
                config = cls(
                    database=database_config,
                    model=model_config,
                    training=training_config,
                    data=data_config,
                    **main_config_data
                )
                
                logger.info(f"Loaded configuration from {config_path}")
                return config
                
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        # Return default configuration
        return cls(
            database=DatabaseConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig()
        )
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        config_dict = asdict(self)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved configuration to {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

def load_config(config_path: Optional[str] = None) -> MLConfig:
    """Convenience function to load configuration."""
    return MLConfig.from_file(config_path)

def get_default_config_path() -> str:
    """Get the default configuration file path."""
    return os.path.join(os.path.dirname(__file__), "default_ml_config.json")

def create_default_config_file():
    """Create a default configuration file."""
    config = MLConfig.from_file()
    default_path = get_default_config_path()
    config.save(default_path)
    logger.info(f"Created default configuration at {default_path}")

if __name__ == "__main__":
    # Create default config file when run as script
    create_default_config_file() 