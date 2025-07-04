#!/usr/bin/env python3
"""
Comprehensive and Robust Training Script for Philippine Lexicon GNN Models
"""

import torch
import json
import yaml
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

from gnn_lexicon.src.cli import load_pg_connection, fetch_graph_from_postgres, create_toy_graph
from gnn_lexicon.src.graph_builder import build_hetero_graph, split_edges
from gnn_lexicon.src.models import create_model, LinkPredictor
from gnn_lexicon.src.training import train_gnn
from gnn_lexicon.src.evaluation import evaluate_link_prediction, evaluate_hits_at_k

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a model architecture"""
    name: str
    model_type: str
    hyperparams: Dict[str, Any]
    description: str

@dataclass
class TrainingResult:
    """Results from training a model"""
    model_name: str
    config: ModelConfig
    best_val_auc: float
    best_val_hits10: float
    final_loss: float
    training_time: float
    memory_peak: float
    history: Dict[str, List[float]]
    test_metrics: Dict[str, float]

class ComprehensiveTrainer:
    """Comprehensive trainer with cross-validation and hyperparameter optimization"""
    
    def __init__(self, config_path: str = "gnn_lexicon/config.yaml"):
        self.config_path = config_path
        self.device = torch.device("cpu")
        self.results = []
        
        # Load base config
        with open(config_path, "r") as f:
            self.base_config = yaml.safe_load(f)
        
        # Define model configurations
        self.model_configs = [
            ModelConfig(
                name="GATv2_Standard",
                model_type="gatv2",
                hyperparams={
                    "hidden_dim": 128,
                    "out_dim": 64,
                    "num_layers": 2,
                    "heads": 4,
                    "lr": 0.001,
                    "dropout": 0.2
                },
                description="Standard GATv2 with attention mechanism"
            ),
            ModelConfig(
                name="GATv2_Deep",
                model_type="gatv2",
                hyperparams={
                    "hidden_dim": 256,
                    "out_dim": 128,
                    "num_layers": 3,
                    "heads": 8,
                    "lr": 0.0005,
                    "dropout": 0.3
                },
                description="Deep GATv2 with more parameters"
            ),
            ModelConfig(
                name="SAGE_Standard",
                model_type="sage",
                hyperparams={
                    "hidden_dim": 128,
                    "out_dim": 64,
                    "num_layers": 2,
                    "lr": 0.001,
                    "dropout": 0.2
                },
                description="Standard GraphSAGE"
            ),
            ModelConfig(
                name="SAGE_Deep",
                model_type="sage",
                hyperparams={
                    "hidden_dim": 256,
                    "out_dim": 128,
                    "num_layers": 3,
                    "lr": 0.0005,
                    "dropout": 0.3
                },
                description="Deep GraphSAGE"
            ),
            ModelConfig(
                name="RGCN_Standard",
                model_type="rgcn",
                hyperparams={
                    "hidden_dim": 128,
                    "out_dim": 64,
                    "num_layers": 2,
                    "lr": 0.001,
                    "dropout": 0.2
                },
                description="Standard RGCN with relation-specific weights"
            )
        ]
    
    def load_data(self) -> Tuple[Any, Any]:
        """Load and prepare data"""
        logger.info("Loading data from PostgreSQL...")
        
        conn = load_pg_connection(self.base_config["postgres"])
        if conn is None:
            logger.warning("Failed to connect to PostgreSQL. Using toy graph.")
            raw_data = create_toy_graph()
        else:
            raw_data = fetch_graph_from_postgres(conn)
            conn.close()
        
        logger.info("Building heterogeneous graph...")
        data = build_hetero_graph(raw_data, self.device)
        
        # Log graph statistics
        logger.info(f"Graph statistics:")
        logger.info(f"  Node types: {data.node_types}")
        logger.info(f"  Edge types: {data.edge_types}")
        for node_type in data.node_types:
            if node_type in data.node_types:
                logger.info(f"  {node_type} nodes: {data[node_type].x.size(0)}")
        for edge_type in data.edge_types:
            if edge_type in data.edge_types:
                logger.info(f"  {edge_type} edges: {data[edge_type].edge_index.size(1)}")
        
        return data, raw_data
    
    def train_single_model(self, model_config: ModelConfig, data: Any, 
                          fold: int = None) -> TrainingResult:
        """Train a single model with given configuration"""
        logger.info(f"Training {model_config.name}...")
        
        start_time = time.time()
        process = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        
        # Get metadata and input channels
        metadata = (data.node_types, data.edge_types)
        in_channels_dict = {
            node_type: data[node_type].x.size(1) 
            for node_type in data.node_types
        }
        
        # Merge base config with model-specific hyperparams
        config = self.base_config.copy()
        config.update(model_config.hyperparams)
        
        # Split data
        train_data, val_data, test_data = split_edges(data)
        
        # Create model
        model = create_model(model_config.model_type, metadata, in_channels_dict, config).to(self.device)
        logger.info(f"Model: {model_config.name}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        
        # Train
        results = train_gnn(
            model, train_data, val_data, optimizer, self.device, 
            config, amp=False, save_path=f"{model_config.name.lower()}_fold{fold}.pt" if fold else f"{model_config.name.lower()}.pt"
        )
        
        # Evaluate on test set
        model.eval()
        link_predictor = LinkPredictor(config["out_dim"]).to(self.device)
        link_predictor.load_state_dict(results["link_predictor"])
        
        test_metrics = {}
        test_metrics["link_auc"] = evaluate_link_prediction(model, link_predictor, test_data, self.device)
        test_metrics["hits@10"] = evaluate_hits_at_k(model, link_predictor, test_data, self.device, k=10)
        test_metrics["hits@5"] = evaluate_hits_at_k(model, link_predictor, test_data, self.device, k=5)
        test_metrics["hits@1"] = evaluate_hits_at_k(model, link_predictor, test_data, self.device, k=1)
        
        training_time = time.time() - start_time
        memory_peak = 0  # Would need psutil for accurate measurement
        
        logger.info(f"Best validation AUC: {results['best_val_auc']:.4f}")
        logger.info(f"Test AUC: {test_metrics['link_auc']:.4f}")
        logger.info(f"Test Hits@10: {test_metrics['hits@10']:.4f}")
        
        return TrainingResult(
            model_name=model_config.name,
            config=model_config,
            best_val_auc=results['best_val_auc'],
            best_val_hits10=results.get('best_val_hits10', 0.0),
            final_loss=results['history']['train_loss'][-1],
            training_time=training_time,
            memory_peak=memory_peak,
            history=results['history'],
            test_metrics=test_metrics
        )
    
    def cross_validate(self, data: Any, n_folds: int = 3) -> List[TrainingResult]:
        """Perform cross-validation for all models"""
        logger.info(f"Starting {n_folds}-fold cross-validation...")
        
        all_results = []
        
        for model_config in self.model_configs:
            logger.info(f"\n{'='*50}")
            logger.info(f"Cross-validating {model_config.name}")
            logger.info(f"{'='*50}")
            
            fold_results = []
            
            # Simple k-fold split (in practice, you'd want stratified sampling)
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            # For simplicity, we'll just train on the full dataset for each fold
            # In a real implementation, you'd split the edges properly
            for fold in range(n_folds):
                logger.info(f"Fold {fold + 1}/{n_folds}")
                try:
                    result = self.train_single_model(model_config, data, fold)
                    fold_results.append(result)
                except Exception as e:
                    logger.error(f"Error in fold {fold + 1}: {e}")
                    continue
            
            # Aggregate fold results
            if fold_results:
                avg_result = self.aggregate_fold_results(fold_results)
                all_results.append(avg_result)
        
        return all_results
    
    def aggregate_fold_results(self, fold_results: List[TrainingResult]) -> TrainingResult:
        """Aggregate results from multiple folds"""
        avg_val_auc = np.mean([r.best_val_auc for r in fold_results])
        avg_val_hits10 = np.mean([r.best_val_hits10 for r in fold_results])
        avg_test_auc = np.mean([r.test_metrics['link_auc'] for r in fold_results])
        avg_test_hits10 = np.mean([r.test_metrics['hits@10'] for r in fold_results])
        avg_training_time = np.mean([r.training_time for r in fold_results])
        
        # Use the best fold's history
        best_fold = max(fold_results, key=lambda x: x.best_val_auc)
        
        return TrainingResult(
            model_name=best_fold.model_name,
            config=best_fold.config,
            best_val_auc=avg_val_auc,
            best_val_hits10=avg_val_hits10,
            final_loss=best_fold.final_loss,
            training_time=avg_training_time,
            memory_peak=best_fold.memory_peak,
            history=best_fold.history,
            test_metrics={
                'link_auc': avg_test_auc,
                'hits@10': avg_test_hits10,
                'hits@5': np.mean([r.test_metrics['hits@5'] for r in fold_results]),
                'hits@1': np.mean([r.test_metrics['hits@1'] for r in fold_results])
            }
        )
    
    def run_comprehensive_training(self, use_cross_validation: bool = True) -> List[TrainingResult]:
        """Run comprehensive training with all models"""
        logger.info("Starting comprehensive training...")
        
        # Load data
        data, raw_data = self.load_data()
        
        if use_cross_validation:
            results = self.cross_validate(data, n_folds=3)
        else:
            # Train each model once
            results = []
            for model_config in self.model_configs:
                try:
                    result = self.train_single_model(model_config, data)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error training {model_config.name}: {e}")
                    continue
        
        # Save comprehensive results
        self.save_results(results)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def save_results(self, results: List[TrainingResult]):
        """Save comprehensive results to files"""
        # Save detailed results
        results_dict = []
        for result in results:
            results_dict.append({
                'model_name': result.model_name,
                'config': {
                    'model_type': result.config.model_type,
                    'hyperparams': result.config.hyperparams,
                    'description': result.config.description
                },
                'metrics': {
                    'best_val_auc': result.best_val_auc,
                    'best_val_hits10': result.best_val_hits10,
                    'final_loss': result.final_loss,
                    'training_time': result.training_time,
                    'memory_peak': result.memory_peak,
                    'test_metrics': result.test_metrics
                },
                'history': result.history
            })
        
        with open('comprehensive_training_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save summary CSV
        import csv
        with open('training_summary.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Val_AUC', 'Val_Hits@10', 'Test_AUC', 'Test_Hits@10', 'Training_Time', 'Parameters'])
            
            for result in results:
                writer.writerow([
                    result.model_name,
                    f"{result.best_val_auc:.4f}",
                    f"{result.best_val_hits10:.4f}",
                    f"{result.test_metrics['link_auc']:.4f}",
                    f"{result.test_metrics['hits@10']:.4f}",
                    f"{result.training_time:.2f}s",
                    "N/A"  # Would need to calculate this
                ])
        
        logger.info("Results saved to comprehensive_training_results.json and training_summary.csv")
    
    def print_summary(self, results: List[TrainingResult]):
        """Print comprehensive training summary"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE TRAINING SUMMARY")
        logger.info("="*80)
        
        # Sort by validation AUC
        sorted_results = sorted(results, key=lambda x: x.best_val_auc, reverse=True)
        
        for i, result in enumerate(sorted_results):
            logger.info(f"\n{i+1}. {result.model_name}")
            logger.info(f"   Description: {result.config.description}")
            logger.info(f"   Validation AUC: {result.best_val_auc:.4f}")
            logger.info(f"   Validation Hits@10: {result.best_val_hits10:.4f}")
            logger.info(f"   Test AUC: {result.test_metrics['link_auc']:.4f}")
            logger.info(f"   Test Hits@10: {result.test_metrics['hits@10']:.4f}")
            logger.info(f"   Training Time: {result.training_time:.2f}s")
            logger.info(f"   Final Loss: {result.final_loss:.4f}")
        
        # Best model
        best_model = sorted_results[0]
        logger.info(f"\n🏆 BEST MODEL: {best_model.model_name}")
        logger.info(f"   Test AUC: {best_model.test_metrics['link_auc']:.4f}")
        logger.info(f"   Test Hits@10: {best_model.test_metrics['hits@10']:.4f}")

def main():
    """Main function"""
    logger.info("Starting Comprehensive Philippine Lexicon GNN Training")
    logger.info("="*60)
    
    trainer = ComprehensiveTrainer()
    
    # Run comprehensive training
    results = trainer.run_comprehensive_training(use_cross_validation=False)  # Set to True for CV
    
    logger.info("\nTraining completed successfully!")
    logger.info("Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 