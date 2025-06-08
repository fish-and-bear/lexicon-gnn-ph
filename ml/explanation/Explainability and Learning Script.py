# === Explainability and Active Learning Script ===

import logging
import os
import sys
import torch
import dgl # type: ignore
import pandas as pd
import numpy as np
from pathlib import Path
import json
import random
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm # Use auto for notebook/script adaptability
import time
from datetime import datetime
import pickle
import gc
import copy
import importlib
from sklearn.metrics import accuracy_score, classification_report # For Active Learning evaluation (optional)
import networkx as nx
import csv # Add csv import

# --- Try to import optional dependencies and set flags ---
try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    # Define a placeholder if dgl is critical and not found, or handle absence gracefully
    class DGLGraphPlaceholder:
        pass # Simplistic placeholder
    dgl = DGLGraphPlaceholder() # Allows type hints like dgl.DGLGraph to not error immediately

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Check if dgl.to_networkx is available (relies on DGL and NetworkX both being available)
DGL_GRAPH_TO_NETWORKX_AVAILABLE = DGL_AVAILABLE and NETWORKX_AVAILABLE and hasattr(dgl, 'to_networkx')

# Check for pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# --- Basic Logging Setup (early for setup messages) ---
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s - %(message)s')
logger_expl = logging.getLogger(__name__) # Specific logger for this script

# --- Early Path Setup Function (minimal, must be defined before project imports) ---
def _early_setup_sys_path_expl() -> Path:
    """Minimal path setup to ensure project modules can be imported for Expl/AL script."""
    current_dir = Path(os.getcwd())
    project_root = current_dir
    if not ( (project_root / 'ml').exists() and (project_root / 'data').exists() ):
        if (project_root.parent / 'ml').exists() and (project_root.parent / 'data').exists():
            project_root = project_root.parent
        else:
            logger_expl.warning(
                f"Early sys.path setup (EXPL): Could not reliably find project root containing 'ml' and 'data' "
                f"from {current_dir} or its parent. Imports may fail."
            )
    abs_project_root = project_root.resolve()
    if str(abs_project_root) not in sys.path:
        sys.path.insert(0, str(abs_project_root))
        logger_expl.info(f"Early sys.path setup (EXPL): Added {abs_project_root} to sys.path")
    logger_expl.info(f"Early sys.path setup (EXPL): Project root for imports determined as: {abs_project_root}")
    return abs_project_root

_project_root_for_imports_expl = _early_setup_sys_path_expl()

# --- Project-specific imports ---
try:
    from ml.models.hgmae import HGMAE
    from ml.data.feature_extraction import LexicalFeatureExtractor
    from ml.data.lexical_graph_builder import LexicalGraphBuilder
    from ml.data.db_adapter import DatabaseAdapter
    from ml.explanation.pg_explainer import PGExplainer # Key for explainability
    logger_expl.info("Successfully imported HGMAE, LFE, LGB, DBAdapter, PGExplainer.")
except ImportError as e:
    logger_expl.error(f"Could not import core project-specific classes: {e}. Using placeholders. Ensure script is run from project root and modules exist.")
    class HGMAE(torch.nn.Module): # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: super().__init__(); logger_expl.warning("Using DUMMY HGMAE")
        def encode(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]: return {"word": torch.empty(0)}
        @property
        def rel_names(self) -> List[str]: return []
        @property
        def edge_decoder(self) -> torch.nn.ModuleList: return torch.nn.ModuleList()
        @property
        def gnn_encoder(self) -> Optional[torch.nn.Module]: return None


    class LexicalFeatureExtractor: # type: ignore
        EXPECTED_DFS: List[str] = []
        def __init__(self, *args: Any, **kwargs: Any) -> None: logger_expl.warning("Using DUMMY LexicalFeatureExtractor")
        def extract_all_features(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[int]]]: return {}, {}

    class LexicalGraphBuilder: # type: ignore
        EXPECTED_DFS: List[str] = ['lemmas_df', 'relations_df', 'definitions_df', 'etymologies_df', 'pos_df']
        EXPECTED_DFS_FOR_BUILD_GRAPH: List[str] = ['lemmas_df', 'relations_df', 'definitions_df', 'etymologies_df']
        def __init__(self, *args: Any, **kwargs: Any) -> None: logger_expl.warning("Using DUMMY LexicalGraphBuilder")
        def build_graph(self, *args: Any, **kwargs: Any) -> Optional[dgl.DGLGraph]: return None

    class DatabaseAdapter: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: logger_expl.warning("Using DUMMY DatabaseAdapter")
        def get_lemmas_df(self, *args: Any, **kwargs: Any) -> pd.DataFrame: return pd.DataFrame()
        def get_definitions_df(self, *args: Any, **kwargs: Any) -> pd.DataFrame: return pd.DataFrame()
        def get_pos_df(self, *args: Any, **kwargs: Any) -> pd.DataFrame: return pd.DataFrame()
        def get_etymologies_df(self, *args: Any, **kwargs: Any) -> pd.DataFrame: return pd.DataFrame()
        def get_relations_df(self, *args: Any, **kwargs: Any) -> pd.DataFrame: return pd.DataFrame()

    # PGExplainer placeholder would be defined here if import failed.
    # However, we want our LocalPGExplainer to be primary.

# Define LocalPGExplainer unconditionally at the global scope.
# This will be used instead of any imported PGExplainer due to instantiation changes made previously.
class LocalPGExplainer: # type: ignore # Renamed from PGExplainer
    def __init__(self, model: torch.nn.Module, graph: dgl.DGLGraph, node_features: Dict[str, torch.Tensor], device: torch.device, config: Optional[Dict] = None, **kwargs: Any):
        logger_expl.warning("Using LocalPGExplainer (modified placeholder).")
        self.model = model # The GNN encoder part
        self.graph = graph
        self.node_features = node_features
            self.device = device
        self.config = config if config else {}
        
        # Example: Initialize any internal models or parameters PGExplainer might need
        # self.explanation_model = self._create_explanation_model()
        # self.optimizer = torch.optim.Adam(self.explanation_model.parameters(), lr=self.config.get("lr", 0.003))


    def _create_explanation_model(self):
        # Placeholder for internal model components if PGExplainer has its own trainable parts for generating explanations
        # This structure depends heavily on the specific PGExplainer algorithm being implemented.
        # For a typical edge-masking PGExplainer, this might involve learning edge masks.
        # For simplicity, this is just a conceptual placeholder.
        # If PGExplainer learns edge masks directly on the input graph structure without a separate nn.Module,
        # then this might not be a separate PyTorch module.
        # Consider what parameters need to be learned for the explanation.
        pass

    # Added new explain_node method to match how it's called later in the code
    def explain_node(self, 
                     original_graph: dgl.DGLGraph,
                     node_idx_to_explain: int,
                     node_ntype: str,
                     all_embs_global: Dict[str, torch.Tensor],
                     k_hop: int = 2,
                     top_k_edges_per_etype: int = 5
                    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Updated explain_node method that better matches how it's called in the main function.
        
        Args:
            original_graph: The complete graph
            node_idx_to_explain: Index of node to explain in the original graph
            node_ntype: Node type of the node to explain
            all_embs_global: Pre-computed embeddings for all nodes
            k_hop: Number of hops for subgraph extraction
            top_k_edges_per_etype: Number of top edges to return per edge type
            
        Returns:
            Dictionary mapping edge types to tuples of (edge_scores, src_nodes, dst_nodes)
        """
        logger_expl.info(f"LocalPGExplainer.explain_node called for node {node_idx_to_explain} ({node_ntype})")
        
        try:
            # Extract k-hop subgraph
            seed_nodes_dict = {node_ntype: torch.tensor([node_idx_to_explain], device=self.device)}
            
            # Fill with empty tensors for other node types
            for ntype_iter in original_graph.ntypes:
                if ntype_iter not in seed_nodes_dict:
                    seed_nodes_dict[ntype_iter] = torch.empty((0,), dtype=torch.long, device=self.device)
            
            # Try to get khop_subgraph, attempting dgl.sampling.khop_subgraph if dgl.khop_subgraph fails
            try:
                khop_fn = dgl.khop_subgraph
            except AttributeError:
                logger_expl.warning("dgl.khop_subgraph not found, trying dgl.sampling.khop_subgraph")
                try:
                    import dgl.sampling
                    khop_fn = dgl.sampling.khop_subgraph
                except (ImportError, AttributeError) as e_sampling:
                    logger_expl.error(f"Could not find khop_subgraph in dgl or dgl.sampling: {e_sampling}")
                    raise AttributeError("khop_subgraph function not found in DGL.") from e_sampling

            # Extract k-hop subgraph
            k_hop_subgraph, node_dict, edge_dict = khop_fn(
                original_graph.to(self.device), 
                seed_nodes_dict, 
                k=k_hop,
                store_ids=True
            )
            
            # Create explanations dictionary to store results
            explanations = {}
            
            # For each edge type, generate simulated importance scores
            for etype in k_hop_subgraph.canonical_etypes:
                src_type, rel_type, dst_type = etype
                
                # Get edges of this type in the subgraph
                src_nodes, dst_nodes = k_hop_subgraph.edges(etype=etype)
                
                # Skip if no edges of this type
                if len(src_nodes) == 0:
                    continue
                    
                # Generate simulated edge importance scores (in a real implementation, these would be learned)
                edge_scores = torch.rand(len(src_nodes), device=self.device)
                
                # Store the results
                explanations[rel_type] = (edge_scores, src_nodes, dst_nodes)
            
            logger_expl.info(f"Generated explanations for {len(explanations)} edge types")
            return explanations
            
        except Exception as e:
            logger_expl.error(f"Error in LocalPGExplainer.explain_node: {e}", exc_info=True)
            return {}

# --- Global Variables for Explainability/Active Learning Script ---
PROJECT_ROOT_EXPL: Optional[Path] = None
DB_CONFIG_PATH_EXPL: Optional[Path] = None
CONFIG_EXPL: Optional[Dict] = None
MODEL_CHECKPOINT_DIR_EXPL: Path = Path("ml/output/pipeline_run_fixed/models/pretrained/") # Default
RESULTS_BASE_DIR_EXPL: Path = Path("data/analysis_results/explainability_active_learning") # Default
SKIP_CACHE_EXPL: bool = False
DEVICE_EXPL: Optional[torch.device] = None
pos_classification_head_global = None  # Define global variable
all_pos_tags_sorted_global = None  # Define global variable

# --- POS Classification Head (adapted from ML_Pipeline_Colab_Script_Mod_with_Plotting.py) ---
class POSClassificationHead(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        logger_expl.info(f"POSClassificationHead initialized: input_dim={input_dim}, num_classes={num_classes}, hidden_dim={hidden_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x # Returns logits

# --- Setup and Helper Functions (adapted from KG script) ---
def _setup_colab_project_root_and_sys_path_expl() -> Path:
    """Sets up the project root and updates sys.path for EXPL script."""
    # This is identical to the KG script's version but uses logger_expl
    current_dir = Path(os.getcwd())
    project_root = current_dir
    if not ( (project_root / 'ml').exists() and (project_root / 'data').exists() ):
        if ( (project_root.parent / 'ml').exists() and (project_root.parent / 'data').exists() ):
            project_root = project_root.parent
        else:
            logger_expl.warning(f"Could not reliably find project root from {current_dir} for EXPL script.")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger_expl.info(f"Added {project_root} to sys.path for EXPL script")
    logger_expl.info(f"Project root determined as: {project_root} for EXPL script")
    return project_root

def _setup_environment_and_config_expl(
    config_file_path_override: Optional[str] = None,
    db_config_file_path_override: Optional[str] = None,
    model_dir_override_param: Optional[str] = None,
    results_dir_override_param: Optional[str] = None,
    device_override_param: Optional[str] = None
) -> Tuple[Optional[Dict], Optional[torch.device]]:
    """Sets up paths, device, and loads main configuration for the EXPL script."""
    global CONFIG_EXPL, PROJECT_ROOT_EXPL, DB_CONFIG_PATH_EXPL, MODEL_CHECKPOINT_DIR_EXPL, RESULTS_BASE_DIR_EXPL, SKIP_CACHE_EXPL

    if PROJECT_ROOT_EXPL is None:
        PROJECT_ROOT_EXPL = _setup_colab_project_root_and_sys_path_expl()
    if PROJECT_ROOT_EXPL is None:
        logger_expl.error("PROJECT_ROOT_EXPL could not be determined. Exiting EXPL setup.")
        return None, None

    actual_config_path_to_load: Optional[Path] = None
    if config_file_path_override:
        candidate_path = Path(config_file_path_override)
        actual_config_path_to_load = PROJECT_ROOT_EXPL / candidate_path if not candidate_path.is_absolute() else candidate_path
    else:
        actual_config_path_to_load = PROJECT_ROOT_EXPL / "ml" / "config" / "default_config.json"

    if actual_config_path_to_load and actual_config_path_to_load.exists():
        try:
            with open(actual_config_path_to_load, 'r') as f:
                CONFIG_EXPL = json.load(f)
            logger_expl.info(f"Main config (CONFIG_EXPL) loaded from {actual_config_path_to_load}")
        except json.JSONDecodeError as e:
            logger_expl.error(f"Error decoding main config JSON for EXPL from {actual_config_path_to_load}: {e}")
            CONFIG_EXPL = None
    else:
        logger_expl.error(f"Main config file for EXPL not found at {actual_config_path_to_load}")
        CONFIG_EXPL = None

    if CONFIG_EXPL is None:
        logger_expl.error("Main configuration (CONFIG_EXPL) could not be loaded. Device context and other params may be incorrect.")
        # Allow to proceed to determine device with defaults, but some operations might fail.
    
    # DB Config Path (reuse logic from KG script, using expl_params if available, else general)
    db_config_section = CONFIG_EXPL.get("expl_active_learning_params", CONFIG_EXPL.get("kg_enhancement_params", {})) if CONFIG_EXPL else {}
    db_config_path_from_main_config = db_config_section.get("db_config_path")
    candidate_db_path: Optional[Path] = None
    if db_config_file_path_override:
        candidate_db_path = Path(db_config_file_path_override)
    elif db_config_path_from_main_config:
        candidate_db_path = Path(db_config_path_from_main_config)
    else: # Default if not in overrides or config
        candidate_db_path = Path("ml/my_db_config.json")
    
    if not candidate_db_path.is_absolute():
            candidate_db_path = PROJECT_ROOT_EXPL / candidate_db_path
            
    if candidate_db_path.exists():
        DB_CONFIG_PATH_EXPL = candidate_db_path
    else: # Fallback if primary candidate not found
        DB_CONFIG_PATH_EXPL = PROJECT_ROOT_EXPL / "ml" / "db_config.json"
        if not DB_CONFIG_PATH_EXPL.exists():
            logger_expl.warning(f"DB config not found at {candidate_db_path} or {DB_CONFIG_PATH_EXPL}. DB ops might fail.")
            DB_CONFIG_PATH_EXPL = None # Explicitly None
    if DB_CONFIG_PATH_EXPL: logger_expl.info(f"DB config path (DB_CONFIG_PATH_EXPL) set to: {DB_CONFIG_PATH_EXPL}")

    # Model Checkpoint Dir
    model_dir_section = CONFIG_EXPL.get("expl_active_learning_params", CONFIG_EXPL.get("kg_enhancement_params", {})) if CONFIG_EXPL else {}
    model_dir_from_config = model_dir_section.get("model_checkpoint_dir")
    model_dir_override_val = model_dir_override_param or model_dir_from_config
    if model_dir_override_val:
        MODEL_CHECKPOINT_DIR_EXPL = Path(model_dir_override_val)
    if not MODEL_CHECKPOINT_DIR_EXPL.is_absolute():
        MODEL_CHECKPOINT_DIR_EXPL = PROJECT_ROOT_EXPL / MODEL_CHECKPOINT_DIR_EXPL
    MODEL_CHECKPOINT_DIR_EXPL = MODEL_CHECKPOINT_DIR_EXPL.resolve()
    logger_expl.info(f"Model checkpoint dir (MODEL_CHECKPOINT_DIR_EXPL) set to: {MODEL_CHECKPOINT_DIR_EXPL}")

    # Results Dir
    results_dir_section = CONFIG_EXPL.get("expl_active_learning_params", {}) if CONFIG_EXPL else {}
    results_dir_from_config = results_dir_section.get("results_base_dir")
    results_dir_override_val = results_dir_override_param or results_dir_from_config
    if results_dir_override_val: # If explicitly passed or in config section for EX_AL
        RESULTS_BASE_DIR_EXPL = Path(results_dir_override_val)
    # Default RESULTS_BASE_DIR_EXPL is already "data/analysis_results/explainability_active_learning"
    if not RESULTS_BASE_DIR_EXPL.is_absolute():
        RESULTS_BASE_DIR_EXPL = PROJECT_ROOT_EXPL / RESULTS_BASE_DIR_EXPL
    RESULTS_BASE_DIR_EXPL = RESULTS_BASE_DIR_EXPL.resolve()
    logger_expl.info(f"Results base dir (RESULTS_BASE_DIR_EXPL) set to: {RESULTS_BASE_DIR_EXPL}")
    try: RESULTS_BASE_DIR_EXPL.mkdir(parents=True, exist_ok=True)
    except Exception as e_mkdir_base_expl: logger_expl.error(f"Could not create base results dir {RESULTS_BASE_DIR_EXPL}: {e_mkdir_base_expl}")

    # Skip Cache
    skip_cache_section = CONFIG_EXPL.get("expl_active_learning_params", CONFIG_EXPL.get("kg_enhancement_params", {})) if CONFIG_EXPL else {}
    SKIP_CACHE_EXPL = skip_cache_section.get("skip_cache", SKIP_CACHE_EXPL)
    logger_expl.info(f"Skip cache (SKIP_CACHE_EXPL) set to: {SKIP_CACHE_EXPL}")

    # Device Setup
    determined_device_expl: Optional[torch.device] = None
    if device_override_param:
        try:
            determined_device_expl = torch.device(device_override_param)
            logger_expl.info(f"Device override from parameter: Using {determined_device_expl}")
        except RuntimeError as e:
            logger_expl.error(f"Invalid device_override_param '{device_override_param}': {e}. Defaulting.")
    if determined_device_expl is None:
        if torch.cuda.is_available():
            determined_device_expl = torch.device("cuda")
        else:
            determined_device_expl = torch.device("cpu")
    logger_expl.info(f"Device determined in _setup_environment_and_config_expl: {determined_device_expl}")
    return CONFIG_EXPL, determined_device_expl

def load_latest_checkpoint_expl(model_dir_path: Path) -> Tuple[Optional[Dict], Optional[Path]]:
    """Loads the latest model checkpoint for EXPL script."""
    if not model_dir_path.exists():
        logger_expl.error(f"Checkpoint directory does not exist: {model_dir_path}")
        return None, None
    checkpoints = list(model_dir_path.glob("hgmae_pretrain_*.pt"))
    if not checkpoints:
        logger_expl.error(f"No checkpoints found in {model_dir_path}")
        return None, None
    latest_checkpoint_path = max(checkpoints, key=os.path.getctime)
    logger_expl.info(f"Loading latest checkpoint for EXPL: {latest_checkpoint_path}")
    effective_device_for_load = DEVICE_EXPL if DEVICE_EXPL is not None else torch.device('cpu')
    try:
        checkpoint = torch.load(latest_checkpoint_path, map_location=effective_device_for_load)
        return checkpoint, latest_checkpoint_path
    except Exception as e:
        logger_expl.error(f"Error loading checkpoint {latest_checkpoint_path}: {e}", exc_info=True)
        return None, None

def _merge_pos_information_expl(
    data_frames: Dict[str, Optional[pd.DataFrame]],
    logger_instance: logging.Logger
) -> Dict[str, Optional[pd.DataFrame]]:
    """Merges POS information into definitions_df and lemmas_df. Adapted for EXPL script."""
    # This function is critical for getting POS tags for words.
    # Reusing the logic from KG script's _merge_pos_information.
    logger_instance.info("--- Starting POS Information Merge Process (EXPL Script Minimal Version) ---")
    current_definitions_df = data_frames.get('definitions_df')
    current_pos_df = data_frames.get('pos_df')
    current_lemmas_df = data_frames.get('lemmas_df')

    if current_pos_df is None or current_pos_df.empty:
        logger_instance.warning("pos_df is missing or empty (EXPL). Cannot merge POS information.")
        if current_lemmas_df is not None and 'pos_code' not in current_lemmas_df.columns:
            current_lemmas_df['pos_code'] = pd.NA # Add empty column if it doesn't exist
        if current_lemmas_df is not None : data_frames['lemmas_df'] = current_lemmas_df
        return data_frames

    if current_definitions_df is None or current_definitions_df.empty:
        logger_instance.warning("definitions_df is missing or empty (EXPL). Cannot merge POS information via definitions.")
        if current_lemmas_df is not None and 'pos_code' not in current_lemmas_df.columns:
             current_lemmas_df['pos_code'] = pd.NA
        if current_lemmas_df is not None: data_frames['lemmas_df'] = current_lemmas_df
        return data_frames

    # Ensure required columns exist
    if 'standardized_pos_id' not in current_definitions_df.columns:
        logger_instance.error("'standardized_pos_id' column missing in definitions_df (EXPL). POS merge will likely fail.")
        return data_frames
    if 'id' not in current_pos_df.columns or 'code' not in current_pos_df.columns:
        logger_instance.error("'id' or 'code' column missing in pos_df (EXPL). POS merge will likely fail.")
        return data_frames

    # Convert IDs to numeric, coercing errors
    current_definitions_df['standardized_pos_id'] = pd.to_numeric(current_definitions_df['standardized_pos_id'], errors='coerce')
    current_pos_df['id'] = pd.to_numeric(current_pos_df['id'], errors='coerce')

    # Merge pos_code into definitions_df
    merged_definitions_df = pd.merge(
        current_definitions_df,
        current_pos_df[['id', 'code']], # Select only necessary columns from pos_df
        left_on='standardized_pos_id',
        right_on='id',
        how='left',
        suffixes=('', '_pos_table') # Suffix for 'id' from pos_df if it's also in definitions_df
    ).rename(columns={'code': 'pos_code', 'id_pos_table': 'pos_id_from_pos_table'}) # Rename 'code' to 'pos_code'

    data_frames['definitions_df'] = merged_definitions_df # Update the DataFrame in the dictionary
    if 'pos_code' not in merged_definitions_df.columns:
        logger_instance.error("'pos_code' column NOT FOUND in definitions_df after merge (EXPL).")
    else:
        logger_instance.info(f"Added 'pos_code' to definitions_df (EXPL). Non-null count: {merged_definitions_df['pos_code'].notnull().sum()}")


    # Merge primary POS from definitions into lemmas_df
    if current_lemmas_df is not None and not current_lemmas_df.empty and \
       'pos_code' in merged_definitions_df.columns and \
       'word_id' in merged_definitions_df.columns: # Ensure 'word_id' is in definitions for linking

        merged_definitions_df['word_id'] = pd.to_numeric(merged_definitions_df['word_id'], errors='coerce')
        # Sort definitions to pick the first POS code per word_id (e.g., by definition ID or some other priority)
        sort_by_cols = ['word_id']
        if 'id' in merged_definitions_df.columns: # If definitions have their own 'id' column
            merged_definitions_df['id'] = pd.to_numeric(merged_definitions_df['id'], errors='coerce')
            sort_by_cols.append('id') # Sort by definition id as secondary key

        sorted_defs = merged_definitions_df.sort_values(by=sort_by_cols)
        # Get the first non-null 'pos_code' for each 'word_id'
        word_primary_pos = sorted_defs.dropna(subset=['pos_code']).groupby('word_id', as_index=False)['pos_code'].first()

        # Merge this primary POS into lemmas_df
        if 'id' not in current_lemmas_df.columns: # lemmas_df 'id' is the word_id
            logger_instance.error("'id' column missing in current_lemmas_df (EXPL). Cannot merge primary POS.")
            if 'pos_code' not in current_lemmas_df.columns : current_lemmas_df['pos_code'] = pd.NA
            data_frames['lemmas_df'] = current_lemmas_df
            return data_frames
        current_lemmas_df['id'] = pd.to_numeric(current_lemmas_df['id'], errors='coerce')

        current_lemmas_df = pd.merge(current_lemmas_df,
                                     word_primary_pos, # Contains 'word_id' and 'pos_code'
                                     left_on='id',     # Lemma's own ID
                                     right_on='word_id', # word_id from aggregated definitions
                                     how='left',
                                     suffixes=('_original_in_lemma', '_from_def')) # Suffix for 'pos_code' if it already existed

        # Consolidate 'pos_code' column
        if 'pos_code_from_def' in current_lemmas_df.columns:
            if 'pos_code_original_in_lemma' in current_lemmas_df.columns:
                current_lemmas_df['pos_code'] = current_lemmas_df['pos_code_from_def'].fillna(current_lemmas_df['pos_code_original_in_lemma'])
                # Drop the temporary columns
                current_lemmas_df.drop(columns=['pos_code_original_in_lemma', 'pos_code_from_def'], inplace=True, errors='ignore')
            else: # 'pos_code_original_in_lemma' didn't exist, so rename 'pos_code_from_def'
                current_lemmas_df.rename(columns={'pos_code_from_def': 'pos_code'}, inplace=True)
        elif 'pos_code_original_in_lemma' in current_lemmas_df.columns:
             current_lemmas_df.rename(columns={'pos_code_original_in_lemma': 'pos_code'}, inplace=True)
        elif 'pos_code' not in current_lemmas_df.columns : # If no pos_code column exists at all after merges
            current_lemmas_df['pos_code'] = pd.NA # Create it as empty

        # Clean up helper 'word_id' column if it was added from word_primary_pos merge
        if 'word_id' in current_lemmas_df.columns:
            # Check if 'word_id' was originally in lemmas_df. If not, it's the merge helper.
            # This is a bit heuristic. A safer way is to check original_lemmas_df.columns if we had it.
            # For now, assume if 'id' is the primary key, 'word_id' is the helper from the merge.
            original_lemmas_df_for_check = data_frames.get('lemmas_df', pd.DataFrame()) # get the df that was passed in
            if 'word_id' not in original_lemmas_df_for_check.columns: # if original didn't have 'word_id'
                 current_lemmas_df.drop(columns=['word_id'], inplace=True, errors='ignore')


        data_frames['lemmas_df'] = current_lemmas_df # Update the DataFrame in the dictionary
        if 'pos_code' in current_lemmas_df.columns:
            logger_instance.info(f"Added/updated 'pos_code' in lemmas_df (EXPL). Missing: {current_lemmas_df['pos_code'].isnull().sum()}/{len(current_lemmas_df)}")
        else:
            logger_instance.warning("'pos_code' column still not found in 'lemmas_df' after primary POS merge attempt (EXPL).")

    logger_instance.info("--- Finished POS Information Merge Process (EXPL Script) ---")
    return data_frames

def load_data_for_expl_active_learning(
    data_cache_dir: Path,
    db_config_path_expl_val: Path,
    main_config_from_file: Dict,
    lfe_expected_feat_dims: Optional[Dict[str, int]] = None,
    skip_cache_val: bool = False,
    device_val: Optional[torch.device] = None
) -> Tuple[Optional[dgl.DGLGraph], Optional[Dict[str, torch.Tensor]], Optional[Dict[str, Any]], Optional[Dict[str, pd.DataFrame]], Optional[Dict[str, List[int]]], Optional[Dict[str, int]], Optional[List[str]]]:
    """Loads data for Explainability and Active Learning. Adapts KG script's load_data_for_postprocessing."""
    cache_file = data_cache_dir / "expl_al_data_cache.pt" # Use a different cache name
    g, features_dict, graph_info, data_frames_loaded, final_ordered_ids_map, actual_final_feat_dims = [None] * 6
    all_pos_tags_sorted: Optional[List[str]] = None


    if not skip_cache_val and cache_file.exists():
        try:
            logger_expl.info(f"Attempting to load all data from EXPL cache: {cache_file}")
            cached_data_expl = torch.load(cache_file)
            g = cached_data_expl.get('graph')
            features_dict = cached_data_expl.get('features_dict')
            graph_info = cached_data_expl.get('graph_info')
            data_frames_loaded = cached_data_expl.get('data_frames_loaded')
            final_ordered_ids_map = cached_data_expl.get('final_ordered_ids_map')
            actual_final_feat_dims = cached_data_expl.get('actual_final_feat_dims')
            all_pos_tags_sorted = cached_data_expl.get('all_pos_tags_sorted')


            if g is not None: g = g.to(device_val if device_val is not None else torch.device('cpu'))
            if features_dict is not None:
                features_dict = {k: v.to(device_val if device_val is not None else torch.device('cpu')) for k, v in features_dict.items()}
            logger_expl.info("Successfully loaded all data from EXPL cache.")
            if not all([g, features_dict, graph_info, data_frames_loaded, final_ordered_ids_map, actual_final_feat_dims, all_pos_tags_sorted]):
                logger_expl.warning("One or more EXPL cached components were None. Will reload from DB.")
                g, features_dict, graph_info, data_frames_loaded, final_ordered_ids_map, actual_final_feat_dims, all_pos_tags_sorted = [None] * 7
            else:
                 if lfe_expected_feat_dims and actual_final_feat_dims and lfe_expected_feat_dims != actual_final_feat_dims:
                    logger_expl.warning(f"Cached EXPL features have dims {actual_final_feat_dims}, but expected {lfe_expected_feat_dims}. Consider clearing cache.")

        except Exception as e_cache:
            logger_expl.error(f"Failed to load data from EXPL cache: {e_cache}. Will reload from DB.", exc_info=True)
            g, features_dict, graph_info, data_frames_loaded, final_ordered_ids_map, actual_final_feat_dims, all_pos_tags_sorted = [None] * 7

    if g is None: # If cache loading failed or skipped
        logger_expl.info("Loading required DataFrames from database for EXPL script...")
        if db_config_path_expl_val is None or not db_config_path_expl_val.exists():
            logger_expl.error(f"DB config path for EXPL is invalid: {db_config_path_expl_val}")
            return None, None, None, None, None, None, None
        
        try:
            if 'ml.data.db_adapter' in sys.modules:
                reloaded_db_adapter_module = importlib.reload(sys.modules['ml.data.db_adapter'])
                DatabaseAdapter_reloaded = getattr(reloaded_db_adapter_module, 'DatabaseAdapter')
            else: from ml.data.db_adapter import DatabaseAdapter as DatabaseAdapter_reloaded
            
            with open(db_config_path_expl_val, 'r') as f: db_config_content = json.load(f)
            actual_db_params = db_config_content.get("db_config", db_config_content) # Handle potential nesting
            db_adapter = DatabaseAdapter_reloaded(db_config=actual_db_params)
        except Exception as e_db_init:
            logger_expl.error(f"Error initializing DatabaseAdapter for EXPL: {e_db_init}", exc_info=True)
            return None, None, None, None, None, None, None

        data_frames_raw = db_adapter.get_all_dataframes(
            languages_to_include=main_config_from_file.get('data_params', {}).get('languages_to_use', ['tl'])
        )
        db_adapter.close()
        if not data_frames_raw or 'lemmas_df' not in data_frames_raw or data_frames_raw['lemmas_df'] is None:
            logger_expl.error("Essential 'lemmas_df' not loaded from DB for EXPL. Cannot proceed.")
            return None, None, None, None, None, None, None

        data_frames_loaded = _merge_pos_information_expl(data_frames_raw, logger_expl)

        # Get unique POS tags from lemmas_df for classification head
        if data_frames_loaded and 'lemmas_df' in data_frames_loaded and data_frames_loaded['lemmas_df'] is not None and 'pos_code' in data_frames_loaded['lemmas_df'].columns:
            all_pos_tags_sorted = sorted(data_frames_loaded['lemmas_df']['pos_code'].dropna().unique().tolist())
            logger_expl.info(f"Found {len(all_pos_tags_sorted)} unique POS tags in lemmas_df for EXPL: {all_pos_tags_sorted[:10]}...")
        else:
            logger_expl.warning("Could not extract unique POS tags from lemmas_df for EXPL.")
            all_pos_tags_sorted = []


        logger_expl.info("Building graph for EXPL script...")
        graph_builder = LexicalGraphBuilder(
            config=main_config_from_file,
            target_languages=main_config_from_file.get('data_params', {}).get('languages_to_use', ['tl'])
        )
        dfs_for_build_graph_call = {}
        mandatory_dfs_for_builder = getattr(LexicalGraphBuilder, 'EXPECTED_DFS_FOR_BUILD_GRAPH', [])
        for df_name in mandatory_dfs_for_builder:
            if df_name in data_frames_loaded and data_frames_loaded[df_name] is not None and not data_frames_loaded[df_name].empty:
                dfs_for_build_graph_call[df_name] = data_frames_loaded[df_name]
            else:
                logger_expl.error(f"Missing mandatory DataFrame '{df_name}' for graph building (EXPL).")
                return None, None, None, data_frames_loaded, None, None, all_pos_tags_sorted
        
        g = graph_builder.build_graph(**dfs_for_build_graph_call)
        if g is None:
            logger_expl.error("Graph building failed for EXPL script.")
            return None, None, None, data_frames_loaded, None, None, all_pos_tags_sorted

        graph_info = {
            'node_types_ordered': getattr(graph_builder, 'node_types_ordered', g.ntypes),
            'etypes': getattr(graph_builder, 'etypes', g.etypes),
            'canonical_etypes': g.canonical_etypes
        }
        
        final_ordered_ids_map = {
            'word': getattr(graph_builder, 'node_to_word_id', {}),
            'definition': getattr(graph_builder, 'node_to_def_id', {}),
            'etymology': getattr(graph_builder, 'node_to_etym_id', {})
        }

        logger_expl.info(f"LFE for EXPL will be initialized with model_original_feat_dims: {lfe_expected_feat_dims}")
        st_model_name_lfe = main_config_from_file.get('feature_extraction', {}).get('sentence_transformer_model_name')
        lfe = LexicalFeatureExtractor(
            data_frames=data_frames_loaded,
            graph_data=final_ordered_ids_map,
            model_original_feat_dims=lfe_expected_feat_dims if lfe_expected_feat_dims else {},
            sentence_transformer_model_name=st_model_name_lfe,
            device=device_val
        )
        features_dict, final_ordered_ids_map_from_lfe = lfe.extract_all_features(
            lemmas_df=data_frames_loaded.get('lemmas_df'),
            graph_num_nodes_per_type={ntype: g.num_nodes(ntype) for ntype in g.ntypes},
            node_to_original_id_maps=final_ordered_ids_map,
            target_device=device_val
        )
        if final_ordered_ids_map_from_lfe: final_ordered_ids_map = final_ordered_ids_map_from_lfe
        actual_final_feat_dims = {ntype: feat.shape[1] for ntype, feat in features_dict.items()} if features_dict else {}

        # Save to EXPL cache
        data_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.save({
                'graph': g.cpu() if g else None, # Save graph on CPU
                'features_dict': {k: v.cpu() for k,v in features_dict.items()} if features_dict else None, # Save features on CPU
                'graph_info': graph_info,
                'data_frames_loaded': data_frames_loaded,
                'final_ordered_ids_map': final_ordered_ids_map,
                'actual_final_feat_dims': actual_final_feat_dims,
                'all_pos_tags_sorted': all_pos_tags_sorted
            }, cache_file)
            logger_expl.info("Data loaded from DB for EXPL. Saved to EXPL cache.")
        except Exception as e_cache_save:
            logger_expl.error(f"Error saving data to EXPL cache: {e_cache_save}", exc_info=True)

    # Move to target device after loading from cache or DB
    if g is not None and device_val: g = g.to(device_val)
    if features_dict is not None and device_val:
        features_dict = {k: v.to(device_val) for k, v in features_dict.items()}
    
    return g, features_dict, graph_info, data_frames_loaded, final_ordered_ids_map, actual_final_feat_dims, all_pos_tags_sorted

def get_node_details_expl(
    original_id_in: Optional[int],
    ntype: str,
    data_frames: Optional[Dict[str, pd.DataFrame]],
    final_ordered_ids_map: Optional[Dict[str, List[int]]], # This map is {ntype: [original_id_for_graph_idx_0, ...]}
    graph_node_idx: Optional[int] = None
) -> Tuple[int, str, Optional[str]]: # original_id, text_representation, pos_tag
    """
    Retrieves details for a node given its graph index or original ID.
    Handles mapping between graph indices and original IDs.
    """
    original_id_out: Optional[int] = original_id_in
    text_representation: str = "Unknown"
    pos_tag_out: Optional[str] = None

    # Prioritize graph_node_idx if provided, to map to original_id first
    if graph_node_idx is not None and final_ordered_ids_map and ntype in final_ordered_ids_map:
        if 0 <= graph_node_idx < len(final_ordered_ids_map[ntype]):
            original_id_out = final_ordered_ids_map[ntype][graph_node_idx]
            else:
            logger_expl.warning(f"Graph node index {graph_node_idx} for ntype '{ntype}' is out of bounds for final_ordered_ids_map. Cannot map to original ID.")
            # original_id_out remains as original_id_in, which might be None

    if original_id_out is None: # If still no original ID (e.g., graph_node_idx was not mappable or not provided)
        # logger_expl.debug(f"Node details: Original ID not determined for ntype '{ntype}', graph_idx {graph_node_idx}.")
        return -1, f"Unknown {ntype} (graph_idx:{graph_node_idx})", None # Return a clear unknown representation

    # Proceed with original_id_out to fetch details
    if data_frames and isinstance(data_frames, dict):
        if ntype == 'word' and 'lemmas_df' in data_frames and data_frames['lemmas_df'] is not None:
            lemma_row = data_frames['lemmas_df'][data_frames['lemmas_df']['id'] == original_id_out] # Changed 'lemma_id' to 'id'
            if not lemma_row.empty:
                text_representation = lemma_row.iloc[0].get('lemma', f"Word OriginalID:{original_id_out}") # Changed 'lemma_text' to 'lemma'
                pos_tag_out = lemma_row.iloc[0].get('pos_code', None) # Changed 'pos_tag_clean' to 'pos_code'
        elif ntype == 'definition' and 'definitions_df' in data_frames and data_frames['definitions_df'] is not None:
            def_row = data_frames['definitions_df'][data_frames['definitions_df']['id'] == original_id_out] # Assuming 'id' for definitions_df
            if not def_row.empty:
                text_representation = def_row.iloc[0].get('definition', f"Definition OriginalID:{original_id_out}") # Assuming 'definition' for text
        elif ntype == 'etymology' and 'etymologies_df' in data_frames and data_frames['etymologies_df'] is not None:
            ety_row = data_frames['etymologies_df'][data_frames['etymologies_df']['id'] == original_id_out] # Assuming 'id' for etymologies_df
            if not ety_row.empty:
                text_representation = ety_row.iloc[0].get('summary', f"Etymology OriginalID:{original_id_out}") # Assuming 'summary' for text
        elif ntype == 'pos' and 'pos_df' in data_frames and data_frames['pos_df'] is not None:
            pos_row = data_frames['pos_df'][data_frames['pos_df']['id'] == original_id_out] # Assuming 'id' for pos_df
            if not pos_row.empty:
                text_representation = pos_row.iloc[0].get('code', f"POS OriginalID:{original_id_out}") # Assuming 'code' for text, and it is the POS tag
    else:
            text_representation = f"{ntype.capitalize()} OriginalID:{original_id_out}"
    else:
        text_representation = f"DataFrames unavailable for {ntype} OriginalID:{original_id_out}"
        
    # Ensure original_id_out is an int before returning
    final_original_id = int(original_id_out) if original_id_out is not None else -1 # Use -1 if somehow still None

    return final_original_id, text_representation, pos_tag_out


# Function to be used by PGExplainer: Predicts POS tag probability for a target node in a subgraph
def prediction_for_pgexplainer_pos(
    model_encoder: HGMAE, # The GNN encoder part of HGMAE
    subgraph: dgl.DGLGraph,
    node_features_on_subgraph: Dict[str, torch.Tensor], # Features for nodes in the subgraph
    target_node_local_id: int, # LOCAL ID of the target node within the subgraph
    target_node_ntype: str # Node type of the target node (should be 'word')
) -> Optional[torch.Tensor]:
    """
    Prediction function for PGExplainer, tailored for POS tagging of 'word' nodes.
    Returns class probabilities (after softmax) for the target node.
    """
    global pos_classification_head_global, DEVICE_EXPL # Use globals
    if target_node_ntype != 'word':
        logger_expl.warning(f"PGExplainer prediction function called for non-'word' ntype: {target_node_ntype}. Returning None.")
        return None
    if pos_classification_head_global is None:
        logger_expl.error("pos_classification_head_global is not set. Cannot make predictions for PGExplainer.")
        return None

    try:
        model_encoder.eval() # Ensure encoder is in eval mode
        pos_classification_head_global.eval() # Ensure head is in eval mode

        # Ensure subgraph and features are on the correct device
        subgraph = subgraph.to(DEVICE_EXPL)
        node_features_on_subgraph = {k: v.to(DEVICE_EXPL) for k, v in node_features_on_subgraph.items()}

        with torch.no_grad():
            # 1. Get all node embeddings from the GNN encoder for the subgraph
            all_node_embeddings_subgraph = model_encoder.encode(subgraph, node_features_on_subgraph)

            if target_node_ntype not in all_node_embeddings_subgraph or \
               all_node_embeddings_subgraph[target_node_ntype] is None or \
               all_node_embeddings_subgraph[target_node_ntype].shape[0] <= target_node_local_id:
                logger_expl.error(f"Could not get embedding for target node {target_node_local_id} ({target_node_ntype}) in subgraph.")
                return None

            # 2. Get the embedding for the specific target node
            target_node_embedding = all_node_embeddings_subgraph[target_node_ntype][target_node_local_id].unsqueeze(0) # [1, hidden_dim]

            # 3. Pass through the POS classification head
            logits = pos_classification_head_global(target_node_embedding) # [1, num_pos_classes]
            
            # 4. Convert logits to probabilities (softmax)
            probabilities = F.softmax(logits, dim=-1) # [1, num_pos_classes]
            return probabilities.squeeze(0) # Return [num_pos_classes]

    except Exception as e:
        logger_expl.error(f"Error in prediction_for_pgexplainer_pos: {e}", exc_info=True)
        return None

# --- Main Explainability and Active Learning Function ---
def main_expl_active_learning(
    config_file_path: Optional[str] = None,
    db_config_file_path: Optional[str] = None,
    model_dir_override: Optional[str] = None,
    results_dir_override: Optional[str] = None,
    device_override: Optional[str] = None,
    num_nodes_to_explain: int = 3,
    num_samples_for_active_learning: int = 10
):
    run_start_time = time.time()
    logger_expl.info("=" * 80)
    logger_expl.info("🚀🚀🚀 STARTING EXPLAINABILITY & ACTIVE LEARNING ANALYSIS 🚀🚀🚀")
    logger_expl.info("=" * 80)

    global CONFIG_EXPL, PROJECT_ROOT_EXPL, DB_CONFIG_PATH_EXPL, MODEL_CHECKPOINT_DIR_EXPL, RESULTS_BASE_DIR_EXPL, SKIP_CACHE_EXPL, DEVICE_EXPL
    global pos_classification_head_global, all_pos_tags_sorted_global # Make them assignable

    config_setup_result_expl, device_from_setup_expl = _setup_environment_and_config_expl(
        config_file_path_override=config_file_path,
        db_config_file_path_override=db_config_file_path,
        model_dir_override_param=model_dir_override,
        results_dir_override_param=results_dir_override,
        device_override_param=device_override
    )
    CONFIG_EXPL = config_setup_result_expl
    DEVICE_EXPL = device_from_setup_expl

    if CONFIG_EXPL is None or DEVICE_EXPL is None or PROJECT_ROOT_EXPL is None:
        logger_expl.error("Failed to setup environment for EXPL script. Terminating.")
        return

    # --- Load Checkpoint and Model ---
    model_expl: Optional[HGMAE] = None
    checkpoint_data_expl, loaded_checkpoint_path_expl = load_latest_checkpoint_expl(MODEL_CHECKPOINT_DIR_EXPL)
    checkpoint_stem_name_expl = "unknown_checkpoint"
    lfe_expected_dims_from_checkpoint_expl: Optional[Dict[str, int]] = None

    if checkpoint_data_expl and loaded_checkpoint_path_expl:
        checkpoint_stem_name_expl = loaded_checkpoint_path_expl.stem
        model_constructor_params_expl = checkpoint_data_expl.get('model_constructor_params_used')
        model_state_dict_expl = checkpoint_data_expl.get('model_state_dict')
        if model_constructor_params_expl:
            lfe_expected_dims_from_checkpoint_expl = model_constructor_params_expl.get('original_feat_dims')
        if model_constructor_params_expl and model_state_dict_expl:
            try:
                model_expl = HGMAE(**model_constructor_params_expl)
                model_expl.load_state_dict(model_state_dict_expl)
                model_expl.to(DEVICE_EXPL)
                model_expl.eval()
                logger_expl.info(f"HGMAE model loaded from {loaded_checkpoint_path_expl.name} and moved to {DEVICE_EXPL}")
            except Exception as e:
                logger_expl.error(f"Error instantiating/loading model for EXPL: {e}", exc_info=True)
                model_expl = None
        else:
            logger_expl.warning("Checkpoint loaded, but missing constructor params or state dict for EXPL model.")
    else:
        logger_expl.warning("No model checkpoint found for EXPL script. Explainability and Active Learning might be limited.")

    # --- Results Directory for this Run ---
    results_dir_for_run_expl = RESULTS_BASE_DIR_EXPL / f"run_{checkpoint_stem_name_expl}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try: results_dir_for_run_expl.mkdir(parents=True, exist_ok=True)
    except OSError as e: logger_expl.error(f"Could not create results dir {results_dir_for_run_expl}: {e}")


    # --- Load Data ---
    if DB_CONFIG_PATH_EXPL is None:
        logger_expl.error("DB_CONFIG_PATH_EXPL is None. Cannot load data.")
        return
        
    graph_expl, features_dict_expl, graph_info_expl, data_frames_expl, final_ordered_ids_map_expl, actual_final_feat_dims_expl, all_pos_tags_sorted_loaded = \
        load_data_for_expl_active_learning(
            data_cache_dir=PROJECT_ROOT_EXPL / "data" / "cache",
            db_config_path_expl_val=DB_CONFIG_PATH_EXPL,
            main_config_from_file=CONFIG_EXPL,
            lfe_expected_feat_dims=lfe_expected_dims_from_checkpoint_expl,
            skip_cache_val=SKIP_CACHE_EXPL,
            device_val=DEVICE_EXPL
        )
    
    all_pos_tags_sorted_global = all_pos_tags_sorted_loaded # Set global for prediction function

    if not all([graph_expl, features_dict_expl, graph_info_expl, data_frames_expl, final_ordered_ids_map_expl, actual_final_feat_dims_expl, all_pos_tags_sorted_global]):
        logger_expl.error("Failed to load one or more critical data components for EXPL script. Terminating.")
        return
    
    # --- Setup POS Classification Head (for explainability and active learning) ---
    if model_expl and 'word' in actual_final_feat_dims_expl and all_pos_tags_sorted_global:
        # word_embedding_dim = actual_final_feat_dims_expl['word'] # This was the original raw feature dim
        # The POS head should operate on the output embeddings of the HGMAE model
        if hasattr(model_expl, 'hidden_dim') and model_expl.hidden_dim is not None:
            word_embedding_dim_for_pos_head = model_expl.hidden_dim
        elif hasattr(model_expl, 'out_dim') and model_expl.out_dim is not None: # Fallback if hidden_dim not top-level
            word_embedding_dim_for_pos_head = model_expl.out_dim
        else:
            # Attempt to infer from constructor params if available
            model_constructor_params_expl = checkpoint_data_expl.get('model_constructor_params_used', {}) if checkpoint_data_expl else {}
            word_embedding_dim_for_pos_head = model_constructor_params_expl.get('hidden_dim', 128) # Default to 128 if not found
            logger_expl.warning(f"Could not directly read hidden_dim/out_dim from model_expl, inferred {word_embedding_dim_for_pos_head} for POS head input dim.")

        num_pos_classes = len(all_pos_tags_sorted_global)
        
        hidden_dim_pos_head = CONFIG_EXPL.get("expl_active_learning_params", {}).get("pos_head_hidden_dim", 128)
        
        # Initialize the head first, assuming it might be new
        pos_classification_head_global = POSClassificationHead(
            input_dim=word_embedding_dim_for_pos_head, 
            num_classes=num_pos_classes, 
            hidden_dim=hidden_dim_pos_head
        )
        logger_expl.info("Initialized a new POSClassificationHead instance.")
        
        # Attempt to load pre-trained state dict if available in the checkpoint
        if checkpoint_data_expl: # Ensure checkpoint_data_expl itself is not None
            pos_head_state_dict = checkpoint_data_expl.get('pos_classification_head_state_dict')
            if pos_head_state_dict:
                logger_expl.info("Found 'pos_classification_head_state_dict' in checkpoint. Attempting to load.")
                try:
                    pos_classification_head_global.load_state_dict(pos_head_state_dict)
                    logger_expl.info(f"Successfully loaded pre-trained POSClassificationHead state from checkpoint into the existing instance.")
                except RuntimeError as e_load_pos_head:
                    logger_expl.error(f"Error loading POSClassificationHead state from checkpoint: {e_load_pos_head}. The existing (newly initialized) head will be used.", exc_info=True)
                except Exception as e_load_generic:
                    logger_expl.error(f"A generic error occurred while trying to load POSClassificationHead state: {e_load_generic}. The existing (newly initialized) head will be used.", exc_info=True)
            else:
                logger_expl.info("No 'pos_classification_head_state_dict' found in checkpoint. The newly initialized POSClassificationHead will be used.")
        else:
            logger_expl.info("Checkpoint data (checkpoint_data_expl) not available. The newly initialized POSClassificationHead will be used.")
            
        pos_classification_head_global.to(DEVICE_EXPL)
        pos_classification_head_global.eval()
        logger_expl.info(f"POSClassificationHead setup complete. Head is on {DEVICE_EXPL}. Input dim: {word_embedding_dim_for_pos_head}, Output classes: {num_pos_classes}")
    else:
        logger_expl.warning("Cannot setup POSClassificationHead: model, word features, or POS tags missing.")


    # === Explainability Section (PGExplainer for Node Classification) ===
    if model_expl and graph_expl and features_dict_expl and pos_classification_head_global and 'word' in graph_expl.ntypes:
        logger_expl.info("=" * 70)
        logger_expl.info("🔍 Starting Explainability Analysis (PGExplainer for POS Tagging) 🔍")
        logger_expl.info("=" * 70)

        pg_explainer_config = CONFIG_EXPL.get("pg_explainer_params", {})
        pg_epochs = pg_explainer_config.get("epochs", 30)
        pg_lr = pg_explainer_config.get("lr", 0.003)
        k_hop_pg = pg_explainer_config.get("k_hop", 2)
        
        # Select a few 'word' nodes to explain
        word_nodes_global_ids = graph_expl.nodes('word').tolist()
        if not word_nodes_global_ids:
            logger_expl.warning("No 'word' nodes in the graph to explain.")
        else:
            sample_word_indices_to_explain = random.sample(
                range(len(word_nodes_global_ids)), 
                min(num_nodes_to_explain, len(word_nodes_global_ids))
            )
            
            for i, global_word_node_idx_in_graph in enumerate(sample_word_indices_to_explain):
                original_word_id, word_text, word_pos_tag_true = get_node_details_expl(
                    None, 'word', data_frames_expl, final_ordered_ids_map_expl, graph_node_idx=global_word_node_idx_in_graph
                )
                logger_expl.info(f"\n--- Explaining Word Node {i+1}/{num_nodes_to_explain}: '{word_text}' (Original ID: {original_word_id}, Graph Index: {global_word_node_idx_in_graph}, True POS: {word_pos_tag_true}) ---")

                if model_expl.encoder is None:
                    logger_expl.error("Model's encoder (HGNN) is None. Cannot proceed with PGExplainer.")
                    continue

                current_gnn_model_for_explainer = model_expl.encoder
                # pos_head_global should be defined and loaded earlier in the script if it's used here.
                # Assuming pos_head_global is the trained POSClassificationHead instance.
                
                # Define the node type to explain (needed both for explanation and visualization)
                node_ntype = 'word'

                # Define a specific prediction function for this node for PGExplainer
                def specific_target_pred_func(subgraph_expl: dgl.DGLGraph, 
                                              subgraph_features_expl: Dict[str, torch.Tensor],
                                              target_node_local_id_in_subgraph: int, # Added: PGExplainer needs to tell us which node in subgraph is target
                                              target_node_ntype_in_subgraph: str # Added: And its type
                                              ) -> Optional[torch.Tensor]:
                    model_device = next(current_gnn_model_for_explainer.parameters()).device
                    subgraph_expl = subgraph_expl.to(model_device)
                    processed_subgraph_features_expl = {
                        ntype: feat.to(model_device) for ntype, feat in subgraph_features_expl.items()
                    }

                    gnn_output_embeddings_subgraph = current_gnn_model_for_explainer(
                        subgraph_expl, 
                        processed_subgraph_features_expl
                    )
                    
                    target_node_embedding_subgraph_all = gnn_output_embeddings_subgraph.get(target_node_ntype_in_subgraph)
                    if target_node_embedding_subgraph_all is None:
                        logger_expl.error(f"PGExplainer pred_fn: Could not get embeddings for ntype {target_node_ntype_in_subgraph} from GNN output on subgraph.")
                        return None
                    
                    if target_node_local_id_in_subgraph >= target_node_embedding_subgraph_all.shape[0]:
                        logger_expl.error(f"PGExplainer pred_fn: target_node_local_id_in_subgraph {target_node_local_id_in_subgraph} is out of bounds for ntype {target_node_ntype_in_subgraph} with {target_node_embedding_subgraph_all.shape[0]} nodes in subgraph GNN output.")
                        return None
                        
                    target_node_final_embedding = target_node_embedding_subgraph_all[target_node_local_id_in_subgraph].unsqueeze(0)

                    if pos_classification_head_global is None: # pos_classification_head_global should be the loaded POS head
                        logger_expl.error("PGExplainer pred_fn: pos_classification_head_global is None.")
                        return None
                    
                    pos_classification_head_global_device = next(pos_classification_head_global.parameters()).device
                    target_node_final_embedding = target_node_final_embedding.to(pos_classification_head_global_device)
                    pos_head_to_use = pos_classification_head_global.to(pos_classification_head_global_device)

                    logits = pos_head_to_use(target_node_final_embedding)
                    probabilities = F.softmax(logits, dim=-1)
                    return probabilities

                try:
                    # Instantiate PGExplainer
                    # PGExplainer's `model` argument should be the part of the model that processes graph structure to get embeddings (i.e., the GNN encoder)
                    # The `target_prediction_function` will then take these embeddings (or work from the subgraph directly) and get final predictions.
                    
                    # The PGExplainer itself doesn't use the classification head directly in its constructor.
                    # The head is used by the `target_prediction_function`.
                    pg_explainer_instance = LocalPGExplainer( # Renamed
                        model=current_gnn_model_for_explainer, 
                        graph=graph_expl, # Full graph
                        node_features=features_dict_expl, # Features for the full graph
                        device=DEVICE_EXPL,
                        config=pg_explainer_config 
                    )

                    # Train the explainer's separate model (if needed by its logic for node classification)
                    # This might involve passing the `specific_target_pred_func` or a similar mechanism
                    # if PGExplainer needs to optimize its explanation mask based on final classification outputs.
                    # The current PGExplainer trains based on embedding reconstruction or link prediction.
                    # For node classification, the `explain_node` method might need to take the full pipeline.
                    
                    # The provided PGExplainer's `explain_node` needs `target_prediction_function`.
                    # It seems PGExplainer itself does not train a separate model for node classification explanations *based on a head*.
                    # It focuses on explaining the GNN's behavior in producing embeddings or predicting links.
                    # Let's assume we want to explain which parts of the graph influence the *final classification*.
                    # The `target_prediction_function` passed to `explain_node` is key.

                    # Get embeddings for all nodes in the graph first
                    all_node_embeddings = {}
                    with torch.no_grad():
                        for ntype in graph_expl.ntypes:
                            node_features = features_dict_expl.get(ntype)
                            if node_features is not None:
                                node_embeddings = current_gnn_model_for_explainer.encode(graph_expl, {ntype: node_features})[ntype]
                                all_node_embeddings[ntype] = node_embeddings

                    explanation_per_etype = pg_explainer_instance.explain_node(
                        original_graph=graph_expl,
                        node_idx_to_explain=global_word_node_idx_in_graph,
                        node_ntype=node_ntype,  # Use the variable defined above
                        all_embs_global=all_node_embeddings,
                        k_hop=k_hop_pg,
                        top_k_edges_per_etype=5  # You can adjust this value based on how many important edges you want per type
                    )

                    if explanation_per_etype:
                        logger_expl.info(f"  PGExplainer found explanations for {len(explanation_per_etype)} edge types.")
                        
                        # Log details for each edge type
                        for etype, (edge_scores, src_ids, dst_ids) in explanation_per_etype.items():
                            logger_expl.info(f"  Edge type '{etype}' - Top edge scores: {edge_scores[:5].tolist() if edge_scores.numel() > 0 else 'Empty'}")
                        
                        # Visualization of the explanation results
                        explanation_output_path_base = results_dir_for_run_expl / f"pgexpl_{word_text.replace(' ', '_')}_{original_word_id}"
                        
                        if DGL_GRAPH_TO_NETWORKX_AVAILABLE and MATPLOTLIB_AVAILABLE:
                        try:
                                # Create a new graph for visualization that includes all explained edges
                                k_hop_subgraph, _ = dgl.khop_subgraph(
                                    graph_expl,
                                    {node_ntype: torch.tensor([global_word_node_idx_in_graph], device=DEVICE_EXPL)},
                                    k=k_hop_pg
                                )
                                k_hop_subgraph = k_hop_subgraph.to('cpu')
                                
                                # Convert to networkx for visualization
                                G_nx = dgl.to_networkx(k_hop_subgraph, node_attrs=None)
                                pos = nx.spring_layout(G_nx, seed=42)
                                
                                plt.figure(figsize=(10, 8))
                                plt.title(f"PGExplainer Results for '{word_text}' (POS: {word_pos_tag_true})", fontsize=16)
                                
                                # Draw nodes
                                node_colors = ['red' if i == 0 else 'skyblue' for i in range(k_hop_subgraph.number_of_nodes())]
                                nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, node_size=50)
                                
                                # Draw edges for each edge type with different colors and weights
                                edge_colors = plt.cm.Set3(np.linspace(0, 1, len(explanation_per_etype)))
                                for (etype, (edge_scores, src_ids, dst_ids)), color in zip(explanation_per_etype.items(), edge_colors):
                                    # Draw edges with width proportional to their importance scores
                                    edges = list(zip(src_ids.tolist(), dst_ids.tolist()))
                                    if edges:  # Only draw if there are edges
                                        edge_weights = edge_scores.tolist()
                                        nx.draw_networkx_edges(G_nx, pos,
                                                            edgelist=edges,
                                                            width=[w * 3 for w in edge_weights],
                                                            edge_color=[color] * len(edges),
                                                            alpha=0.6,
                                                            label=etype)
                                
                                # Add legend for edge types
                                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                
                                # Try to add node labels
                                node_labels = {}
                                for node_idx in range(k_hop_subgraph.number_of_nodes()):
                                    _, node_text, _ = get_node_details_expl(None, 'word', data_frames_expl, final_ordered_ids_map_expl, graph_node_idx=node_idx)
                                    node_labels[node_idx] = node_text[:10] if node_text else str(node_idx)
                                nx.draw_networkx_labels(G_nx, pos, labels=node_labels, font_size=8)
                                
                                plt.tight_layout()
                                plt.savefig(explanation_output_path_base.with_suffix(".png"), bbox_inches='tight')
                                logger_expl.info(f"  Saved PGExplainer visualization to {explanation_output_path_base.with_suffix('.png')}")
                                plt.close()
                                
                                # Save explanation data
                                expl_data_to_save = {
                                    'explained_node_original_id': original_word_id,
                                    'explained_node_text': word_text,
                                    'explained_node_true_pos': word_pos_tag_true,
                                    'explanations_per_edge_type': {
                                        etype: {
                                            'edge_scores': scores.tolist(),
                                            'source_nodes': srcs.tolist(),
                                            'target_nodes': dsts.tolist()
                                        }
                                        for etype, (scores, srcs, dsts) in explanation_per_etype.items()
                                    }
                                }
                                with open(explanation_output_path_base.with_suffix(".json"), 'w') as f_json:
                                    json.dump(expl_data_to_save, f_json, indent=2)
                                logger_expl.info(f"  Saved PGExplainer data to {explanation_output_path_base.with_suffix('.json')}")
                            except Exception as e_plot:
                                logger_expl.error(f"  Error during PGExplainer visualization: {e_plot}", exc_info=True)
                    else:
                            logger_expl.warning("  Skipping PGExplainer visualization (dgl.to_networkx or matplotlib not available/failed).")
                    else:
                        logger_expl.warning(f"  PGExplainer did not return a valid explanation for node {word_text}.")

                except ImportError:
                    logger_expl.error("PGExplainer module not found or import error. Skipping PGExplainer.")
                except Exception as e_pg:
                    logger_expl.error(f"Error during PGExplainer execution for node {word_text}: {e_pg}", exc_info=True)
    else:
        logger_expl.warning("Skipping Explainability section: model, graph, features, or POS head not available.")


    # === Active Learning Section (Uncertainty Sampling for POS Tagging) ===
    if model_expl and graph_expl and features_dict_expl and data_frames_expl and \
       final_ordered_ids_map_expl and pos_classification_head_global and all_pos_tags_sorted_global and 'word' in graph_expl.ntypes:
        logger_expl.info("=" * 70)
        logger_expl.info("🎯 Starting Active Learning Analysis (Uncertainty Sampling for POS) 🎯")
        logger_expl.info("=" * 70)

        # 1. Get all 'word' node embeddings
        if 'word' not in features_dict_expl or features_dict_expl['word'] is None:
            logger_expl.error("Word features not found. Cannot proceed with Active Learning for POS.")
        else:
            word_embeddings_all = model_expl.encode(graph_expl, features_dict_expl)['word'] # Get 'word' embeddings
            word_embeddings_all = word_embeddings_all.to(DEVICE_EXPL)

            # 2. Get POS predictions (probabilities) for all words
            pos_classification_head_global.eval()
            all_word_pos_probs_list = []
            # Batch process if too many words
            batch_size_al = CONFIG_EXPL.get("expl_active_learning_params", {}).get("al_batch_size", 1024)
            
            with torch.no_grad():
                for i in range(0, word_embeddings_all.shape[0], batch_size_al):
                    batch_embs = word_embeddings_all[i:i+batch_size_al]
                    batch_logits = pos_classification_head_global(batch_embs)
                    batch_probs = F.softmax(batch_logits, dim=-1)
                    all_word_pos_probs_list.append(batch_probs.cpu()) # Move to CPU for numpy ops
            
            all_word_pos_probs = torch.cat(all_word_pos_probs_list, dim=0)

            # 3. Calculate uncertainty (entropy)
            # entropy = -torch.sum(all_word_pos_probs * torch.log(all_word_pos_probs + 1e-9), dim=1) # Add epsilon for log(0)
            # Using torch.distributions.Categorical for cleaner entropy calculation
            try:
                entropy = torch.distributions.Categorical(probs=all_word_pos_probs).entropy()
            except ValueError as e_entropy: # Can happen if probs don't sum to 1 or have NaNs
                 logger_expl.error(f"Error calculating entropy for active learning (likely due to probability values): {e_entropy}. Masking problematic rows.")
                 # Fallback: calculate entropy manually, masking rows with invalid probabilities
                 valid_probs_mask = torch.isfinite(all_word_pos_probs).all(dim=1) & (all_word_pos_probs >= 0).all(dim=1) & (torch.abs(all_word_pos_probs.sum(dim=1) - 1.0) < 1e-3)
                 entropy = torch.full((all_word_pos_probs.shape[0],), float('-inf'), dtype=torch.float32, device=all_word_pos_probs.device) # Initialize with very low entropy, ensure device match
                 if valid_probs_mask.any():
                    valid_probs = all_word_pos_probs[valid_probs_mask]
                    entropy[valid_probs_mask] = -torch.sum(valid_probs * torch.log(valid_probs + 1e-9), dim=1)


            # 4. Identify words that are "unlabeled" or need review
            # For this example, we'll consider words whose true POS tag is missing or unknown.
            # In a real scenario, you'd have a dataset split or query based on actual labels.
            lemmas_df = data_frames_expl.get('lemmas_df')
            unlabeled_word_indices_global = [] # Global graph indices of words considered "unlabeled"
            
            if lemmas_df is not None and 'pos_code' in lemmas_df.columns and 'id' in lemmas_df.columns and final_ordered_ids_map_expl.get('word'):
                # Create a reverse map: original_word_id -> global_graph_idx
                # final_ordered_ids_map_expl['word'] is a list: [orig_id_for_graph_idx_0, orig_id_for_graph_idx_1, ...]
                original_id_to_graph_idx_map = {orig_id: idx for idx, orig_id in enumerate(final_ordered_ids_map_expl['word'])}

                for idx, row in lemmas_df.iterrows():
                    if pd.isna(row['pos_code']) or row['pos_code'] == '' or str(row['pos_code']).lower() == 'unknown':
                        original_id = row['id']
                        if original_id in original_id_to_graph_idx_map:
                            unlabeled_word_indices_global.append(original_id_to_graph_idx_map[original_id])
                logger_expl.info(f"Found {len(unlabeled_word_indices_global)} 'word' nodes considered unlabeled (missing/unknown POS tag in lemmas_df).")
            else:
                logger_expl.warning("Could not determine unlabeled words for active learning. Selecting from all words.")
                unlabeled_word_indices_global = list(range(len(entropy))) # Fallback: consider all words

            if not unlabeled_word_indices_global:
                logger_expl.info("No unlabeled words found to select for active learning.")
            else:
                unlabeled_entropy = entropy[unlabeled_word_indices_global]
                
                # --- Collect all uncertain words for CSV export ---
                all_uncertain_words_data = []
                for global_graph_idx_uncertain in unlabeled_word_indices_global:
                    original_id_uncertain, text_uncertain, current_pos_uncertain = get_node_details_expl(
                        None, 'word', data_frames_expl, final_ordered_ids_map_expl, graph_node_idx=global_graph_idx_uncertain
                    )
                    pred_probs_uncertain = all_word_pos_probs[global_graph_idx_uncertain]
                    pred_pos_idx_uncertain = torch.argmax(pred_probs_uncertain).item()
                    pred_pos_tag_uncertain = all_pos_tags_sorted_global[pred_pos_idx_uncertain] if all_pos_tags_sorted_global else "N/A"
                    uncertainty_score_val = entropy[global_graph_idx_uncertain].item()
                    
                    all_uncertain_words_data.append({
                        'original_id': original_id_uncertain,
                        'graph_idx': global_graph_idx_uncertain,
                        'text': text_uncertain,
                        'current_pos_in_db': current_pos_uncertain,
                        'predicted_pos': pred_pos_tag_uncertain,
                        'predicted_pos_prob': pred_probs_uncertain[pred_pos_idx_uncertain].item(),
                        'uncertainty_entropy': uncertainty_score_val,
                        'top_3_predictions': ", ".join([f"{all_pos_tags_sorted_global[i]} ({pred_probs_uncertain[i]:.3f})" for i in torch.topk(pred_probs_uncertain, 3).indices.tolist() if all_pos_tags_sorted_global])
                    })
                
                # Sort all uncertain words by entropy (descending)
                all_uncertain_words_data.sort(key=lambda x: x['uncertainty_entropy'], reverse=True)

                # Save to CSV
                csv_file_path = results_dir_for_run_expl / "uncertain_word_nodes_sorted_by_entropy.csv"
                try:
                    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        if all_uncertain_words_data: # Check if there's data to write
                            fieldnames = all_uncertain_words_data[0].keys()
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(all_uncertain_words_data)
                            logger_expl.info(f"Saved all {len(all_uncertain_words_data)} uncertain words sorted by entropy to {csv_file_path}")
                        else:
                            logger_expl.info(f"No uncertain word data to save to CSV at {csv_file_path}")
                except Exception as e_csv:
                    logger_expl.error(f"Error saving uncertain words to CSV: {e_csv}", exc_info=True)
                # --- End CSV export ---

                
                # 5. Select top N most uncertain words from the "unlabeled" pool
                # Ensure we don't try to select more than available
                num_to_select_actual = min(num_samples_for_active_learning, len(unlabeled_entropy))
                
                if num_to_select_actual > 0:
                    top_uncertain_indices_in_unlabeled_pool = torch.topk(unlabeled_entropy, k=num_to_select_actual).indices
                    
                    # Map these indices back to global graph indices
                    selected_global_graph_indices = [unlabeled_word_indices_global[i] for i in top_uncertain_indices_in_unlabeled_pool.tolist()]

                    logger_expl.info(f"\n--- Top {num_to_select_actual} Most Uncertain Words for Active Learning (POS Tagging) ---")
                    for rank, global_graph_idx in enumerate(selected_global_graph_indices):
                        original_id, text, current_pos = get_node_details_expl(None, 'word', data_frames_expl, final_ordered_ids_map_expl, graph_node_idx=global_graph_idx)
                        pred_probs_for_word = all_word_pos_probs[global_graph_idx]
                        pred_pos_idx = torch.argmax(pred_probs_for_word).item()
                        pred_pos_tag = all_pos_tags_sorted_global[pred_pos_idx] if all_pos_tags_sorted_global else "N/A"
                        uncertainty_score = entropy[global_graph_idx].item()
                        
                        logger_expl.info(
                            f"  {rank+1}. Word: '{text}' (Orig ID: {original_id}, Graph Idx: {global_graph_idx})\n"
                            f"     Current POS in DB: {current_pos}\n"
                            f"     Predicted POS: {pred_pos_tag} (Prob: {pred_probs_for_word[pred_pos_idx]:.4f})\n"
                            f"     Uncertainty (Entropy): {uncertainty_score:.4f}\n"
                            f"     Top 3 Predicted POS: " + ", ".join([f"{all_pos_tags_sorted_global[i]} ({pred_probs_for_word[i]:.3f})" for i in torch.topk(pred_probs_for_word, 3).indices.tolist() if all_pos_tags_sorted_global])
                        )
                else:
                    logger_expl.info("No samples to select for active learning (either num_samples_for_active_learning is 0 or no unlabeled items).")

        # After calculating `entropy` for active learning
        # Corrected variable name from all_unlabeled_word_indices_global to unlabeled_word_indices_global
        # Also ensuring unlabeled_word_indices_global is a tensor for .numel()
        # or a list that can be checked for emptiness.
        plot_condition_met = False
        if isinstance(unlabeled_word_indices_global, torch.Tensor) and unlabeled_word_indices_global.numel() > 0:
            plot_condition_met = True
            entropy_to_plot_unlabeled = entropy[unlabeled_word_indices_global].cpu().numpy()
            title_unlabeled_plot = "Distribution of Uncertainty (Entropy) for UNLABELED Word Nodes (Tensor)"
            filename_unlabeled_plot = "uncertainty_distribution_unlabeled_tensor.png"
        elif isinstance(unlabeled_word_indices_global, list) and len(unlabeled_word_indices_global) > 0:
            plot_condition_met = True
            entropy_to_plot_unlabeled = entropy[torch.tensor(unlabeled_word_indices_global)].cpu().numpy()
            title_unlabeled_plot = "Distribution of Uncertainty (Entropy) for UNLABELED Word Nodes (List)"
            filename_unlabeled_plot = "uncertainty_distribution_unlabeled_list.png"

        if MATPLOTLIB_AVAILABLE:
            # Plot for all words first
            plt.figure(figsize=(10, 6))
            sns.histplot(entropy.cpu().numpy(), bins=50, kde=True)
            plt.title("Distribution of Uncertainty (Entropy) for ALL Word Nodes")
            plt.xlabel("Entropy")
            plt.ylabel("Frequency")
            uncertainty_dist_path_all = results_dir_for_run_expl / "uncertainty_distribution_all_words.png"
            plt.savefig(uncertainty_dist_path_all)
            logger_expl.info(f"Saved uncertainty distribution plot for ALL words to {uncertainty_dist_path_all}")
            plt.close()

            # Plot for unlabeled words if conditions were met
            if plot_condition_met:
                plt.figure(figsize=(10, 6))
                sns.histplot(entropy_to_plot_unlabeled, bins=50, kde=True)
                plt.title(title_unlabeled_plot)
                plt.xlabel("Entropy")
                plt.ylabel("Frequency")
                uncertainty_dist_path_unlabeled = results_dir_for_run_expl / filename_unlabeled_plot
                plt.savefig(uncertainty_dist_path_unlabeled)
                logger_expl.info(f"Saved uncertainty distribution plot for UNLABELED words to {uncertainty_dist_path_unlabeled}")
                plt.close()
    else:
        logger_expl.warning("Skipping Active Learning section: model, graph, features, dataframes, or POS head not available.")

    # Collect garbage aggressively
    gc.collect()
    if DEVICE_EXPL and DEVICE_EXPL.type == 'cuda':
        torch.cuda.empty_cache()
        logger_expl.info("Cleared CUDA cache at the end of main_expl_active_learning.")


    logger_expl.info(f"Total run time for Explainability & Active Learning script: {time.time() - run_start_time:.2f} seconds.")
    logger_expl.info("=" * 80)
    logger_expl.info("✅✅✅ EXPLAINABILITY & ACTIVE LEARNING SCRIPT FINISHED ✅✅✅")
    logger_expl.info("=" * 80)


# --- Entry point for Colab/Jupyter ---
if __name__ == '__main__' and ('ipykernel' in sys.modules or 'google.colab' in sys.modules) :
    logger_expl.info("Running Explainability & Active Learning Script in a Colab/Jupyter environment.")
    
    # Example Call (adjust parameters as needed, or load from a config file)
    # These can be driven by widgets or manual settings in a notebook.
    main_expl_active_learning(
        config_file_path=None, # Optional: "ml/config/expl_al_config.json"
        db_config_file_path=None, # Optional: "ml/my_db_config.json"
        model_dir_override=None, # Optional: "ml/output/some_other_model_dir"
        results_dir_override=None, # Optional: "data/analysis_results/my_expl_run"
        device_override=None, # Optional: "cpu" or "cuda:0"
        num_nodes_to_explain=3,
        num_samples_for_active_learning=10
    )
elif __name__ == '__main__':
    logger_expl.info("Running Explainability & Active Learning Script as a standalone Python file.")
    # Add command-line argument parsing here if needed (e.g., using argparse)
    main_expl_active_learning(num_nodes_to_explain=2, num_samples_for_active_learning=5)
