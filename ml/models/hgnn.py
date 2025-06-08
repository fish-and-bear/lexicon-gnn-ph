"""
Heterogeneous Graph Neural Network architecture for lexical knowledge graphs.

This module implements a heterogeneous GNN with:
1. R-GCN-style local message passing
2. Exphormer-style global attention
3. Multi-task output heads for link prediction and node classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import math
import logging
import time  # Add time module for performance tracking
import gc # Import garbage collector
import numpy as np
import re

logger = logging.getLogger(__name__)

class RelationalGraphConv(nn.Module):
    """
    Relational Graph Convolutional layer for heterogeneous graphs.
    Based on R-GCN but with additional features for heterogeneous graphs.
    """
    
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 rel_names: List[str],
                 global_node_type_order: Optional[List[str]] = None,
                 global_num_nodes_per_type_in_order: Optional[List[int]] = None,
                 num_bases: int = 8,
                 activation: Optional[nn.Module] = None,
                 self_loop: bool = True,
                 dropout: float = 0.0,
                 layer_norm: bool = True):
        """
        Initialize the RelationalGraphConv layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            rel_names: List of relation types in the graph
            global_node_type_order: List of node types in the graph
            global_num_nodes_per_type_in_order: List of number of nodes per type in the graph
            num_bases: Number of bases for weight decomposition
            activation: Activation function to use
            self_loop: Whether to include self-loops
            dropout: Dropout probability
            layer_norm: Whether to apply layer normalization
        """
        super().__init__()
        
        # Ensure all weight-related attributes are initialized to None by default FIRST.
        self.weight_bases = None
        self.weight_coefficients = None
        self.direct_weights = None
        self.self_weight = None # Initialize self_weight as well
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rel_names = rel_names
        self.num_bases = min(num_bases, len(rel_names))
        self.activation = activation
        self.self_loop = self_loop
        
        # Store global feature structure info
        self.global_node_type_order = global_node_type_order
        self.global_num_nodes_per_type_in_order = global_num_nodes_per_type_in_order
        if self.global_node_type_order is not None and self.global_num_nodes_per_type_in_order is not None:
            self.global_cumulative_nodes = torch.cumsum(torch.tensor([0] + self.global_num_nodes_per_type_in_order[:-1]), dim=0)
            self.global_node_type_to_start_idx = {
                ntype: self.global_cumulative_nodes[i].item()
                for i, ntype in enumerate(self.global_node_type_order)
            }
        else:
            # This layer might not be used in a context where feat is concatenated globally
            # Or it's an error if it is. For now, allow None.
            logger.debug("RelationalGraphConv initialized without global_node_type_order. Assumes non-concatenated input or single node type context for 'feat'.")
        
        # Initialize weight attributes to None by default
        self.weight_bases = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        self.weight_coefficients = nn.Parameter(torch.Tensor(len(rel_names), self.num_bases))
        nn.init.xavier_uniform_(self.weight_bases)
        nn.init.xavier_uniform_(self.weight_coefficients)
        
        # Self-loop weight (if needed)
        if self.self_loop:
            self.self_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        
        # Additional components
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_out = nn.LayerNorm(out_dim) if layer_norm else None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize learnable parameters."""
        if self.weight_bases is not None and self.weight_coefficients is not None:
        nn.init.xavier_uniform_(self.weight_bases)
        nn.init.xavier_uniform_(self.weight_coefficients)
        
        if self.direct_weights is not None: # Added to re-initialize direct weights
            for rel in self.rel_names: # self.rel_names must be non-empty if direct_weights is non-None
                 # Ensure rel a key exists before trying to init. replace(':', '_') is applied at creation.
                 # So, self.direct_weights keys are already sanitized.
                 weight_key = rel.replace(':', '_')
                 if weight_key in self.direct_weights:
                    nn.init.xavier_uniform_(self.direct_weights[weight_key])
                 else:
                    # This case should ideally not happen if __init__ is correct
                    logger.warning(f"Key {weight_key} (from rel {rel}) not found in direct_weights during reset_parameters.")

        if self.self_loop and self.self_weight is not None: # Check self_weight is not None
            nn.init.xavier_uniform_(self.self_weight)
    
    def get_rel_weight(self, rel_name_or_idx: Union[str, int]) -> torch.Tensor:
        actual_rel_idx: Optional[int] = None
        actual_rel_name: Optional[str] = None

        if not self.rel_names:
            # If there are no relations defined for this layer, no relation-specific weight can be fetched.
            # This situation should ideally be handled by the caller (e.g., not calling this method,
            # or the layer only using self-loops if configured).
            raise ValueError(f"get_rel_weight called but self.rel_names is empty. Cannot fetch weight for '{rel_name_or_idx}'.")

        if isinstance(rel_name_or_idx, str):
            try:
                actual_rel_idx = self.rel_names.index(rel_name_or_idx)
                actual_rel_name = rel_name_or_idx
            except ValueError:
                # String name not found in self.rel_names
                raise ValueError(f"Relation string '{rel_name_or_idx}' not in self.rel_names (current: {self.rel_names}) for this RGC layer.")
        else: # int
            actual_rel_idx = rel_name_or_idx
            if not (0 <= actual_rel_idx < len(self.rel_names)): # Check bounds
                raise IndexError(f"Relation index {actual_rel_idx} is out of bounds for self.rel_names (length {len(self.rel_names)}, content: {self.rel_names}).")
            actual_rel_name = self.rel_names[actual_rel_idx]

        # At this point, actual_rel_idx is a valid index for self.rel_names (which is non-empty),
        # and actual_rel_name is the corresponding relation name string.

        if self.weight_bases is not None and self.weight_coefficients is not None:
            # Ensure actual_rel_idx is valid for self.weight_coefficients
            # self.weight_coefficients has shape (len(rel_names), self.num_bases)
            if not (0 <= actual_rel_idx < self.weight_coefficients.shape[0]):
                 raise IndexError(f"Internal error: actual_rel_idx {actual_rel_idx} out of bounds for weight_coefficients first dimension {self.weight_coefficients.shape[0]}. This should not happen if rel_names is consistent.")
            coeff = self.weight_coefficients[actual_rel_idx]
            return torch.einsum('b, bio -> io', coeff, self.weight_bases)
        elif self.direct_weights is not None:
            # actual_rel_name is derived from a valid index.
            # Keys in direct_weights are sanitized (e.g., rel.replace(':', '_'))
            sanitized_rel_name = actual_rel_name.replace(':', '_')
            if sanitized_rel_name not in self.direct_weights:
                raise KeyError(f"Internal error: Sanitized relation name '{sanitized_rel_name}' (from '{actual_rel_name}') not found in direct_weights. Keys: {list(self.direct_weights.keys())}")
            return self.direct_weights[sanitized_rel_name]
        else:
            # This state (self.rel_names is non-empty, but neither basis nor direct weights are initialized)
            # should not be possible if __init__ is correct.
            raise RuntimeError(f"Internal error in get_rel_weight: No weights (bases or direct) found, but self.rel_names is non-empty ('{actual_rel_name}' requested). RGC layer may be misconfigured.")

    def batched_matmul(self, matrix_a, matrix_b, batch_size_gpu_direct_threshold=500000, cpu_batch_size=64):
        """
        Performs matrix multiplication. Prioritizes direct GPU matmul.
        Falls back to CPU batched matmul if direct GPU fails or if input is too large.
        If CPU fallback occurs for a CUDA input, attempts to move result back to CUDA;
        raises RuntimeError if this move fails.
        
        Args:
            matrix_a: First matrix of shape (M, K)
            matrix_b: Second matrix of shape (K, N)
            batch_size_gpu_direct_threshold: Max rows in matrix_a for direct GPU matmul attempt.
            cpu_batch_size: Batch size to use if computation falls back to CPU.
            
        Returns:
            Result matrix of shape (M, N) on the original device of matrix_a,
            unless CPU fallback occurs and moving back to CUDA fails (raises error).
        """
        start_time = time.time()
        a_rows, a_cols = matrix_a.shape
        k_rows, b_cols = matrix_b.shape # matrix_b.shape[0] should be a_cols
        original_device = matrix_a.device

        logger.debug(f"batched_matmul: {a_rows}x{a_cols} @ {k_rows}x{b_cols} on {original_device}")

        if original_device.type == 'cuda':
            if a_rows <= batch_size_gpu_direct_threshold:
                try:
                    # Direct GPU matmul - assumes this requires gradients
                    result = torch.matmul(matrix_a, matrix_b)
                    logger.debug(f"Direct GPU matmul successful for {a_rows}x{a_cols} @ {k_rows}x{b_cols}. Time: {time.time() - start_time:.2f}s")
                    return result
                except RuntimeError as e:
                    logger.warning(f"Direct GPU matmul failed for {a_rows}x{a_cols} @ {k_rows}x{b_cols} (matrix_a device: {matrix_a.device}, matrix_b device: {matrix_b.device}): {e}. Falling back to CPU batched matmul.")
                    # Ensure inputs are on CPU for the fallback
                    matrix_a = matrix_a.to('cpu')
                    matrix_b = matrix_b.to('cpu')
                    # Proceed to CPU batched logic
            else:
                logger.info(f"Matrix A has {a_rows} rows, exceeding GPU direct threshold {batch_size_gpu_direct_threshold}. Using CPU batched matmul.")
                matrix_a = matrix_a.to('cpu')
                matrix_b = matrix_b.to('cpu')
                # Proceed to CPU batched logic
        
        # --- CPU Batched Matmul Fallback/Path ---
        # Ensure matrices are on CPU if we reached here
        if matrix_a.device.type != 'cpu': matrix_a = matrix_a.to('cpu')
        if matrix_b.device.type != 'cpu': matrix_b = matrix_b.to('cpu')

        # Adjust CPU batch size for very large matrices
        effective_cpu_batch_size = cpu_batch_size
        if a_rows > 100000:
            effective_cpu_batch_size = min(8, cpu_batch_size)
            logger.info(f"  Using micro-batch CPU size {effective_cpu_batch_size} for large matrix ({a_rows} rows)")
        elif a_rows > 10000:
            effective_cpu_batch_size = min(16, cpu_batch_size)
            logger.info(f"  Using small CPU batch size {effective_cpu_batch_size} for large matrix ({a_rows} rows)")
        
        result_on_cpu = torch.zeros(a_rows, b_cols, device='cpu')
        num_batches = (a_rows + effective_cpu_batch_size - 1) // effective_cpu_batch_size
        
        logger.info(f"Starting CPU batched matmul: {num_batches} batches of size {effective_cpu_batch_size}.")

        for batch_idx in range(num_batches):
            if batch_idx % max(1, num_batches // 20) == 0 and num_batches > 1: 
                logger.debug(f"  CPU matmul progress: {batch_idx}/{num_batches} batches ({batch_idx/num_batches*100:.1f}%)")
            
            start_idx = batch_idx * effective_cpu_batch_size
            end_idx = min(start_idx + effective_cpu_batch_size, a_rows)
            batch_a_cpu = matrix_a[start_idx:end_idx]
            
            try:
                # This matmul is part of the computation graph, do NOT use torch.no_grad() here
                batch_result_cpu = torch.matmul(batch_a_cpu, matrix_b) # matrix_b is already on CPU
                result_on_cpu[start_idx:end_idx] = batch_result_cpu
                del batch_result_cpu, batch_a_cpu
            except RuntimeError as e_batch:
                logger.error(f"Error in CPU matmul batch {batch_idx} for rows {start_idx}-{end_idx}: {e_batch}. Trying row-by-row for this batch.")
                for i in range(start_idx, end_idx):
                    single_row_a_cpu = matrix_a[i:i+1]
                    try:
                        row_result_cpu = torch.matmul(single_row_a_cpu, matrix_b)
                        result_on_cpu[i:i+1] = row_result_cpu
                        del row_result_cpu, single_row_a_cpu
                    except RuntimeError as e_row:
                        logger.error(f"CRITICAL: Row-by-row CPU matmul failed at row {i}: {e_row}. Cannot proceed with this matmul.")
                        # This is a hard stop for the matmul, as even row-by-row failed.
                        raise RuntimeError(f"Row-by-row CPU matmul failed at row {i}: {e_row}") from e_row
            if batch_idx % 10 == 0: # More frequent GC for CPU intensive loop
                 gc.collect() 

        gc.collect()
        logger.info(f"CPU Batched matmul completed. Total time so far: {time.time() - start_time:.2f}s")

        if original_device.type == 'cuda':
            logger.info(f"Attempting to move CPU batched matmul result ({result_on_cpu.shape}) back to {original_device}")
            try:
                # Consider chunked transfer if result_on_cpu is huge, but try direct first
                result = result_on_cpu.to(original_device)
                del result_on_cpu 
                gc.collect()
                logger.info(f"Successfully moved result back to {original_device}. Total matmul time: {time.time() - start_time:.2f}s")
                return result
            except RuntimeError as e_move_back:
                err_msg = f"CRITICAL: Failed to move CPU batched matmul result back to {original_device} from CPU. Result shape: {result_on_cpu.shape}. Error: {e_move_back}"
                logger.error(err_msg)
                # This is a hard failure for GPU mode if the result cannot be moved back.
                raise RuntimeError(err_msg) from e_move_back
        else:
            # Original device was CPU, result is already on CPU
            logger.debug(f"Original device was CPU. Returning result on CPU. Total matmul time: {time.time() - start_time:.2f}s")
            return result_on_cpu

    def manual_message_passing(self, graph, src_ntype, dst_ntype, src_features):
        """
        Manually implement message passing without relying on DGL's update_all.
        
        Args:
            graph: DGL graph
            src_ntype: Source node type
            dst_ntype: Destination node type
            src_features: Features of source nodes
            
        Returns:
            Aggregated features for destination nodes
        """
        start_time = time.time()  # Start timing
        # Get the edge type
        etype = None
        for canonical_etype in graph.canonical_etypes:
            if canonical_etype[0] == src_ntype and canonical_etype[2] == dst_ntype:
                etype = canonical_etype[1]
                break
        
        if etype is None:
            logger.warning(f"No edge type found for {src_ntype}->{dst_ntype}")
            return None
        
        canonical_etype = (src_ntype, etype, dst_ntype)
        logger.info(f"Manual message passing for {canonical_etype} with {graph.num_edges(etype=canonical_etype)} edges")
        
        # Get all edges of this type
        edges = graph.edges(etype=canonical_etype)
        src_ids, dst_ids = edges[0], edges[1]
        
        # Execute on CPU for safety
        device = src_features.device
        cpu_device = torch.device('cpu')
        
        if device.type == 'cuda':
            logger.info(f"  Moving source features to CPU, shape: {src_features.shape}")
            src_features_cpu = src_features.to(cpu_device)
            src_ids_cpu = src_ids.to(cpu_device)
            dst_ids_cpu = dst_ids.to(cpu_device)
        else:
            src_features_cpu = src_features
            src_ids_cpu = src_ids
            dst_ids_cpu = dst_ids
        
        # Create result tensor on CPU
        out_dim = src_features_cpu.shape[1]
        dst_num_nodes = graph.num_nodes(dst_ntype)
        logger.info(f"  Creating result tensor for {dst_num_nodes} destination nodes with {out_dim} features")
        result = torch.zeros(dst_num_nodes, out_dim, device=cpu_device)
        
        # Create counters to compute mean
        dst_counts = torch.zeros(dst_num_nodes, device=cpu_device)
        
        # Process in batches to save memory
        batch_size = 10000
        num_edges = src_ids_cpu.shape[0]
        
        num_batches = (num_edges + batch_size - 1) // batch_size
        logger.info(f"  Processing {num_edges} edges in {num_batches} batches")
        
        for batch_idx in range(0, num_batches):
            if batch_idx % max(1, num_batches // 10) == 0:  # Log every ~10% progress
                logger.info(f"  Manual message passing progress: {batch_idx}/{num_batches} batches ({batch_idx/num_batches*100:.1f}%)")
                
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_edges)
            batch_src_ids = src_ids_cpu[start_idx:end_idx]
            batch_dst_ids = dst_ids_cpu[start_idx:end_idx]
            
            # Get features for source nodes in this batch
            batch_src_features = src_features_cpu[batch_src_ids]
            
            # Aggregate features for destination nodes
            for i, (src_id, dst_id) in enumerate(zip(batch_src_ids, batch_dst_ids)):
                result[dst_id] += batch_src_features[i]
                dst_counts[dst_id] += 1
        
        # Compute mean - avoid division by zero
        mask = dst_counts > 0
        n_active_dst = mask.sum().item()
        logger.info(f"  Computing mean for {n_active_dst} active destination nodes out of {dst_num_nodes}")
        result[mask] = result[mask] / dst_counts[mask].unsqueeze(1)
        
        # Move result back to original device
        if device.type == 'cuda':
            logger.info(f"  Moving result back to {device}")
            result = result.to(device)
        
        logger.info(f"  Manual message passing completed in {time.time() - start_time:.2f}s")
        return result

    def forward(self, g: dgl.DGLGraph, feat: Union[torch.Tensor, Dict[str, torch.Tensor]],
                global_node_type_order_runtime: Optional[List[str]] = None,
                global_num_nodes_per_type_runtime: Optional[List[int]] = None) -> torch.Tensor:
        
        start_time = time.time()  # Start timing for the entire forward pass
        
        # Determine the primary computation device from the input features
        # This device will be used for most operations unless explicitly stated otherwise.
        computation_device = feat.device if isinstance(feat, torch.Tensor) else next(iter(feat.values())).device
        logger.info(f"RGC Forward: Computation device determined as: {computation_device}")

        # Log basic information about the graph
        if hasattr(g, 'num_nodes'):
            total_nodes = g.num_nodes()
            logger.info(f"RGC Forward: Graph with {total_nodes} total nodes, {g.num_edges()} total edges. Graph device: {g.device}")
            if hasattr(g, 'ntypes'):
                for ntype in g.ntypes:
                    logger.info(f"  - {ntype}: {g.num_nodes(ntype)} nodes")
        
        # Log CUDA memory status if applicable
        if computation_device.type == 'cuda' and hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
            # Aggressively clear cache before starting operations
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info()
            logger.info(f"CUDA memory before RGC ops: {free_mem/1024**3:.2f} GB free / {total_mem/1024**3:.2f} GB total ({free_mem/total_mem*100:.1f}% free)")
            
        # If feat is a dictionary (standard DGL heterogeneous input)
        if isinstance(feat, dict):
            # This mode is typically handled by DGL's HeteroGraphConv, which calls a
            # simpler GraphConv-like layer for each relation. This custom RGC's
            # main path assumes a single concatenated feature tensor.
            # For now, this path remains simplified for a single relation context.
            if not g.ntypes: # Check if g.ntypes is empty
                logger.warning("RGC Forward (dict input): Graph has no node types. Returning empty dict.")
                return {}
            if len(g.ntypes) == 1: # Homogeneous block graph for one relation type
                ntype = g.ntypes[0]
                if ntype not in feat:
                    raise ValueError(f"Node type {ntype} from graph not in feat dictionary.")
                h = feat[ntype].to(computation_device) # Ensure features are on computation_device
                
                # Ensure graph 'g' is on the computation_device
                if g.device != computation_device:
                    g = g.to(computation_device)
                    logger.info(f"  Moved input graph 'g' to {computation_device} for dict-input path.")

                if not self.rel_names:
        if self.self_loop:
                        self_loop_weight = self.self_weight.to(computation_device)
                        out_h = self.batched_matmul(h, self_loop_weight)
                    else:
                        out_h = torch.zeros(h.shape[0], self.out_dim, device=computation_device)
                else:
                    # This block assumes self.rel_names is NOT empty.
                    current_rel_weight = self.get_rel_weight(0).to(computation_device) # This line is fine IF self.rel_names is not empty
                    
                    # Standard R-GCN message passing
                    g.srcdata['h_src'] = self.batched_matmul(h, current_rel_weight)
                    g.update_all(fn.copy_u('h_src', 'm'), fn.mean('m', 'h_agg'))
                    out_h = g.dstdata.pop('h_agg')

                    if self.self_loop:
                         self_loop_weight = self.self_weight.to(computation_device) # Ensure weight is on correct device
                         out_h = out_h + self.batched_matmul(h, self_loop_weight)
                
                if self.layer_norm_out: out_h = self.layer_norm_out(out_h)
                if self.activation: out_h = self.activation(out_h)
                out_h = self.dropout(out_h)
                logger.info(f"RGC Forward (dict input) completed. Time: {time.time() - start_time:.2f}s")
                return {ntype: out_h}
            else: 
                raise NotImplementedError("RelationalGraphConv with dict 'feat' for multi-ntype block graph not fully implemented. Expects single global tensor or single ntype graph.")

        # If feat is a single tensor (global concatenated features)
        active_node_type_order = global_node_type_order_runtime if global_node_type_order_runtime is not None else self.global_node_type_order
        active_num_nodes_per_type = global_num_nodes_per_type_runtime if global_num_nodes_per_type_runtime is not None else self.global_num_nodes_per_type_in_order

        if not (active_node_type_order and active_num_nodes_per_type):
            raise ValueError("RGC received single tensor 'feat' but node type order/counts are missing.")

        if not active_num_nodes_per_type:
             cumulative_nodes_runtime = torch.tensor([0], device=computation_device)
        else:
             cumulative_nodes_runtime = torch.cumsum(torch.tensor([0] + active_num_nodes_per_type[:-1], device=computation_device), dim=0)
        
        node_type_to_start_idx_runtime = {
            ntype: cumulative_nodes_runtime[i].item()
            for i, ntype in enumerate(active_node_type_order)
            if i < len(cumulative_nodes_runtime)
        }
        
        # Initialize aggregated_feat directly on the computation_device
        logger.info(f"Creating aggregated_feat tensor of shape {(feat.shape[0], self.out_dim)} on {computation_device} with dtype {feat.dtype}")
        aggregated_feat = torch.zeros(feat.shape[0], self.out_dim, device=computation_device, dtype=feat.dtype)
        
        # Ensure the main graph 'g' is on the computation_device
        if g.device != computation_device:
            g = g.to(computation_device)
            logger.info(f"Moved main graph 'g' to {computation_device}")
            
        logger.info(f"Processing {len(self.rel_names)} relation types on {computation_device}")
        for rel_idx, rel_name in enumerate(self.rel_names):
            rel_start_time = time.time()
            logger.info(f"[{rel_idx+1}/{len(self.rel_names)}] Processing relation: {rel_name}")
            
            if computation_device.type == 'cuda':
                torch.cuda.empty_cache() # Clear cache per relation
            
            found_canonical_etype_for_rel = False
            current_canonical_etype = None
            for can_etype_tuple in g.canonical_etypes:
                if can_etype_tuple[1] == rel_name: # rel_name is the string like 'definition_of'
                    current_canonical_etype = can_etype_tuple
                    src_ntype, _, dst_ntype = current_canonical_etype
                    found_canonical_etype_for_rel = True
                    break
            
            if not found_canonical_etype_for_rel or g.num_edges(current_canonical_etype) == 0:
                logger.info(f"  Skipping relation {rel_name}: not found in graph canonical_etypes or zero edges for {current_canonical_etype}.")
                continue

            logger.info(f"  Found canonical edge type: {current_canonical_etype} with {g.num_edges(current_canonical_etype)} edges.")
            
            # Get relation weight, ensure it's on computation_device
            # Model parameters (self.weight_bases, self.weight_coefficients, self.direct_weights)
            # are expected to be on computation_device if model was moved correctly (e.g. model.to(device))
            current_rel_weight = self.get_rel_weight(rel_idx) 
            if current_rel_weight.device != computation_device:
                current_rel_weight = current_rel_weight.to(computation_device)
                logger.debug(f"  Moved current_rel_weight for {rel_name} to {computation_device}")
            logger.info(f"  Relation weight for {rel_name} shape: {current_rel_weight.shape}, device: {current_rel_weight.device}")
            
            # Create a subgraph for this specific canonical edge type.
            # This subgraph will inherit the device of 'g'.
            logger.info(f"  Creating edge subgraph for {current_canonical_etype} (source graph device: {g.device})")
            rel_graph = dgl.edge_type_subgraph(g, [current_canonical_etype])
            # Ensure rel_graph is on computation_device. It should be if 'g' is.
            if rel_graph.device != computation_device:
                 rel_graph = rel_graph.to(computation_device) # Should not happen if g is already moved
                 logger.warning(f"  Subgraph for {current_canonical_etype} was on {rel_graph.device}, moved to {computation_device}")
            logger.info(f"  Subgraph for {current_canonical_etype} created with {rel_graph.num_nodes()} total nodes, {rel_graph.num_edges()} edges on {rel_graph.device}")

            transformed_src_features_for_rel_graph = None
            if src_ntype in rel_graph.ntypes and src_ntype in node_type_to_start_idx_runtime:
                src_node_ids_in_rel_graph = rel_graph.nodes(src_ntype) # These are type-specific, 0-indexed IDs *within this ntype*
                
                if src_node_ids_in_rel_graph.shape[0] > 0:
                    start_idx_in_global_feat = node_type_to_start_idx_runtime[src_ntype]
                    
                    try:
                        src_ntype_idx_in_order = active_node_type_order.index(src_ntype)
                        num_nodes_of_src_ntype_global = active_num_nodes_per_type[src_ntype_idx_in_order]
                    except ValueError:
                        logger.error(f"src_ntype {src_ntype} not found in active_node_type_order. Skipping relation {rel_name}.")
                        continue
                    
                    # Get all features for this source node type from the global 'feat' tensor
                    # These are already on computation_device because 'feat' is.
                    all_feats_for_src_ntype = feat[start_idx_in_global_feat : start_idx_in_global_feat + num_nodes_of_src_ntype_global]
                    
                    # Select only the features for nodes present in the current rel_graph
                    # src_node_ids_in_rel_graph are the indices into all_feats_for_src_ntype
                    relevant_src_features_in_rel_graph_order = all_feats_for_src_ntype[src_node_ids_in_rel_graph]
                    
                    logger.info(f"  Transforming {relevant_src_features_in_rel_graph_order.shape[0]} '{src_ntype}' features for relation '{rel_name}' using weight on {current_rel_weight.device}")
                    # Perform matrix multiplication. batched_matmul handles device placement internally, prioritizing computation_device.
                    transformed_src_features_for_rel_graph = self.batched_matmul(relevant_src_features_in_rel_graph_order, current_rel_weight)
                    logger.info(f"    Transformed features shape: {transformed_src_features_for_rel_graph.shape}, device: {transformed_src_features_for_rel_graph.device}")

                    # Assign transformed features to the source nodes in the rel_graph
                    # Note: rel_graph.nodes(src_ntype) gives IDs local to that ntype, matching the order of transformed_src_features_for_rel_graph
                    # DGL expects features to be set for 'rel_graph.srcdata' using these local, type-specific IDs.
                    # However, DGL message passing APIs (like update_all) often work with features set for ALL nodes of a type.
                    # Let's ensure this assignment is what DGL expects for update_all using specific ntype source.
                    # We are setting features for *all* source nodes of type `src_ntype` *within the rel_graph*.
                    # The keys in `rel_graph.srcdata` will be `src_ntype`.
                    rel_graph.nodes[src_ntype].data['h_src'] = transformed_src_features_for_rel_graph
                    logger.debug(f"    Set 'h_src' for '{src_ntype}' nodes in rel_graph. Device: {transformed_src_features_for_rel_graph.device}")

            # Perform message passing on rel_graph
            # DGL's update_all should operate on the device of rel_graph and its features.
            # It needs source features for `src_ntype` and will aggregate into `dst_ntype`.
            if transformed_src_features_for_rel_graph is not None and dst_ntype in rel_graph.ntypes:
                logger.info(f"  Running DGL update_all for {current_canonical_etype} on {rel_graph.device}. Expecting msgs from '{src_ntype}' to '{dst_ntype}'.")
                # Define message and reduce functions for the specific canonical edge type
                # fn.copy_u takes ('src_node_feature_name', 'output_message_name')
                # fn.mean takes ('input_message_name', 'dst_node_feature_name')
                # We use (src_ntype, rel_name, dst_ntype) to specify the edge type for update_all
                try:
                    rel_graph.update_all(fn.copy_u('h_src', 'm'), 
                                         fn.mean('m', 'h_agg'))
                    
                    # Retrieve aggregated features for destination nodes in rel_graph
                    # These are 0-indexed IDs *within this ntype* in the subgraph
                    dst_node_ids_in_rel_graph = rel_graph.nodes(dst_ntype) 
                    rel_graph_dst_h_agg = rel_graph.nodes[dst_ntype].data.pop('h_agg')
                    logger.info(f"    Aggregated features for '{dst_ntype}' shape: {rel_graph_dst_h_agg.shape}, device: {rel_graph_dst_h_agg.device}")

                    # Scatter aggregated features back to the global aggregated_feat tensor
                    if dst_node_ids_in_rel_graph.shape[0] > 0 and dst_ntype in node_type_to_start_idx_runtime:
                        start_idx_global_dst = node_type_to_start_idx_runtime[dst_ntype]
                        
                        # We need global indices for destination nodes to update aggregated_feat
                        # rel_graph.nodes[dst_ntype].data['_ID'] should give original global IDs if preserved
                        # Or, if `dgl.edge_type_subgraph` preserves original node IDs mapped to its new compacted IDs,
                        # we can use `rel_graph.ndata[dgl.NID]` for the dst_ntype.
                        # Let's test if NID is available and correctly maps.
                        # Simpler: `dst_node_ids_in_rel_graph` are type-specific indices for `dst_ntype`
                        # So, `start_idx_global_dst + dst_node_ids_in_rel_graph` gives global indices for these nodes.
                        
                        global_indices_for_dst_nodes = start_idx_global_dst + dst_node_ids_in_rel_graph
                        
                        # Ensure aggregated_feat is on the same device before scatter-adding
                        if aggregated_feat.device != rel_graph_dst_h_agg.device:
                             # This should not happen if all ops are on computation_device
                             logger.warning(f"Device mismatch before scatter! aggregated_feat on {aggregated_feat.device}, rel_graph_dst_h_agg on {rel_graph_dst_h_agg.device}. Moving h_agg.")
                             rel_graph_dst_h_agg = rel_graph_dst_h_agg.to(aggregated_feat.device)

                        logger.debug(f"    Scattering {rel_graph_dst_h_agg.shape[0]} aggregated features for '{dst_ntype}' into aggregated_feat.")
                        # Ensure dtypes match before index_add_
                        if aggregated_feat.dtype != rel_graph_dst_h_agg.dtype:
                            logger.warning(f"Dtype mismatch before index_add_! aggregated_feat is {aggregated_feat.dtype}, rel_graph_dst_h_agg is {rel_graph_dst_h_agg.dtype}. Casting h_agg.")
                            rel_graph_dst_h_agg = rel_graph_dst_h_agg.to(aggregated_feat.dtype)
                        aggregated_feat.index_add_(0, global_indices_for_dst_nodes, rel_graph_dst_h_agg)
                    else:
                        logger.info(f"    No destination nodes of type '{dst_ntype}' in rel_graph or '{dst_ntype}' not in runtime mapping, or no aggregated features to scatter for {rel_name}.")

                except dgl.DGLError as e:
                    logger.error(f"  DGL Error during update_all for etype {current_canonical_etype} (rel: {rel_name}): {e}")
                    logger.error(f"    src_ntype: {src_ntype}, dst_ntype: {dst_ntype}")
                    if src_ntype in rel_graph.nodes:
                        logger.error(f"    rel_graph.nodes['{src_ntype}'].data keys: {rel_graph.nodes[src_ntype].data.keys()}")
                    else:
                        logger.error(f"    src_ntype {src_ntype} not in rel_graph.nodes")
                    # Potentially skip this relation or re-raise depending on severity
                    # For now, log and continue to allow other relations to process
                    pass # Continue to the next relation
            
            else: # No transformed_src_features or dst_ntype not in rel_graph
                if transformed_src_features_for_rel_graph is None:
                    logger.info(f"  No transformed source features for relation {rel_name} (e.g. no src nodes of type {src_ntype} in subgraph). Skipping message passing for this relation.")
                if dst_ntype not in rel_graph.ntypes:
                     logger.info(f"  Destination node type {dst_ntype} not in subgraph for relation {rel_name}. Skipping message passing.")


            # Clean up subgraph data to free memory
            if src_ntype in rel_graph.ntypes and 'h_src' in rel_graph.nodes[src_ntype].data:
                del rel_graph.nodes[src_ntype].data['h_src']
            if hasattr(rel_graph, 'ndata') and 'h_src' in rel_graph.ndata: # For older DGL or homogeneous case
                del rel_graph.ndata['h_src']
            
            # Explicitly clean h_agg for dst_ntype if it exists (it should have been popped, but as a safeguard)
            if dst_ntype in rel_graph.ntypes and 'h_agg' in rel_graph.nodes[dst_ntype].data:
                del rel_graph.nodes[dst_ntype].data['h_agg']
            if hasattr(rel_graph, 'ndata') and 'h_agg' in rel_graph.ndata:
                del rel_graph.ndata['h_agg']

            del rel_graph # Explicitly delete subgraph
            if computation_device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"  Processed relation {rel_name}. Time: {time.time() - rel_start_time:.2f}s")
        
        # Apply self-loop if enabled
        if self.self_loop:
            loop_start_time = time.time()
            logger.info("Applying self-loop.")
            # feat is the original input to this RGC layer (h_in for this layer)
            self_loop_transformed_feat = self.batched_matmul(feat, self.self_weight)
            logger.info(f"  Self-loop transformed_feat shape: {self_loop_transformed_feat.shape}, device: {self_loop_transformed_feat.device}")

            # Explicitly cast to the dtype of the accumulator to prevent AMP warnings for self-loop addition
            if self_loop_transformed_feat.dtype != aggregated_feat.dtype:
                logger.warning(f"Dtype mismatch for self-loop add! aggregated_feat is {aggregated_feat.dtype}, self_loop_transformed_feat is {self_loop_transformed_feat.dtype}. Casting self_loop_transformed_feat.")
                self_loop_transformed_feat = self_loop_transformed_feat.to(aggregated_feat.dtype)
            
            aggregated_feat += self_loop_transformed_feat # In-place addition
            # Corrected timing reference if you want time for self-loop specifically, or remove if overall RGC forward time is enough.
            # logger.info(f"Self-loop applied. Time: {time.time() - rel_start_time:.2f}s") # rel_start_time might be out of scope or not what's intended here.
            logger.info(f"Self-loop applied.") # Simplified log

        # Apply Layer Normalization, Activation, and Dropout
        if self.layer_norm_out:
            logger.info("Applying LayerNorm.")
            aggregated_feat = self.layer_norm_out(aggregated_feat)
        if self.activation:
            logger.info("Applying Activation.")
            aggregated_feat = self.activation(aggregated_feat)
        
        aggregated_feat = self.dropout(aggregated_feat) # Dropout always applied
        
        if computation_device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            try:
                # Use re to parse memory summary robustly
                summary_str = torch.cuda.memory_summary(device=computation_device, abbreviated=False)
                
                # Try to find patterns for allocated, reserved, and free memory using common English terms first
                allocated_match = re.search(r"Allocated By PyTorch:\s*([\d\.]+)\s*(\w+)", summary_str)
                reserved_match = re.search(r"Reserved By PyTorch:\s*([\d\.]+)\s*(\w+)", summary_str)
                total_capacity_match = re.search(r"Total Capacity:\s*([\d\.]+)\s*(\w+)", summary_str) # Often this is the total device memory
                free_in_reserved_match = re.search(r"Free Within Reserved Pool:\s*([\d\.]+)\s*(\w+)", summary_str)

                log_parts = []
                if allocated_match:
                    log_parts.append(f"Allocated: {allocated_match.group(1)} {allocated_match.group(2)}")
                if reserved_match:
                    log_parts.append(f"Reserved: {reserved_match.group(1)} {reserved_match.group(2)}")
                if total_capacity_match:
                    log_parts.append(f"Total Capacity: {total_capacity_match.group(1)} {total_capacity_match.group(2)}")
                if free_in_reserved_match:
                    log_parts.append(f"Free in Reserved: {free_in_reserved_match.group(1)} {free_in_reserved_match.group(2)}")
                
                if log_parts:
                    logger.info(f"CUDA memory after RGC ops: {'; '.join(log_parts)}")
                else:
                    # Fallback if specific parsing fails - log a snippet of the summary
                    logger.info(f"CUDA memory after RGC ops (unable to parse specific values, showing snippet):\n{summary_str[:500]}...")
            except Exception as e_mem_log:
                logger.warning(f"Could not log CUDA memory summary: {e_mem_log}")
        else:
            gc.collect()

        logger.info(f"RGC Forward (single tensor input) completed. Output shape: {aggregated_feat.shape}, device: {aggregated_feat.device}. Total time: {time.time() - start_time:.2f}s")
        return aggregated_feat

class SparseMultiHeadAttention(nn.Module):
    """
    Sparse Multi-Head Attention layer inspired by Exphormer.
    Implements efficient attention using sparse patterns.
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1,
                 sparsity: float = 0.9):
        """
        Initialize the Sparse Multi-Head Attention layer.
        
        Args:
            hidden_dim: Hidden feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            sparsity: Sparsity factor (0.0 = dense, 1.0 = no connections)
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sparsity = sparsity
        
        # Projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Create a virtual node for global information exchange
        self.virtual_token = nn.Parameter(torch.Tensor(1, hidden_dim))
        nn.init.normal_(self.virtual_token, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass computing sparse multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        original_device = x.device
        processing_device = original_device
        batch_size, seq_len, _ = x.shape

        # Heuristic: if input is large and on CUDA, offload to CPU
        # seq_len corresponds to total number of nodes when batch_size is 1 (GraphGPSLayer usage)
        # Threshold for 'large' seq_len can be tuned.
        # Also consider total elements: batch_size * seq_len
        is_large_input_on_cuda = (original_device.type == 'cuda' and 
                                  (seq_len > 80000 or (batch_size * seq_len > 160000)) and 
                                  x.nelement() * x.element_size() > 2 * 1024 * 1024 * 1024) # Input > 2GB

        if is_large_input_on_cuda:
            logger.info(f"SparseMultiHeadAttention: Large input detected on {original_device} (shape {x.shape}). Offloading to CPU.")
            processing_device = torch.device('cpu')
            x_processed = x.to(processing_device)
            virtual_token_processed = self.virtual_token.to(processing_device)
        else:
            x_processed = x
            virtual_token_processed = self.virtual_token
        
        # Append virtual token to the sequence
        virtual_tokens_expanded = virtual_token_processed.expand(batch_size, 1, self.hidden_dim)
        x_with_virtual = torch.cat([x_processed, virtual_tokens_expanded], dim=1)
        
        # Project queries, keys, and values
        q_proj_w = self.q_proj.weight.to(processing_device)
        q_proj_b = self.q_proj.bias.to(processing_device) if self.q_proj.bias is not None else None
        q = F.linear(x_with_virtual, q_proj_w, q_proj_b).view(batch_size, seq_len + 1, self.num_heads, self.head_dim)
        del q_proj_w, q_proj_b

        k_proj_w = self.k_proj.weight.to(processing_device)
        k_proj_b = self.k_proj.bias.to(processing_device) if self.k_proj.bias is not None else None
        k = F.linear(x_with_virtual, k_proj_w, k_proj_b).view(batch_size, seq_len + 1, self.num_heads, self.head_dim)
        del k_proj_w, k_proj_b

        v_proj_w = self.v_proj.weight.to(processing_device)
        v_proj_b = self.v_proj.bias.to(processing_device) if self.v_proj.bias is not None else None
        v = F.linear(x_with_virtual, v_proj_w, v_proj_b).view(batch_size, seq_len + 1, self.num_heads, self.head_dim)
        del v_proj_w, v_proj_b
        
        del x_with_virtual # Free intermediate tensor
        if processing_device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        
        # Reshape for multi-head attention
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len+1, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len+1, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len+1, head_dim]
        
        # Generate sparse attention pattern
        attention_mask = None # Default to None for dense or no mask needed cases
        if self.sparsity > 0:
            # Every node attends to the virtual token
            virtual_mask = torch.zeros(batch_size, self.num_heads, seq_len + 1, 1, device=processing_device)
            
            # Virtual token attends to all nodes
            global_mask = torch.zeros(batch_size, self.num_heads, 1, seq_len + 1, device=processing_device)
            
            # Generate random sparse connections for regular nodes
            sparse_mask_rand = torch.rand(batch_size, self.num_heads, seq_len, seq_len, device=processing_device)
            sparse_mask_bool = sparse_mask_rand > self.sparsity  # Keep only (1-sparsity) fraction of connections
            del sparse_mask_rand
            
            # Ensure each node has at least some connections (prevent disconnected nodes)
            min_connections = max(2, int((1 - self.sparsity) * seq_len * 0.1)) # Reduced min_connections for very sparse cases
            min_connections = min(min_connections, seq_len) # Cannot exceed seq_len
            if min_connections > 0:
                for i_node in range(seq_len):
                    if torch.sum(sparse_mask_bool[:, :, i_node, :]) < min_connections:
                        # Select random connections more carefully if seq_len is small
                        if seq_len > 0:
                            rand_indices = torch.randperm(seq_len, device=processing_device)[:min_connections]
                            sparse_mask_bool[:, :, i_node, rand_indices] = True
            
            # Combine masks
            # Full attention mask
            full_mask = torch.zeros(batch_size, self.num_heads, seq_len + 1, seq_len + 1, device=processing_device, dtype=torch.bool)
            full_mask[:, :, :seq_len, :seq_len] = sparse_mask_bool
            full_mask[:, :, :seq_len, seq_len] = True # All nodes attend to virtual token
            full_mask[:, :, seq_len, :seq_len+1] = True # Virtual token attends to all (including itself)
            
            attention_mask = full_mask # This is a boolean mask now
            del sparse_mask_bool, virtual_mask, global_mask
        
        # Compute scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(2, 3)) / scale
        del q, k # Free q, k after matmul
        if processing_device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        
        if attention_mask is not None:
            # scores are [batch, n_head, seq+1, seq+1], attention_mask is also [batch, n_head, seq+1, seq+1]
            scores = scores.masked_fill(~attention_mask, float('-inf')) # Use boolean not operator
            del attention_mask
        
        if mask is not None: # This is the padding mask from input
            # Expand mask for broadcasting: mask [B, S+1] -> [B, 1, 1, S+1] (for key padding)
            expanded_mask = mask.unsqueeze(1).unsqueeze(2).to(processing_device) # mask for keys
            scores = scores.masked_fill(~expanded_mask, float('-inf')) # Invert mask for masked_fill
            del expanded_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        del scores # Free scores
        if processing_device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)
        del attn_weights, v # Free attn_weights, v
        if processing_device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        
        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len + 1, self.hidden_dim)
        
        # Extract only the original sequence (without virtual token)
        context = context[:, :seq_len, :]
        
        # Final projection
        o_proj_w = self.o_proj.weight.to(processing_device)
        o_proj_b = self.o_proj.bias.to(processing_device) if self.o_proj.bias is not None else None
        output = F.linear(context, o_proj_w, o_proj_b)
        del o_proj_w, o_proj_b, context

        if processing_device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()

        if output.device != original_device:
            logger.info(f"SparseMultiHeadAttention: Moving output from {output.device} back to {original_device}.")
            output = output.to(original_device)
            if original_device.type == 'cuda': torch.cuda.empty_cache()
        
        return output

class GraphGPSLayer(nn.Module):
    """
    Graph GPS-style layer combining local message passing with global attention.
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 rel_names: List[str],
                 global_node_type_order: Optional[List[str]] = None,
                 global_num_nodes_per_type_in_order: Optional[List[int]] = None,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_bases: int = 8,
                 sparsity: float = 0.9,
                 layer_norm: bool = True,
                 use_global_attention: bool = False):
        """
        Initialize the GraphGPS layer.
        
        Args:
            hidden_dim: Hidden feature dimension
            rel_names: List of relation types in the graph
            global_node_type_order: List of node types in the graph
            global_num_nodes_per_type_in_order: List of number of nodes per type in the graph
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_bases: Number of bases for weight decomposition
            sparsity: Sparsity factor for attention
            layer_norm: Whether to apply layer normalization
            use_global_attention: Whether to use the global attention mechanism
        """
        super().__init__()
        
        self.use_global_attention = use_global_attention # Store parameter
        
        # Local message passing
        self.local_mp = RelationalGraphConv(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            rel_names=rel_names,
            global_node_type_order=global_node_type_order,
            global_num_nodes_per_type_in_order=global_num_nodes_per_type_in_order,
            num_bases=num_bases,
            activation=None,
            self_loop=True,
            dropout=dropout,
            layer_norm=False
        )
        
        # Global attention - conditionally initialize
        if self.use_global_attention:
        self.global_attn = SparseMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            sparsity=sparsity
        )
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.global_attn = None
            self.gate = None
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            if self.use_global_attention:
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            else:
                self.layer_norm2 = None # Explicitly None if not used
            self.layer_norm3 = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm1 = None
            self.layer_norm2 = None
            self.layer_norm3 = None
        
        # Gating mechanism to combine local and global features - moved to conditional init for self.gate
        # self.gate = nn.Linear(hidden_dim * 2, hidden_dim) # Already handled
    
    def forward(self, g, x: torch.Tensor, 
                global_node_type_order: Optional[List[str]] = None,
                global_num_nodes_per_type_in_order: Optional[List[int]] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the GraphGPS layer.
        
        Args:
            g: DGL heterogeneous graph
            x: Node feature tensor
            global_node_type_order: List of node types in the graph
            global_num_nodes_per_type_in_order: List of number of nodes per type in the graph
            mask: Optional attention mask
            
        Returns:
            Updated node features
        """
        layer_start_time = time.time() 
        original_input_device = x.device
        logger.info(f"GraphGPSLayer forward: Input tensor shape {x.shape}, device {original_input_device}")

        # --- Helper function for chunked LayerNorm ---
        def apply_layer_norm_chunked(ln_layer, input_tensor, chunk_size=10000, op_name="LayerNorm"):
            if ln_layer is None:
                return input_tensor
            if input_tensor.shape[0] == 0:
                return input_tensor # Or an empty tensor with correct shape if ln changes dim

            input_device = input_tensor.device # Device for all chunk processing
            logger.info(f"  Applying {op_name} in chunks on device {input_device}, input shape: {input_tensor.shape}")
            
            # Determine original device of ln_layer's parameters
            original_ln_layer_device = None
            if hasattr(ln_layer, 'weight') and ln_layer.weight is not None: # Check if it has parameters
                original_ln_layer_device = ln_layer.weight.device
            elif list(ln_layer.parameters()): # Fallback for general nn.Module
                original_ln_layer_device = next(ln_layer.parameters()).device


            # Move ln_layer to the device of input_tensor for computation
            ln_layer_compute = ln_layer.to(input_device)

            num_ln_chunks = (input_tensor.shape[0] + chunk_size - 1) // chunk_size
            output_ln_chunks = []
            for i_ln_chunk in range(num_ln_chunks):
                # logger.info(f"    {op_name} chunk {i_ln_chunk+1}/{num_ln_chunks}") # Optional verbose log
                start_idx = i_ln_chunk * chunk_size
                end_idx = min((i_ln_chunk + 1) * chunk_size, input_tensor.shape[0])
                chunk = input_tensor[start_idx:end_idx] # This chunk is on input_device
                
                # ln_layer_compute's parameters are now on input_device
                processed_chunk = ln_layer_compute(chunk)
                output_ln_chunks.append(processed_chunk)
                
                del chunk 
                if input_device.type == 'cuda': torch.cuda.empty_cache()
            
            result = torch.cat(output_ln_chunks, dim=0) # Result will be on input_device
            del output_ln_chunks
            
            # Move ln_layer back to its original device if it was moved and had an original device
            if original_ln_layer_device is not None and ln_layer_compute.weight.device != original_ln_layer_device:
                 # Ensure ln_layer (the original module passed in) is modified
                ln_layer.to(original_ln_layer_device)

            logger.info(f"  {op_name} application completed, output shape: {result.shape}, device: {result.device}")
            return result

        # 1. Local Message Passing
        local_out = self.local_mp(g, x,
                                 global_node_type_order_runtime=global_node_type_order,
                                 global_num_nodes_per_type_runtime=global_num_nodes_per_type_in_order)
        
        # Apply LayerNorm1 to local_out (chunked)
        local_out = apply_layer_norm_chunked(self.layer_norm1, local_out, op_name="LayerNorm1 (local_out)")
        
        # 2. Conditional Global Attention 
        if self.use_global_attention and self.global_attn is not None:
            # x (original input to GPS layer) is used for global_attn
            # The SparseMultiHeadAttention has its own internal CPU offloading if x is large.
            global_out = self.global_attn(x.unsqueeze(0), mask) # x is [N, D], unsqueeze to [1, N, D]
            global_out = global_out.squeeze(0)  # Remove batch dimension -> [N, D]
            
            # Apply LayerNorm2 to global_out (chunked)
            global_out = apply_layer_norm_chunked(self.layer_norm2, global_out, op_name="LayerNorm2 (global_out)")
            if torch.cuda.is_available() and global_out.device.type == 'cuda': torch.cuda.empty_cache()
            gc.collect() # After global_attn and its LayerNorm2
            
            # 3. Combine local and global features using a gating mechanism (chunked)
            if self.gate is not None:
                logger.info(f"  Combining local and global features with gating, local_out: {local_out.shape}, global_out: {global_out.shape}")
                chunk_size_gate = 5000 
                num_gate_chunks = (local_out.shape[0] + chunk_size_gate - 1) // chunk_size_gate
                combined_chunks_list = []
                
                # Determine primary device for gating operations
                # If either local_out or global_out is on CPU, prefer CPU for cat to avoid GPU OOM
                # If both are CUDA, try CUDA, but with offload if cat is too big
                gating_primary_device = local_out.device
                if local_out.device.type == 'cpu' or global_out.device.type == 'cpu':
                    gating_primary_device = torch.device('cpu')
                    logger.info(f"    Gating: At least one input (local/global) on CPU. Will use CPU for gating ops.")
                
                for i_gate_chunk in range(num_gate_chunks):
                    start_idx = i_gate_chunk * chunk_size_gate
                    end_idx = min((i_gate_chunk + 1) * chunk_size_gate, local_out.shape[0])

                    local_chunk = local_out[start_idx:end_idx]
                    global_chunk = global_out[start_idx:end_idx]
                    
                    comp_device_gate = gating_primary_device
                    # Move chunks to the computation device for gating
                    local_chunk_comp = local_chunk.to(comp_device_gate)
                    global_chunk_comp = global_chunk.to(comp_device_gate)

                    # Estimate size of concatenated tensor for this chunk
                    # Concatenated dim will be local_chunk_comp.shape[-1] + global_chunk_comp.shape[-1]
                    cat_dim_size = local_chunk_comp.shape[-1] + global_chunk_comp.shape[-1]
                    cat_elements = local_chunk_comp.shape[0] * cat_dim_size
                    cat_bytes = cat_elements * local_chunk_comp.element_size()

                    if comp_device_gate.type == 'cuda':
                        free_mem_gate_cat, _ = torch.cuda.mem_get_info(comp_device_gate)
                        # If cat is too large OR not enough memory for cat + buffer
                        if cat_bytes > (2 * 1024 * 1024 * 1024) or free_mem_gate_cat < (cat_bytes + 1 * 1024 * 1024 * 1024): # 2GB threshold, 1GB buffer
                            logger.info(f"    Gating chunk {i_gate_chunk}: Forcing concat to CPU. Cat_bytes: {cat_bytes/(1024**2):.2f}MB, Free_mem: {free_mem_gate_cat/(1024**2):.2f}MB")
                            comp_device_gate = torch.device('cpu')
                            local_chunk_comp = local_chunk.to(comp_device_gate) # Re-move if prev was CUDA
                            global_chunk_comp = global_chunk.to(comp_device_gate)

                    gate_input_chunk = torch.cat([local_chunk_comp, global_chunk_comp], dim=-1)
                    
                    # Apply gate linear layer
                    gate_weight_param = self.gate.weight.to(comp_device_gate)
                    gate_bias_param = self.gate.bias.to(comp_device_gate) if self.gate.bias is not None else None
                    gate_output_chunk = F.linear(gate_input_chunk, gate_weight_param, gate_bias_param)
                    gate_values_chunk = torch.sigmoid(gate_output_chunk)
                    
                    # Combine: result should be on comp_device_gate
                    combined_chunk_processed = gate_values_chunk * local_chunk_comp + (1 - gate_values_chunk) * global_chunk_comp
                    
                    # The chunk should remain on comp_device_gate for now
                    combined_chunks_list.append(combined_chunk_processed)

                    del local_chunk, global_chunk, local_chunk_comp, global_chunk_comp
                    del gate_input_chunk, gate_weight_param, gate_bias_param, gate_output_chunk, gate_values_chunk
                    if comp_device_gate.type == 'cuda': torch.cuda.empty_cache()
                    gc.collect() # Inside gating chunk loop
                    if gating_primary_device.type == 'cuda' and comp_device_gate.type == 'cpu' and original_input_device.type == 'cuda':
                        # If original was CUDA but this chunk was forced to CPU, mark for overall CPU concat for `combined`
                        # This logic is simplified: if any gating chunk goes to CPU and original was CUDA, final combined will be on CPU then moved.
                        pass # This is handled by where combined_chunks_list elements reside

                # Concatenate all processed combined_chunks
                # Determine device for this final concatenation of 'combined' parts
                final_combined_cat_device = original_input_device # Default to original input device
                # If any chunk in combined_chunks_list is on CPU AND original_input_device was CUDA, concat on CPU
                if original_input_device.type == 'cuda' and any(c.device.type == 'cpu' for c in combined_chunks_list):
                    final_combined_cat_device = torch.device('cpu')
                    logger.info(f"    Gating: Final concatenation of combined chunks will be on CPU.")

                materialized_combined_for_cat = []
                for ch_idx, ch_tensor in enumerate(combined_chunks_list):
                    if ch_tensor.device != final_combined_cat_device:
                        materialized_combined_for_cat.append(ch_tensor.to(final_combined_cat_device))
                    else:
                        materialized_combined_for_cat.append(ch_tensor)
                
                if materialized_combined_for_cat:
                    combined = torch.cat(materialized_combined_for_cat, dim=0)
                elif local_out.shape[0] == 0: # Handle empty input case for 'combined'
                     combined = torch.empty_like(local_out) # Ensure correct shape and device for empty
                else: # Should not be reached if input was not empty
                    logger.error("    Gating: materialized_combined_for_cat is empty unexpectedly. Fallback.")
                    combined = local_out # Fallback, though implies an issue

                del combined_chunks_list, materialized_combined_for_cat
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect() # After concatenating combined_chunks_list

                # If 'combined' is now on a different device than original_input_device (e.g. CPU),
                # it will be moved back after FFN if FFN also ends up on CPU, or handled by FFN's logic.
                # For now, 'combined' might be on CPU if offloading occurred.
                logger.info(f"  Gating completed, combined tensor shape: {combined.shape}, device: {combined.device}")

            else: # No gate defined, but use_global_attention was true. Should not happen with proper init.
                logger.warning("  Global attention used but self.gate is None. Using local_out as combined.")
                combined = local_out 
        else:
            # If global attention is not used, 'combined' is just 'local_out'
            combined = local_out
        
        # 4. Feed-Forward Network (FFN)
        # The FFN block (`self.ffn`) has its own comprehensive chunking and CPU offloading logic
        # It will try to keep its final output `ffn_out` on `combined.device` if possible,
        # or on CPU if `combined` was on CPU or FFN internal offloading forced it.
        
        # `combined` is the input to FFN. Its device dictates FFN's target input device.
        ffn_out = self.ffn_block_forward(combined) # Renamed internal FFN call
        if torch.cuda.is_available() and ffn_out.device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect() # After ffn_block_forward returns

        # 5. Final Residual Connection (ffn_out + combined)
        logger.info(f"  Applying final residual connection. ffn_out: {ffn_out.shape} ({ffn_out.device}), combined: {combined.shape} ({combined.device})")
        
        # Determine the target device for the sum, ideally the layer's original input device.
        target_sum_device = original_input_device
        
        # Determine the actual computation device for the sum, defaulting to CPU if CUDA checks fail.
        actual_sum_comp_device = torch.device('cpu') # Default to CPU for safety

        if target_sum_device.type == 'cuda':
            # Check if there's enough memory on CUDA for both operands and the result.
            mem_needed_ffn_out_on_cuda = 0
            if ffn_out.device.type == 'cpu':
                mem_needed_ffn_out_on_cuda = ffn_out.nelement() * ffn_out.element_size()
            
            mem_needed_combined_on_cuda = 0
            if combined.device.type == 'cpu':
                mem_needed_combined_on_cuda = combined.nelement() * combined.element_size()
            
            mem_needed_for_result = ffn_out.nelement() * ffn_out.element_size()
            
            free_cuda_mem, _ = torch.cuda.mem_get_info(target_sum_device)
            projected_free_mem_after_moves = free_cuda_mem - mem_needed_ffn_out_on_cuda - mem_needed_combined_on_cuda
            
            if projected_free_mem_after_moves > (mem_needed_for_result + 1 * 1024 * 1024 * 1024): # MODIFIED: 1GB buffer
                actual_sum_comp_device = target_sum_device # Sufficient memory to perform sum on CUDA
                logger.info(f"  Final residual: Performing sum on CUDA ({target_sum_device}). Projected free after moves: {projected_free_mem_after_moves/(1024**2):.2f}MB, Result: {mem_needed_for_result/(1024**2):.2f}MB")
            else:
                logger.info(f"  Final residual: Insufficient CUDA memory for sum. Forcing to CPU. Projected free after moves: {projected_free_mem_after_moves/(1024**2):.2f}MB, Result needed: {mem_needed_for_result/(1024**2):.2f}MB")
                # actual_sum_comp_device remains CPU (default)
        else: # target_sum_device is CPU
            actual_sum_comp_device = torch.device('cpu')
            logger.info(f"  Final residual: Target device is CPU. Performing sum on CPU.")

        # Move operands to actual_sum_comp_device and perform sum
        ffn_out_comp = ffn_out.to(actual_sum_comp_device)
        combined_comp = combined.to(actual_sum_comp_device)
        
        result_after_residual = ffn_out_comp + combined_comp
        logger.info(f"  Final residual sum performed on {actual_sum_comp_device}. Result shape {result_after_residual.shape}, device {result_after_residual.device}")

        # Clean up intermediate tensors used for the sum
        # If ffn_out_comp is not the same tensor as ffn_out (i.e., a copy was made)
        if ffn_out_comp is not ffn_out: del ffn_out_comp
        if combined_comp is not combined: del combined_comp
        # Delete original ffn_out and combined if they are not needed anymore and are on different devices than the result
        if ffn_out is not result_after_residual and ffn_out.device != result_after_residual.device: del ffn_out
        if combined is not result_after_residual and combined.device != result_after_residual.device: del combined
        
        if actual_sum_comp_device.type == 'cuda': torch.cuda.empty_cache()
        
        # result_after_residual is now on actual_sum_comp_device
        
        # 6. Final LayerNorm3 (applied to result_after_residual)
        # apply_layer_norm_chunked will operate on result_after_residual on its current device (actual_sum_comp_device)
        final_output = apply_layer_norm_chunked(self.layer_norm3, result_after_residual, op_name="LayerNorm3 (final_output)")
        
        # If result_after_residual was processed into final_output and they are different tensors,
        # and result_after_residual is not on the same device as where final_output might end up (original_input_device),
        # it can be deleted. This is tricky because apply_layer_norm_chunked might return input if layer is None.
        if final_output is not result_after_residual : del result_after_residual
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect() # After LayerNorm3

        # Ensure final_output is on original_input_device before returning
        if final_output.device != original_input_device:
            logger.info(f"  Moving final GraphGPSLayer output from {final_output.device} to {original_input_device} after LayerNorm3.")
            # This is the last major allocation/move for this layer's output.
            # If this fails, it means the final output tensor cannot fit on the target device.
            final_output = final_output.to(original_input_device) 
            if original_input_device.type == 'cuda': torch.cuda.empty_cache()
        
        logger.info(f"GraphGPSLayer forward completed in {time.time() - layer_start_time:.2f}s. Final output shape {final_output.shape}, device {final_output.device}")
        return final_output

    def ffn_block_forward(self, ffn_input_tensor: torch.Tensor) -> torch.Tensor:
        # This is the FFN processing logic, previously part of the main forward
        # ffn_input_tensor is 'combined' from the main forward method
        ffn_block_start_time = time.time()
        logger.info(f"  FFN Block internal: Input shape {ffn_input_tensor.shape}, device {ffn_input_tensor.device}")

        ffn_input_device_for_block = ffn_input_tensor.device
        if ffn_input_tensor.shape[0] == 0:
            out_features_dim = self.ffn[2].out_features if len(self.ffn) > 2 and hasattr(self.ffn[2], 'out_features') else self.hidden_dim
            ffn_result = torch.empty((0, out_features_dim), device=ffn_input_device_for_block, dtype=ffn_input_tensor.dtype)
            logger.info(f"  FFN Block: Input is empty. Outputting empty tensor of shape {ffn_result.shape}.")
            return ffn_result

        chunk_size = 5000
        num_chunks = (ffn_input_tensor.shape[0] + chunk_size - 1) // chunk_size
        output_chunks_ffn_block = []
        # Flag to indicate if any chunk processing for FFN was forced to CPU when input was CUDA
        concat_ffn_block_on_cpu = False 

        for i in range(num_chunks):
            if i % max(1, num_chunks // 10) == 0:
                logger.info(f"    FFN Block chunk {i+1}/{num_chunks}")
            
            chunk_input_ffn = ffn_input_tensor[i*chunk_size:(i+1)*chunk_size]
            current_op_input_ffn = chunk_input_ffn

            # --- First Linear Layer (self.ffn[0]) ---
            lin_layer0 = self.ffn[0]
            comp_device_lin0_ffn = ffn_input_device_for_block
            if ffn_input_device_for_block.type == 'cuda':
                free_mem_lin0, _ = torch.cuda.mem_get_info(ffn_input_device_for_block)
                output_elements_lin0 = current_op_input_ffn.shape[0] * lin_layer0.out_features
                output_size_bytes_lin0 = output_elements_lin0 * current_op_input_ffn.element_size()
                if output_size_bytes_lin0 > (2 * 1024 * 1024 * 1024) or free_mem_lin0 < (output_size_bytes_lin0 + 1 * 1024 * 1024 * 1024): # 2GB threshold, 1GB buffer
                    logger.info(f"    FFN Block Lin0: Forcing to CPU. Input: {current_op_input_ffn.shape}, Output_bytes: {output_size_bytes_lin0/(1024**2):.2f}MB, Free_mem: {free_mem_lin0/(1024**2):.2f}MB")
                    comp_device_lin0_ffn = torch.device('cpu')
            
            input_for_lin0_ffn = current_op_input_ffn.to(comp_device_lin0_ffn)
            weight_lin0_ffn = lin_layer0.weight.to(comp_device_lin0_ffn)
            bias_lin0_ffn = lin_layer0.bias.to(comp_device_lin0_ffn) if lin_layer0.bias is not None else None
            current_op_input_ffn = F.linear(input_for_lin0_ffn, weight_lin0_ffn, bias_lin0_ffn)
            del input_for_lin0_ffn, weight_lin0_ffn, bias_lin0_ffn
            if comp_device_lin0_ffn.type == 'cuda': torch.cuda.empty_cache()
            gc.collect() # After Lin0 FFN Block
            
            if ffn_input_device_for_block.type == 'cuda' and current_op_input_ffn.device.type == 'cpu':
                concat_ffn_block_on_cpu = True

            # --- GELU Activation (self.ffn[1]) ---
            current_op_input_ffn = self.ffn[1](current_op_input_ffn) 

            # --- Second Linear Layer (self.ffn[2]) ---
            lin_layer2 = self.ffn[2]
            comp_device_lin2_ffn = current_op_input_ffn.device
            if current_op_input_ffn.device.type == 'cuda':
                free_mem_lin2, _ = torch.cuda.mem_get_info(current_op_input_ffn.device)
                output_elements_lin2 = current_op_input_ffn.shape[0] * lin_layer2.out_features
                output_size_bytes_lin2 = output_elements_lin2 * current_op_input_ffn.element_size()
                if output_size_bytes_lin2 > (2 * 1024 * 1024 * 1024) or free_mem_lin2 < (output_size_bytes_lin2 + 1 * 1024 * 1024 * 1024): # 2GB threshold, 1GB buffer
                    logger.info(f"    FFN Block Lin2: Forcing to CPU. Input: {current_op_input_ffn.shape}, Output_bytes: {output_size_bytes_lin2/(1024**2):.2f}MB, Free_mem: {free_mem_lin2/(1024**2):.2f}MB")
                    comp_device_lin2_ffn = torch.device('cpu')
            elif current_op_input_ffn.device.type == 'cpu':
                comp_device_lin2_ffn = torch.device('cpu')

            input_for_lin2_ffn = current_op_input_ffn.to(comp_device_lin2_ffn)
            weight_lin2_ffn = lin_layer2.weight.to(comp_device_lin2_ffn)
            bias_lin2_ffn = lin_layer2.bias.to(comp_device_lin2_ffn) if lin_layer2.bias is not None else None
            current_op_input_ffn = F.linear(input_for_lin2_ffn, weight_lin2_ffn, bias_lin2_ffn)
            del input_for_lin2_ffn, weight_lin2_ffn, bias_lin2_ffn
            if comp_device_lin2_ffn.type == 'cuda': torch.cuda.empty_cache()
            gc.collect() # After Lin2 FFN Block

            if ffn_input_device_for_block.type == 'cuda' and current_op_input_ffn.device.type == 'cpu':
                concat_ffn_block_on_cpu = True
            
            # --- Dropout (self.ffn[3]) ---
            current_op_input_ffn = self.ffn[3](current_op_input_ffn)
            
            output_chunks_ffn_block.append(current_op_input_ffn)
            del chunk_input_ffn 
            if ffn_input_device_for_block.type == 'cuda': torch.cuda.empty_cache()

        # Concatenation logic for FFN block
        concat_target_device_ffn_block = ffn_input_device_for_block
        if concat_ffn_block_on_cpu:
            concat_target_device_ffn_block = torch.device('cpu')
            logger.info(f"  FFN Block: Concatenation will be performed on CPU.")
        
        materialized_chunks_for_cat_ffn = []
        if output_chunks_ffn_block:
            for chunk_tensor_cat_ffn in output_chunks_ffn_block:
                if chunk_tensor_cat_ffn.device != concat_target_device_ffn_block:
                    materialized_chunks_for_cat_ffn.append(chunk_tensor_cat_ffn.to(concat_target_device_ffn_block))
                else:
                    materialized_chunks_for_cat_ffn.append(chunk_tensor_cat_ffn)
            
            ffn_result = torch.cat(materialized_chunks_for_cat_ffn, dim=0)
            del materialized_chunks_for_cat_ffn
            output_chunks_ffn_block.clear()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect() # After FFN block concat

            # Ensure ffn_result is on ffn_input_device_for_block for consistent output from this block
            # MODIFICATION: Return ffn_result on its current device (concat_target_device_ffn_block).
            # The caller (GraphGPSLayer.forward) will handle further device placement for residual sum.
            # if ffn_result.device != ffn_input_device_for_block:
            #     logger.info(f"  FFN Block: Moving final FFN result from {ffn_result.device} to {ffn_input_device_for_block}")
            #     ffn_result = ffn_result.to(ffn_input_device_for_block)
            #     if torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info(f"  FFN Block: Result is on {ffn_result.device} (determined by concat_target_device_ffn_block) before returning.")

        else: # Should only happen if ffn_input_tensor was empty, handled at start
            logger.error("  FFN Block: output_chunks_ffn_block is empty unexpectedly. This path should not be reached if input was not empty.")
            # Fallback to create an empty tensor with correct shape based on FFN's output dimension
            out_features_dim_fb = self.ffn[2].out_features if len(self.ffn) > 2 and hasattr(self.ffn[2], 'out_features') else self.hidden_dim
            ffn_result = torch.empty((ffn_input_tensor.shape[0], out_features_dim_fb), device=ffn_input_device_for_block, dtype=ffn_input_tensor.dtype)


        logger.info(f"  FFN Block internal completed in {time.time() - ffn_block_start_time:.2f}s, output shape: {ffn_result.shape if hasattr(ffn_result, 'shape') else 'unknown'}")
        return ffn_result

class HeterogeneousGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network architecture for lexical knowledge graphs.
    
    Combines R-GCN message passing with Exphormer-style global attention
    using GraphGPS principles.
    """
    
    def __init__(self, 
                 original_feat_dims: Dict[str, int],
                 hidden_dim: int,
                 out_dim: int,
                 node_types: List[str],
                 rel_names: List[str],
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 residual: bool = True,
                 layer_norm: bool = True,
                 num_bases: int = 8,
                 sparsity: float = 0.9,
                 use_global_attention: bool = False):
        """
        Initialize the HeterogeneousGNN model.
        
        Args:
            original_feat_dims: Dictionary mapping node type to its original raw feature dimension.
            hidden_dim: The common feature dimension to which all node features are projected
                        and which is used by the internal GNN layers. MUST BE AN INTEGER.
            out_dim: The final output feature dimension for each node type from this GNN block. MUST BE AN INTEGER.
            node_types: List of all unique node type names in the graph.
            rel_names: List of relation types in the graph.
            num_layers: Number of GraphGPS layers.
            num_heads: Number of attention heads per GraphGPS layer.
            dropout: Dropout probability.
            residual: Whether GraphGPS layers use residual connections.
            layer_norm: Whether GraphGPS layers use layer normalization.
            num_bases: Number of bases for RelationalGraphConv within GraphGPS.
            sparsity: Sparsity factor for global attention within GraphGPS.
            use_global_attention: Whether to use the global attention mechanism in GraphGPS layers.
        """
        super().__init__()
        
        if not isinstance(hidden_dim, int):
            raise TypeError(
                f"HeterogeneousGNN 'hidden_dim' parameter must be an int, but got {type(hidden_dim)} (value: {hidden_dim}). "
                f"This usually means the calling code (e.g., in HGMAE.__init__ or pretrain_hgmae.py during HGMAE instantiation) "
                f"is incorrectly passing a dictionary of feature dimensions (like feat_dims) "
                f"instead of a single integer for the common GNN hidden layer size."
            )
        if not isinstance(out_dim, int):
            # Although not the cause of the current error, out_dim should also be an integer.
            raise TypeError(
                f"HeterogeneousGNN 'out_dim' parameter must be an int, but got {type(out_dim)} (value: {out_dim})."
            )
        
        self.original_feat_dims = original_feat_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.node_types_ordered = sorted(list(set(node_types)))
        self.rel_names = rel_names
        self.residual_hgnn = residual
        self.use_global_attention = use_global_attention # Store parameter

        # Input projections: one for each node type to project to common hidden_dim
        self.input_projs = nn.ModuleDict()
        for ntype in self.node_types_ordered:
            if ntype not in self.original_feat_dims:
                logger.warning(f"Node type '{ntype}' not found in original_feat_dims. Cannot create input projection. Features for this type must match hidden_dim ({self.hidden_dim}) if provided in forward pass, or will be zeros if not.")
                continue

            in_features_for_proj = self.original_feat_dims[ntype]

            if not isinstance(in_features_for_proj, int) or in_features_for_proj <= 0:
                logger.error(f"Invalid in_features_for_proj '{in_features_for_proj}' for ntype '{ntype}'. Defaulting to 1 for nn.Linear and attempting to proceed. Upstream feature dimension calculation needs review.")
                in_features_for_proj = 1 # Attempt to make it a valid integer for nn.Linear

            logger.info(f"Creating nn.Linear for input projection of '{ntype}' with input_dim={in_features_for_proj}, output_dim={self.hidden_dim}.")
            
            # Create the Linear layer
            linear_layer = nn.Linear(in_features_for_proj, self.hidden_dim) # Bias is True by default
            
            # Explicitly cast all parameters of the layer to float32
            # .float() casts all floating point parameters and buffers to float datatype.
            linear_layer.float()
            
            self.input_projs[ntype] = linear_layer
            
            # Log dtypes after creation and casting to confirm
            log_msg = f"  '{ntype}' input projection Linear layer created. Weight dtype: {self.input_projs[ntype].weight.dtype}"
            if self.input_projs[ntype].bias is not None:
                log_msg += f", Bias dtype: {self.input_projs[ntype].bias.dtype}"
            else:
                log_msg += ", Bias: None"
            logger.info(log_msg)
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                GraphGPSLayer(
                    hidden_dim=self.hidden_dim,
                    rel_names=self.rel_names,
                    global_node_type_order=self.node_types_ordered,
                    global_num_nodes_per_type_in_order=None,
                    num_heads=num_heads,
                    dropout=dropout,
                    num_bases=num_bases,
                    sparsity=sparsity,
                    layer_norm=layer_norm,
                    use_global_attention=self.use_global_attention # Pass to GraphGPSLayer
                )
            )
        
        # Output projections: one for each node type, from hidden_dim to out_dim
        self.output_projs = nn.ModuleDict()
        for ntype in self.node_types_ordered:
            self.output_projs[ntype] = nn.Linear(self.hidden_dim, self.out_dim)
        
        self.dropout_out = nn.Dropout(dropout)
    
    def forward(self, g: dgl.DGLGraph, feats_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the HeterogeneousGNN model.
        
        Args:
            g: DGL heterogeneous graph.
            feats_dict: Dictionary of node features {node_type: DGLFeatures}.
            
        Returns:
            out_dict: Dictionary of output node embeddings {node_type: Tensor}.
        """
        start_time = time.time()
        if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            logger.info(f"HeterogeneousGNN forward: CUDA memory: {free_mem/1024**3:.2f} GB free, {total_mem/1024**3:.2f} GB total ({free_mem/total_mem*100:.1f}% free)")
        
        logger.info(f"HeterogeneousGNN forward: Processing graph with {len(g.ntypes)} node types on device {g.device}")

        projection_start_time = time.time()
        logger.info("1. Applying input projections and preparing for concatenation")
        
        projected_feats_list = []
        num_nodes_per_type_in_order_actual = [] # For splitting later
        graph_input_device = g.device

        # --- Stage 1: Estimate concatenated size and determine concatenation device ---
        estimated_total_elements = 0
        # Use a common dtype for estimation, e.g., float32, or try to infer from first available feature
        example_dtype = torch.float32 
        has_any_feats = any(ntype in feats_dict and feats_dict[ntype] is not None for ntype in self.node_types_ordered)
        if has_any_feats:
            for ntype_est in self.node_types_ordered:
                if ntype_est in feats_dict and feats_dict[ntype_est] is not None:
                    example_dtype = feats_dict[ntype_est].dtype
                    break
        
        for ntype_est in self.node_types_ordered:
            num_nodes = g.num_nodes(ntype_est)
            feat_dim_after_proj = self.hidden_dim # Assume all project to hidden_dim
            if ntype_est in feats_dict and feats_dict[ntype_est] is not None and ntype_est not in self.input_projs:
                 # If no projection layer, it means existing feats are already hidden_dim (or error later)
                feat_dim_after_proj = feats_dict[ntype_est].shape[1]
            estimated_total_elements += num_nodes * feat_dim_after_proj

        element_size_bytes = torch.tensor([], dtype=example_dtype).element_size()
        estimated_concat_bytes = estimated_total_elements * element_size_bytes
        
        concat_processing_device = graph_input_device # Default to graph's device
        if graph_input_device.type == 'cuda':
            free_cuda_mem, _ = torch.cuda.mem_get_info(graph_input_device)
            # If estimated concat size is > 8GB OR (available_mem < est_size + 1GB buffer)
            # Use a large buffer for this initial, potentially massive, concatenation
            if estimated_concat_bytes > (8 * 1024 * 1024 * 1024) or \
               free_cuda_mem < (estimated_concat_bytes + 1 * 1024 * 1024 * 1024):
                logger.warning(
                    f"HeteroGNN: Initial feature concat (est. {estimated_concat_bytes / (1024**2):.2f}MB) "
                    f"is large for CUDA ({free_cuda_mem / (1024**2):.2f}MB free). Forcing projections and concat to CPU."
                )
                concat_processing_device = torch.device('cpu')
        logger.info(f"HeteroGNN: Projections and initial concatenation will occur on {concat_processing_device}.")

        # --- Stage 2: Perform projections and collect features on concat_processing_device ---
        for ntype in self.node_types_ordered:
            num_nodes = g.num_nodes(ntype)
            num_nodes_per_type_in_order_actual.append(num_nodes)

            if num_nodes == 0:
                projected_feats_list.append(torch.empty((0, self.hidden_dim), device=concat_processing_device, dtype=example_dtype))
                continue

            feat_val = feats_dict.get(ntype)
            projected_for_ntype = None # Initialize

            if ntype in self.input_projs:
                layer_to_use_for_proj = self.input_projs[ntype]
                original_device_of_layer = layer_to_use_for_proj.weight.device
                
                # Move layer for computation if necessary
                if original_device_of_layer != concat_processing_device:
                    layer_to_use_for_proj.to(concat_processing_device)
                
                # Perform projection using layer_to_use_for_proj (which is now on concat_processing_device)
                if feat_val is None:
                    current_feat_dtype = layer_to_use_for_proj.weight.dtype
                    projected_for_ntype = torch.zeros(num_nodes, self.hidden_dim, device=concat_processing_device, dtype=current_feat_dtype)
            else:
                    projected_for_ntype = layer_to_use_for_proj(feat_val.to(concat_processing_device))
                
                projected_feats_list.append(projected_for_ntype)
                
                # Move layer back to its original device if it was moved
                if original_device_of_layer != concat_processing_device:
                    # Ensure the original shared module self.input_projs[ntype] is restored
                    self.input_projs[ntype].to(original_device_of_layer)

            elif feat_val is not None:
                if feat_val.shape[1] != self.hidden_dim: # Corrected indentation
                    raise ValueError(
                        f"Feature dim for {ntype} ({feat_val.shape[1]}) doesn't match hidden_dim "
                        f"({self.hidden_dim}) and no input projection layer defined for this type."
                    )
                projected_feats_list.append(feat_val.to(concat_processing_device)) # Corrected indentation
            else: # No features, no projection layer defined
                projected_feats_list.append(torch.zeros(num_nodes, self.hidden_dim, device=concat_processing_device, dtype=example_dtype)) # Corrected indentation
        
        logger.info(f"1.1. Input projections processing complete. Features ready for concat on {concat_processing_device}. Time: {time.time() - projection_start_time:.2f}s")

        if not projected_feats_list:
            logger.warning("No features to process in HeterogeneousGNN.forward after projections.")
            return {}

        # 2. Concatenate to a single tensor for GraphGPS layers on concat_processing_device
        h_concatenated = torch.cat(projected_feats_list, dim=0)
        del projected_feats_list 
        if torch.cuda.is_available(): torch.cuda.empty_cache() 
        gc.collect() 
        
        logger.info(f"2. Concatenated features to single tensor: {h_concatenated.shape}, device: {h_concatenated.device}. Time: {time.time() - projection_start_time:.2f}s")
        
        # 3. Apply GraphGPS layers
        # GraphGPS layers are designed to handle input (h_concatenated) that might be on CPU
        # and will manage their own internal device placements and offloading.
        h_current_iter = h_concatenated
        for i, layer in enumerate(self.gnn_layers):
            layer_process_start_time = time.time()
            logger.info(f"  Starting GNN Layer {i+1}/{len(self.gnn_layers)} with input {h_current_iter.shape} on {h_current_iter.device}")
            h_prev_iter = h_current_iter
            h_current_iter = layer(g, h_prev_iter, 
                                 global_node_type_order=self.node_types_ordered,
                                 global_num_nodes_per_type_in_order=num_nodes_per_type_in_order_actual)
            
            logger.info(f"  Finished GNN Layer {i+1}/{len(self.gnn_layers)}. Output {h_current_iter.shape} on {h_current_iter.device}. Time: {time.time() - layer_process_start_time:.2f}s")
            if h_prev_iter is not h_current_iter and h_prev_iter.device != h_current_iter.device: # If device changed, old one might be fully freeable
                 del h_prev_iter
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 gc.collect()
            elif h_prev_iter is not h_current_iter: # Same device, but different tensor
                 del h_prev_iter 
                 # gc.collect might be too frequent here if all ops stay on same device
            # If h_prev_iter is h_current_iter (in-place), do nothing to h_prev_iter
        
        h_final_gnn = h_current_iter
        # If h_concatenated (initial input to layers) is different from final GNN output and not on same device, it might be safe to del
        if h_concatenated is not h_final_gnn and h_concatenated.device != h_final_gnn.device and 'h_concatenated' in locals():
             del h_concatenated
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             gc.collect() 
        elif 'h_concatenated' in locals() and h_concatenated is not h_final_gnn:
            del h_concatenated # Delete if different, even if on same device, as h_final_gnn is the one we need
            # gc.collect might be too frequent here

        logger.info(f"3. GraphGPS layers completed. Final GNN features: {h_final_gnn.shape}, device: {h_final_gnn.device}. Total GNN layer time: {time.time() - projection_start_time:.2f}s (since proj start)")

        # 4. Split the concatenated tensor back into a dictionary by node type
        # This operation should ideally happen on the device of h_final_gnn.
        # If h_final_gnn is on CPU, splitting is fine. If on CUDA, it's also fine.
        logger.info(f"4. Splitting GNN features ({h_final_gnn.device}) into dictionary.")
        split_start_time = time.time()
        h_dict = {}
        current_idx = 0
        for i_split, ntype_split in enumerate(self.node_types_ordered):
            num_nodes_for_split = num_nodes_per_type_in_order_actual[i_split]
            if num_nodes_for_split > 0:
                h_dict[ntype_split] = h_final_gnn[current_idx : current_idx + num_nodes_for_split]
            else: # Handle cases with zero nodes of a type, ensure dict entry exists if expected downstream
                h_dict[ntype_split] = torch.empty((0, self.hidden_dim), device=h_final_gnn.device, dtype=h_final_gnn.dtype)
            current_idx += num_nodes_for_split
        
        # It's possible h_final_gnn is no longer needed if all parts are successfully split and copied (if slicing creates views)
        # However, h_dict values are views. For safety, keep h_final_gnn until output projections.
        logger.info(f"Splitting completed in {time.time() - split_start_time:.2f}s")
            
        # 5. Apply output projections for each node type
        # These projections should bring features to the final `out_dim`
        # and ideally back to the original graph_input_device if they were processed on CPU.
        logger.info(f"5. Applying output projections. Target device for final output: {graph_input_device}")
        output_proj_start_time = time.time()
        out_dict = {}
        for ntype_out in self.node_types_ordered:
            if ntype_out in h_dict and ntype_out in self.output_projs:
                layer_to_use_for_out_proj = self.output_projs[ntype_out]
                original_device_of_out_layer = layer_to_use_for_out_proj.weight.device
                
                # Device for this projection operation is the device of its input
                op_device_for_out_proj = h_dict[ntype_out].device 
                if original_device_of_out_layer != op_device_for_out_proj:
                    layer_to_use_for_out_proj.to(op_device_for_out_proj)
                
                projected_out_chunk = layer_to_use_for_out_proj(h_dict[ntype_out]) # Input h_dict[ntype_out] is already on op_device
                projected_out_chunk = self.dropout_out(projected_out_chunk)
                
                out_dict[ntype_out] = projected_out_chunk.to(graph_input_device) # Move final result to graph_input_device

                # Move the original layer back if its original device was different from op_device_for_out_proj
                if original_device_of_out_layer != op_device_for_out_proj:
                    # Ensure the original shared module self.output_projs[ntype_out] is restored
                    self.output_projs[ntype_out].to(original_device_of_out_layer)

            elif ntype_out in h_dict: # Has processed embeddings but no specific output projection for out_dim
                                     # This implies hidden_dim might be the same as out_dim for these types
                if h_dict[ntype_out].shape[1] != self.out_dim:
                     logger.warning(f"Node type {ntype_out} has GNN embeddings (dim {h_dict[ntype_out].shape[1]}) "
                                    f"but no output projection to out_dim ({self.out_dim}). Using as is and moving to target device.")
                out_dict[ntype_out] = h_dict[ntype_out].to(graph_input_device)
        
        del h_dict, h_final_gnn # Delete intermediate dict and concatenated tensor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Output projections completed in {time.time() - output_proj_start_time:.2f}s. All outputs on {graph_input_device}.")
        logger.info(f"HeterogeneousGNN forward pass completed in {time.time() - start_time:.2f}s")
        return out_dict
    
    def encode(self, g: dgl.DGLGraph, feats_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode the graph into node embeddings (dictionary format).
        This is the primary method HGMAE will call on the encoder.
        """
        return self.forward(g, feats_dict) 