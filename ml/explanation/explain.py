import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl # type: ignore
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import shap # type: ignore
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class NodeClassificationModelWrapper(nn.Module):
    """
    Wraps a GNN encoder and a classification head for SHAP explanations.
    SHAP typically perturbs input features. For GNNs, this means perturbing
    features of target nodes while keeping graph structure and context node
    features fixed.
    """
    def __init__(self, 
                 gnn_encoder: nn.Module, 
                 classification_head: nn.Module, 
                 full_graph_structure: dgl.DGLGraph,
                 original_full_features: Dict[str, torch.Tensor],
                 target_node_type_for_explanation: str,
                 global_node_type_order: List[str],
                 global_num_nodes_per_type: List[int]):
        """
        Args:
            gnn_encoder: The GNN model component that outputs node embeddings.
            classification_head: The classification layer that takes GNN embeddings.
            full_graph_structure: The complete DGL graph.
            original_full_features: A dictionary {ntype: features_tensor} for ALL nodes in the graph.
                                     These are the base features that SHAP will perturb for target nodes.
            target_node_type_for_explanation: The node type of the nodes being explained.
            global_node_type_order: Ordered list of all node types in the graph.
            global_num_nodes_per_type: List of node counts for each type, in order.
        """
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.classification_head = classification_head
        self.full_graph_structure = full_graph_structure
        self.original_full_features = {
            k: v.clone() for k, v in original_full_features.items()
        } # Store a copy
        self.target_node_type_for_explanation = target_node_type_for_explanation
        
        # For HeterogeneousGNN's concatenated input format
        self.global_node_type_order = global_node_type_order
        self.global_num_nodes_per_type = global_num_nodes_per_type
        self.node_type_to_start_idx: Dict[str, int] = {}
        current_idx = 0
        for i, ntype in enumerate(self.global_node_type_order):
            self.node_type_to_start_idx[ntype] = current_idx
            current_idx += self.global_num_nodes_per_type[i]


    def forward(self, perturbed_target_node_features: torch.Tensor, 
                target_node_original_indices_in_type: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass using perturbed features for specified target nodes.

        Args:
            perturbed_target_node_features: Tensor of shape (num_target_nodes, feature_dim)
                                            containing perturbed features for the target nodes.
                                            These are the features SHAP modifies.
            target_node_original_indices_in_type: Tensor of shape (num_target_nodes,)
                                                  containing the original 0-based indices
                                                  of these target nodes *within their specific node type*
                                                  (self.target_node_type_for_explanation).
        Returns:
            Logits or probabilities from the classification_head for the target nodes.
            Shape: (num_target_nodes, num_classes)
        """
        device = perturbed_target_node_features.device
        
        # 1. Prepare the full feature dictionary for the GNN encoder input
        # Start with a deep copy of original features for all node types
        current_run_features = {
            ntype: feat.clone().to(device) for ntype, feat in self.original_full_features.items()
        }

        # 2. Substitute the perturbed features for the target nodes
        # Ensure target_node_original_indices_in_type are valid indices for the target ntype
        if self.target_node_type_for_explanation not in current_run_features:
            raise ValueError(f"Target node type {self.target_node_type_for_explanation} not in original features.")
        
        # Update the features for the target node type at the specified indices
        # This assumes perturbed_target_node_features are for self.target_node_type_for_explanation
        current_run_features[self.target_node_type_for_explanation][target_node_original_indices_in_type, :] = perturbed_target_node_features

        # 3. Run the GNN encoder
        # The gnn_encoder (HeterogeneousGNN) handles its own device placement and feature concatenation internally.
        # Ensure graph is on the same device as the model/features if not handled by GNN.
        # Most DGL models expect graph and features on the same device as model parameters.
        model_device = next(self.gnn_encoder.parameters()).device
        graph_for_run = self.full_graph_structure.to(model_device)
        features_for_gnn_run = {
            k: v.to(model_device) for k, v in current_run_features.items()
        }
        
        # The HeterogeneousGNN's forward pass returns a dictionary of embeddings.
        all_node_embeddings_dict = self.gnn_encoder(graph_for_run, features_for_gnn_run)

        # 4. Extract embeddings for the target nodes
        target_node_embeddings = all_node_embeddings_dict.get(self.target_node_type_for_explanation)
        if target_node_embeddings is None:
            raise ValueError(f"GNN encoder did not return embeddings for target type {self.target_node_type_for_explanation}")

        # Select the embeddings corresponding to target_node_original_indices_in_type
        # These indices are for the original feature matrix of that type.
        extracted_target_embeddings = target_node_embeddings[target_node_original_indices_in_type]

        # 5. Pass target node embeddings through the classification head
        head_device = next(self.classification_head.parameters()).device
        self.classification_head.to(head_device) # Ensure head is on its device
        
        output_logits = self.classification_head(extracted_target_embeddings.to(head_device))
        
        # SHAP often works better with probabilities for classification.
        # However, some SHAP methods (like GradientExplainer) might prefer logits.
        # For KernelExplainer, probabilities are generally fine.
        output_probs = F.softmax(output_logits, dim=-1)
        
        return output_probs # Return probabilities: (num_target_nodes, num_classes)

def get_shap_explanations_for_node_classification(
    model_wrapper: NodeClassificationModelWrapper,
    features_of_nodes_to_explain: torch.Tensor, # Raw features for the specific nodes (N, D_feat_raw)
    original_indices_of_nodes_to_explain_in_type: torch.Tensor, # Original indices within their ntype (N,)
    background_data_features: torch.Tensor, # Background dataset for SHAP (M, D_feat_raw)
                                            # These should also be raw features of the target_node_type.
    feature_names: Optional[List[str]] = None,
    shap_explainer_type: str = "kernel", # "kernel", "gradient", "deep" (currently only kernel supported)
    save_plot_path_prefix: Optional[str] = None,
    top_k_features_summary: int = 10
) -> Optional[Tuple[np.ndarray, Optional[shap.Explanation]]]:
    """
    Generates SHAP explanations for node classification.

    Args:
        model_wrapper: An instance of NodeClassificationModelWrapper.
        features_of_nodes_to_explain: Tensor of raw features for the nodes to be explained.
                                      Shape: (num_nodes_to_explain, num_raw_features).
        original_indices_of_nodes_to_explain_in_type: Tensor of original indices of these nodes
                                                      within their specific node type.
        background_data_features: Tensor of features from a background dataset used by SHAP
                                  to compute expected values. Shape: (num_background_samples, num_raw_features).
        feature_names: Optional list of names for the raw features.
        shap_explainer_type: Type of SHAP explainer to use.
        save_plot_path_prefix: If provided, path prefix to save SHAP plots.
        top_k_features_summary: Number of features to show in summary plots.

    Returns:
        A tuple containing:
        - shap_values_np: Numpy array of SHAP values (num_nodes, num_features, num_classes) or (num_nodes, num_features)
        - shap_explanation_obj: Optional SHAP Explanation object (if using newer SHAP versions).
        Returns None if explanation fails.
    """
    logger.info(f"Starting SHAP explanation for {features_of_nodes_to_explain.shape[0]} nodes using {shap_explainer_type} explainer.")
    
    device = features_of_nodes_to_explain.device
    model_wrapper.to(device) # Ensure model_wrapper and its submodules are on the right device

    # SHAP expects a function that takes a (N, D_feat) tensor and returns (N, D_out) tensor
    # The model_wrapper.forward needs two arguments. We need to create a lambda or partial.
    # SHAP KernelExplainer gives input as a numpy array.
    
    def shap_prediction_function(numpy_perturbed_features: np.ndarray) -> np.ndarray:
        # This function will be called by SHAP with perturbed features for a batch of instances.
        # 'numpy_perturbed_features' corresponds to one or more rows from 'features_of_nodes_to_explain'
        # where some features have been perturbed by SHAP.
        # The number of rows in numpy_perturbed_features depends on SHAP's internal batching.
        
        # We assume SHAP calls this for ONE target node at a time if we explain node-by-node,
        # or a batch if we explain multiple nodes simultaneously and SHAP supports it for the explainer type.
        # For KernelExplainer, it typically gets called with one row from the data to be explained,
        # but with features replaced by background data features.
        
        # This adaptation is tricky. KernelExplainer expects a function f(X) where X is (n_samples, n_features).
        # We need to map these n_samples (which are variations of a single node's features)
        # back to the GNN context.
        
        # For simplicity, let's assume for now we explain one node at a time with KernelExplainer.
        # The `features_of_nodes_to_explain` would be (1, num_raw_features) for a single node.
        # `original_indices_of_nodes_to_explain_in_type` would be (1,).
        
        # If explaining multiple nodes, this needs careful batching.
        # For now, assuming numpy_perturbed_features is (num_shap_samples, num_features) for ONE explained node.
        
        perturbed_features_torch = torch.from_numpy(numpy_perturbed_features).float().to(device)
        
        # We need the original index of the node being explained to pass to model_wrapper.
        # This setup is more suited if SHAP is called iteratively for each node to explain.
        # If we pass `features_of_nodes_to_explain` (multiple nodes) to `KernelExplainer`,
        # it will call `shap_prediction_function` with rows from this input.
        
        # Let's assume `shap_prediction_function` is called with feature sets corresponding to
        # variations of the nodes in `features_of_nodes_to_explain`.
        # The `target_node_original_indices_in_type` must align with these.
        # This requires careful handling of how SHAP calls the prediction function.
        
        # If `numpy_perturbed_features` has `S` samples (SHAP samples for coalitions),
        # and we are explaining `N` nodes, then for KernelExplainer, N=1 usually.
        # If N > 1, and KernelExplainer is passed X of shape (N, D_feat), it makes `S` calls
        # for *each* of the N samples, replacing features with background.
        # The call to model_wrapper here needs to correspond to *which* of the N original nodes
        # this current SHAP sample belongs to.
        
        # This is a simplification: Assumes `original_indices_of_nodes_to_explain_in_type`
        # can be broadcasted or that we handle batching outside if explaining multiple nodes.
        # For this example, let's assume `original_indices_of_nodes_to_explain_in_type`
        # refers to the indices for the rows in `numpy_perturbed_features` if they directly map.
        # THIS IS A COMPLEX PART FOR BATCHED GNN EXPLANATIONS WITH SHAP.
        
        # A common way for KernelExplainer: explain one instance at a time.
        # If `features_of_nodes_to_explain` is (1, D), then this function is called with (S, D)
        # where S is nsamples for KernelSHAP. All S samples refer to perturbations of that one node.
        # So, `target_node_original_indices_in_type` should be the index of that single node.

        # Let's assume we are explaining nodes one by one if using this current structure with KernelSHAP.
        # The `get_shap_explanations_for_node_classification` function would loop through nodes.
        # OR, if features_of_nodes_to_explain has N rows, SHAP's call will also have N rows,
        # and `original_indices_of_nodes_to_explain_in_type` would be the corresponding indices.

        # Simplest: Assume this function is called for a batch of inputs
        # where each row in `perturbed_features_torch` corresponds to a node
        # whose original index is in `original_indices_of_nodes_to_explain_in_type` (assuming direct mapping).
        # This is true if `KernelExplainer(model_fn, data=X_to_explain)`
        # Then `model_fn` is called with rows from `X_to_explain` (perturbed).
        
        # The model_wrapper expects `original_indices_of_nodes_to_explain_in_type` to match
        # the batch size of `perturbed_target_node_features`.
        # If SHAP calls with S samples for a single explained node, we need the index of that *single* node, repeated S times.
        # This function signature for SHAP is usually `f(X)` where `X` is `(num_samples, num_features)`.
        
        # This part needs to be robust. For now, assume that `original_indices_of_nodes_to_explain_in_type`
        # is passed correctly for the batch `perturbed_features_torch` represents.
        # This would imply `get_shap_explanations_for_node_classification` might call SHAP per node,
        # or `target_node_original_indices_in_type` is pre-tiled if KernelExplainer makes background samples.

        # Let's refine this after choosing a SHAP explainer.
        # For KernelExplainer, if `X` is the data to explain (N_nodes, N_features),
        # and `bg` is background (M_bg, N_features), `explainer.shap_values(X[i])` for one node.
        # Then `shap_prediction_function` is called with (N_samples_shap, N_features)
        # which are perturbations of `X[i]`. So `target_node_original_indices_in_type`
        # should be the index of `X[i]`.

        # This placeholder does not handle the indices correctly for a batch.
        # This function expects `target_node_original_indices_in_type` to be passed or be in scope.
        # This indicates `get_shap_explanations_for_node_classification` should iterate.
        # The current `model_wrapper.forward` takes `target_node_original_indices_in_type`.
        
        # For this to work with `shap.KernelExplainer(shap_prediction_function, background_data_features_np)`
        # and then `explainer.shap_values(features_of_nodes_to_explain_np[i_node_idx])`,
        # the `shap_prediction_function` must capture `original_indices_of_nodes_to_explain_in_type[i_node_idx]`.

        # This will be handled by iterating outside and capturing the index.
        # The function passed to KernelExplainer will be specific to ONE node.
        # For now, this is a generic def.
        
        # This function (shap_prediction_function) is the one passed to SHAP explainer.
        # It must take a NumPy array of shape (k, num_features) and return (k, num_classes).
        # k is the number of samples SHAP generates for its expectation calculation.
        # We assume each of these k samples is a perturbation of ONE original node's features.
        # So, `target_node_original_indices_in_type` should be the index of that ONE original node.
        
        # This definition needs to be created *inside* a loop in `get_shap_explanations_for_node_classification`
        # if explaining node by node.
        # For now, assume it's implicitly handled by the calling context of SHAP.

        # Simplified: assume this func is used per-node.
        # `original_indices_of_nodes_to_explain_in_type` here should be a single index, expanded for the batch `perturbed_features_torch`.
        # This is the most complex part to generalize for all SHAP explainers and batching.
        # For now, let's assume this is called for a single node's perturbations.
        
        # Revisit this when `get_shap_explanations_for_node_classification` is fleshed out.
        # The current structure is problematic.
        
        # To make it runnable, placeholder:
        # This needs to be a closure if used in a loop for each node.
        # For now, it won't run correctly without that loop structure.
        pass # Placeholder, to be defined in the loop of get_shap_explanations

    num_nodes_to_explain = features_of_nodes_to_explain.shape[0]
    all_shap_values_list = []
    
    # It's generally easier to run SHAP KernelExplainer node by node
    for i in range(num_nodes_to_explain):
        logger.info(f"  Explaining node {i+1}/{num_nodes_to_explain}...")
        current_node_features_np = features_of_nodes_to_explain[i:i+1].cpu().numpy() # SHAP expects numpy (1, D_feat)
        current_node_original_idx_in_type = original_indices_of_nodes_to_explain_in_type[i:i+1].to(device) # (1,)
        
        # Define the prediction function for SHAP for this specific node
        def single_node_shap_pred_fn(perturbed_features_for_current_node_np: np.ndarray) -> np.ndarray:
            # perturbed_features_for_current_node_np is (num_shap_samples, D_feat)
            p_features_torch = torch.from_numpy(perturbed_features_for_current_node_np).float().to(device)
            
            # The model_wrapper needs the original index of THIS node, repeated for the SHAP sample batch.
            # `current_node_original_idx_in_type` is already a tensor for the single node.
            # If p_features_torch has S samples, we need S indices.
            indices_for_wrapper = current_node_original_idx_in_type.repeat(p_features_torch.shape[0])
            
            with torch.no_grad(): # SHAP does perturbations, model pass should be inference
                model_output_probs = model_wrapper(p_features_torch, indices_for_wrapper)
            return model_output_probs.cpu().numpy()

        if shap_explainer_type == "kernel":
            # KernelExplainer: model-agnostic, uses LIME. Can be slow.
            # Background data should be representative of the typical feature values.
            # Using a subset (e.g., KMedoids) of background_data_features is common.
            # For GNNs, background could be features of other nodes of the same type.
            summarized_background = shap.kmeans(background_data_features.cpu().numpy(), k=min(100, background_data_features.shape[0]))
            explainer = shap.KernelExplainer(single_node_shap_pred_fn, summarized_background)
            # `nsamples`: number of times to re-evaluate the model for each explanation.
            # "auto" is usually 2 * D + 2048. Can be int.
            shap_values_one_node = explainer.shap_values(current_node_features_np, nsamples="auto") 
                                                        # l1_reg="aic" can help with feature selection
        # Note: GradientExplainer and DeepExplainer support can be added for PyTorch models
        # These would require model_wrapper to be directly a PyTorch model and inputs to be tensors
        else:
            logger.error(f"Unsupported SHAP explainer type: {shap_explainer_type}")
            return None
        
        all_shap_values_list.append(shap_values_one_node)
        logger.info(f"  SHAP values computed for node {i+1}. Shape: {np.array(shap_values_one_node).shape}")

    if not all_shap_values_list:
        logger.warning("No SHAP values were generated.")
        return None

    # Stack SHAP values:
    # If shap_values_one_node is a list (for multi-class output), stacking needs care.
    # For (num_classes, num_features) per node -> (num_nodes, num_classes, num_features)
    # For (num_features) per node (single output) -> (num_nodes, num_features)
    
    try:
        if isinstance(all_shap_values_list[0], list): # Multi-output (multi-class)
             # Each element of all_shap_values_list is a list of arrays (one per class)
             # Target shape: (num_nodes, num_classes, num_features)
             num_classes = len(all_shap_values_list[0])
             num_features = all_shap_values_list[0][0].shape[-1] # features are last dim of array
             
             shap_values_np = np.zeros((num_nodes_to_explain, num_classes, num_features))
             for i_node, sv_node_list in enumerate(all_shap_values_list):
                 for i_class, sv_class_array in enumerate(sv_node_list):
                     # sv_class_array might be (1, num_features) from explainer.shap_values(X_instance)
                     shap_values_np[i_node, i_class, :] = sv_class_array.reshape(-1) 
        else: # Single-output or already aggregated
            # Each element is (num_samples_shap_took_for_instance (usually 1), num_features)
            # or directly (num_features)
            shap_values_np = np.array([sv.reshape(-1) for sv in all_shap_values_list])
            # Target shape: (num_nodes, num_features)
            
    except Exception as e_stack:
        logger.error(f"Error stacking SHAP values: {e_stack}. Raw list: {all_shap_values_list}")
        return None


    logger.info(f"Final stacked SHAP values shape: {shap_values_np.shape}")

    # Create SHAP Explanation object for easier plotting (if shap version supports it)
    shap_explanation_obj = None
    try:
        # features_of_nodes_to_explain is (N, D_feat)
        # shap_values_np could be (N, D_feat) or (N, C, D_feat)
        # Base values: KernelExplainer stores `explainer.expected_value`
        # It can be scalar or (num_classes,)
        expected_value = explainer.expected_value 
        
        # For multi-class, shap_values_np might be (N,C,F) and expected_value (C,)
        # If shap_values_np is (N,F), expected_value is scalar.
        
        # Check dimensions for Explanation object
        # data = features_of_nodes_to_explain.cpu().numpy()
        # if shap_values_np.ndim == 3 and data.ndim == 2: # (N,C,F) values, (N,F) data
        #    pass # This is okay for Explanation object with multi-output
        
        shap_explanation_obj = shap.Explanation(
            values=shap_values_np,
            base_values=expected_value,
            data=features_of_nodes_to_explain.cpu().numpy(), # Raw features of explained instances
            feature_names=feature_names
        )
    except Exception as e_expl_obj:
        logger.warning(f"Could not create SHAP Explanation object: {e_expl_obj}. Plotting may use raw values.")

    # Generate and save plots if path prefix is given
    if save_plot_path_prefix and shap_explanation_obj:
        try:
            # Summary plot (bar chart of mean absolute SHAP values)
            plt.figure()
            shap.summary_plot(shap_explanation_obj, plot_type="bar", max_display=top_k_features_summary, show=False)
            plt.savefig(f"{save_plot_path_prefix}_summary_bar.png", bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP summary bar plot to {save_plot_path_prefix}_summary_bar.png")

            # Summary plot (beeswarm) - can be slow for many features/nodes
            if num_nodes_to_explain < 50: # Limit beeswarm for performance
                plt.figure()
                shap.summary_plot(shap_explanation_obj, plot_type="dot", max_display=top_k_features_summary, show=False) # or "violin"
                plt.savefig(f"{save_plot_path_prefix}_summary_beeswarm.png", bbox_inches='tight')
                plt.close()
                logger.info(f"Saved SHAP summary beeswarm plot to {save_plot_path_prefix}_summary_beeswarm.png")

            # Force plots for individual explanations (if explaining few nodes)
            if num_nodes_to_explain <= 5: # Generate force plots for first few
                for i in range(num_nodes_to_explain):
                    # Force plot requires JS, often saved as HTML or displayed in notebook
                    # shap.force_plot(explainer.expected_value, shap_values_np[i,:], features_of_nodes_to_explain.cpu().numpy()[i,:], feature_names=feature_names, matplotlib=True, show=False)
                    # plt.savefig(f"{save_plot_path_prefix}_force_plot_node_{i}.png", bbox_inches='tight')
                    # plt.close()
                    # logger.info(f"Saved SHAP force plot for node {i} to {save_plot_path_prefix}_force_plot_node_{i}.png")
                    
                    # HTML force plot
                    try:
                        force_plot_html = shap.force_plot(shap_explanation_obj.base_values, 
                                                          shap_explanation_obj.values[i], 
                                                          shap_explanation_obj.data[i],
                                                          feature_names=shap_explanation_obj.feature_names,
                                                          show=False)
                        shap.save_html(f"{save_plot_path_prefix}_force_plot_node_{i}.html", force_plot_html)
                        logger.info(f"Saved SHAP force plot HTML for node {i} to {save_plot_path_prefix}_force_plot_node_{i}.html")
                    except Exception as e_force_html:
                         logger.warning(f"Could not save HTML force plot for node {i}: {e_force_html}")


        except Exception as e_plot:
            logger.error(f"Error generating SHAP plots: {e_plot}")
            
    elif save_plot_path_prefix and not shap_explanation_obj: # Fallback for older SHAP or if Explanation obj failed
        try:
            plt.figure()
            shap.summary_plot(shap_values_np, features_of_nodes_to_explain.cpu().numpy(), 
                              plot_type="bar", feature_names=feature_names, max_display=top_k_features_summary, show=False)
            plt.savefig(f"{save_plot_path_prefix}_summary_bar.png", bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP summary bar plot (raw) to {save_plot_path_prefix}_summary_bar.png")
        except Exception as e_raw_plot:
            logger.error(f"Error generating raw SHAP summary plot: {e_raw_plot}")


    return shap_values_np, shap_explanation_obj 