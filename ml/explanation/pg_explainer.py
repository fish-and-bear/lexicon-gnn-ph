import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Tuple, List, Dict, Optional, Union

# GNNToBeExplained class removed as PGExplainer will work with the main HGMAE model.

class PGExplainer(nn.Module):
    def __init__(self,
                 model_to_explain, # This would be an instance of your main GNN (e.g., HGMAE) or a specific head
                 concatenated_feature_dim: int, # Dimension of (src_emb + dst_emb)
                 hidden_dim: int = 64,
                 temp: float = 5.0,
                 epochs: int = 30,
                 lr: float = 0.005,
                 device: Optional[Union[str, torch.device]] = None): # Added device parameter
        super().__init__()
        self.model_to_explain = model_to_explain
        self.epochs = epochs
        self.lr = lr
        self.temp = temp # Temperature for concrete distribution

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        self.elayers = nn.Sequential(
            nn.Linear(concatenated_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device) # Move elayers to device

    def _get_edge_embeddings_hetero(self,
                                    graph: dgl.DGLGraph,
                                    all_embs: Dict[str, torch.Tensor],
                                    edges_src_ids: torch.Tensor,
                                    edges_dst_ids: torch.Tensor,
                                    src_ntype: str,
                                    dst_ntype: str) -> Optional[torch.Tensor]:
        """
        Helper to get concatenated embeddings for edges of a specific canonical edge type.
        Assumes all embeddings in all_embs for different node types have the SAME dimension.
        Args:
            graph: The graph (can be a subgraph corresponding to one etype).
            all_embs: Dictionary mapping node type to its embedding tensor.
            edges_src_ids: Source node IDs for the edges (local to src_ntype in graph).
            edges_dst_ids: Destination node IDs for the edges (local to dst_ntype in graph).
            src_ntype: Node type of source nodes.
            dst_ntype: Node type of destination nodes.
        Returns:
            Concatenated embeddings tensor or None if embeddings are missing.
        """
        if src_ntype not in all_embs or dst_ntype not in all_embs:
            print(f"Error: Embeddings for source type '{src_ntype}' or destination type '{dst_ntype}' not in all_embs.")
            return None
        
        # Ensure embeddings are on the correct device
        src_embeddings_tensor = all_embs[src_ntype].to(self.device)
        dst_embeddings_tensor = all_embs[dst_ntype].to(self.device)

        # Check if edge IDs are within bounds for the provided embedding tensors
        if src_embeddings_tensor.numel() == 0 and edges_src_ids.numel() > 0: # Check if embedding tensor is empty but IDs are not
             print(f"Error: Source embeddings for ntype '{src_ntype}' are empty, but source IDs are present.")
             return None
        if dst_embeddings_tensor.numel() == 0 and edges_dst_ids.numel() > 0:
             print(f"Error: Destination embeddings for ntype '{dst_ntype}' are empty, but destination IDs are present.")
             return None
        
        if src_embeddings_tensor.numel() > 0 and edges_src_ids.numel() > 0 and edges_src_ids.max() >= src_embeddings_tensor.shape[0]:
            print(f"Error: Max source ID {edges_src_ids.max().item()} out of bounds for {src_ntype} embeddings shape {src_embeddings_tensor.shape}")
            return None
        if dst_embeddings_tensor.numel() > 0 and edges_dst_ids.numel() > 0 and edges_dst_ids.max() >= dst_embeddings_tensor.shape[0]:
            print(f"Error: Max destination ID {edges_dst_ids.max().item()} out of bounds for {dst_ntype} embeddings shape {dst_embeddings_tensor.shape}")
            return None
            
        # Handle cases where there are no edges (but embeddings might exist for the type)
        if edges_src_ids.numel() == 0 or edges_dst_ids.numel() == 0:
            # Should ideally return an empty tensor with the correct feature dimension
            # For now, this case is typically handled before calling _get_edge_embeddings_hetero
            # by checking if sg.num_edges(cetype) > 0. If called with empty edge IDs,
            # this would cause an error at emb_s/emb_d indexing.
            # Let's assume it's called with non-empty edge IDs based on upstream checks.
            # If, for some reason, it's called with empty IDs, and embeddings exist, this is problematic.
            # This function should primarily be called when there are edges.
            # If edges_src_ids is empty, this implies 0 edges, so an empty feature tensor is appropriate.
            # The concatenated dim is self.elayers[0].in_features
            return torch.empty((0, self.elayers[0].in_features), device=self.device, dtype=src_embeddings_tensor.dtype if src_embeddings_tensor.numel() > 0 else (dst_embeddings_tensor.dtype if dst_embeddings_tensor.numel() > 0 else torch.float))


        emb_s = src_embeddings_tensor[edges_src_ids]
        emb_d = dst_embeddings_tensor[edges_dst_ids]
        
        return torch.cat([emb_s, emb_d], dim=1)

    def forward(self,
                graph_or_subgraph: dgl.DGLGraph,
                all_embs: Dict[str, torch.Tensor],
                edges_src_ids: torch.Tensor,
                edges_dst_ids: torch.Tensor,
                src_ntype_for_edges: str,
                dst_ntype_for_edges: str):
        """
        Predicts edge mask probabilities for the given edges (of a single canonical type).
        Args:
            graph_or_subgraph: The DGL graph or subgraph (e.g., 1-hop neighborhood).
            all_embs: Dictionary of all node embeddings {ntype: tensor}.
                      These embeddings should correspond to the nodes in graph_or_subgraph.
                      E.g., if graph_or_subgraph is 'sg', all_embs should be 'sg_all_embs'.
            edges_src_ids: Source node IDs for the edges *within graph_or_subgraph* for src_ntype_for_edges.
            edges_dst_ids: Destination node IDs for the edges *within graph_or_subgraph* for dst_ntype_for_edges.
            src_ntype_for_edges: Node type of source nodes for these edges.
            dst_ntype_for_edges: Node type of destination nodes for these edges.
        """
        edge_feats = self._get_edge_embeddings_hetero(graph_or_subgraph, all_embs,
                                                      edges_src_ids, edges_dst_ids,
                                                      src_ntype_for_edges, dst_ntype_for_edges)
        if edge_feats is None:
            num_edges = edges_src_ids.shape[0]
            print(f"Warning: _get_edge_embeddings_hetero returned None for ({src_ntype_for_edges} -> {dst_ntype_for_edges}). Returning zero logits.")
            return torch.zeros(num_edges, 1, device=self.device)
        
        if edge_feats.shape[0] == 0: # No edges, so no features
            return torch.empty((0,1), device=self.device)


        edge_logits = self.elayers(edge_feats)
        return edge_logits

    def _loss(self, pred_edge_logits, target_pred_masked, original_pred_detached, reg_coeff1=0.01, reg_coeff2=0.01):
        """
        Calculates the loss for training PGExplainer.
        """
        if target_pred_masked is None or original_pred_detached is None:
            print("Warning: target_pred_masked or original_pred_detached is None in _loss. Returning high loss.")
            return torch.tensor(float('inf'), device=pred_edge_logits.device)

        if target_pred_masked.shape != original_pred_detached.shape:
            print(f"Warning: Shape mismatch in _loss. target_pred_masked: {target_pred_masked.shape}, original_pred_detached: {original_pred_detached.shape}")
            if original_pred_detached.shape[0] == 1 and target_pred_masked.shape[0] > 1 and original_pred_detached.shape[1:] == target_pred_masked.shape[1:]:
                 original_pred_detached = original_pred_detached.expand_as(target_pred_masked)
            elif target_pred_masked.numel() == 0 and original_pred_detached.numel() > 0 : # If masked pred is empty (e.g. node isolated)
                 print("Warning: target_pred_masked is empty in _loss. This might mean the node was isolated. Fidelity loss will be high.")
                 # Create a zero tensor of the same shape as original_pred_detached but with opposite "probabilities"
                 # to maximize KL divergence, or simply make fidelity loss very high.
                 # For simplicity, let's make target_pred_masked very different from original.
                 # This is a placeholder for a more principled way to handle isolated nodes.
                 target_pred_masked = torch.zeros_like(original_pred_detached) - 100 # very wrong logits
            else:
                return torch.tensor(float('inf'), device=pred_edge_logits.device) # Unhandled mismatch

        # Fidelity loss: KL divergence
        # Ensure inputs to kl_div are in log-space for target if log_target=True
        log_softmax_masked = F.log_softmax(target_pred_masked, dim=-1)
        softmax_original = F.softmax(original_pred_detached, dim=-1)
        
        fidelity_loss = F.kl_div(log_softmax_masked, softmax_original, reduction='batchmean', log_target=False) # log_target=False as softmax_original is not log

        edge_probs = torch.sigmoid(pred_edge_logits)
        sparsity_loss = torch.mean(edge_probs) if edge_probs.numel() > 0 else torch.tensor(0.0, device=self.device)
        
        total_loss = reg_coeff1 * fidelity_loss + reg_coeff2 * sparsity_loss
        return total_loss

    def train_explainer_node(self,
                             original_graph: dgl.DGLGraph,
                             node_idx_to_explain: int,
                             node_ntype: str,
                             all_embs_global: Dict[str, torch.Tensor],
                             target_pred_func, # Function: (graph, all_embs_for_graph, node_id, node_type) -> prediction_for_node
                             k_hop: int = 1,
                             **kwargs):
        print(f"Starting PGExplainer training for node {node_idx_to_explain} (type: {node_ntype}).")
        optimizer = torch.optim.Adam(self.elayers.parameters(), lr=self.lr)
        if hasattr(self.model_to_explain, 'eval'): # Ensure model_to_explain (main GNN/head) is in eval mode
            self.model_to_explain.eval()

        sg_nodes_global_dict = {node_ntype: torch.tensor([node_idx_to_explain], device=original_graph.device)}
        sg = dgl.khop_subgraph(original_graph, sg_nodes_global_dict, k=k_hop, store_ids=True)
        sg = sg.to(self.device)

        if sg.num_edges() == 0:
            print(f"Node {node_idx_to_explain} has no edges in its {k_hop}-hop subgraph. Cannot train PGExplainer.")
            return

        sg_all_embs = {}
        for ntype_sg in sg.ntypes:
            if ntype_sg in all_embs_global:
                global_ids_for_ntype_in_sg = sg.nodes[ntype_sg].data[dgl.NID]
                sg_all_embs[ntype_sg] = all_embs_global[ntype_sg][global_ids_for_ntype_in_sg].to(self.device)
            else:
                emb_dim_fallback = next(iter(all_embs_global.values())).shape[1] if all_embs_global else self.elayers[0].in_features // 2
                print(f"Warning: Embeddings for node type '{ntype_sg}' not in all_embs_global. Using zeros of dim {emb_dim_fallback}.")
                sg_all_embs[ntype_sg] = torch.zeros((sg.num_nodes(ntype_sg), emb_dim_fallback), device=self.device)
        
        with torch.no_grad():
            all_embs_global_pred_device = {k: v.to(original_graph.device) for k, v in all_embs_global.items()}
            original_node_pred = target_pred_func(original_graph, all_embs_global_pred_device, node_idx_to_explain, node_ntype)
            if original_node_pred is None:
                print(f"Error: Original prediction for node {node_idx_to_explain} could not be obtained. Aborting training.")
                return
            original_node_pred = original_node_pred.detach().to(self.device)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            all_edge_logits_sg_list = []
            all_edge_masks_sg_list = []
            data_dict_for_masked_graph = {}
            num_nodes_dict_original = {ntype: original_graph.num_nodes(ntype) for ntype in original_graph.ntypes}

            for cetype_sg in sg.canonical_etypes:
                if sg.num_edges(cetype_sg) == 0:
                    continue
                
                s_nodes_local_sg, d_nodes_local_sg = sg.edges(etype=cetype_sg)
                src_nt_sg, et_name_sg, dst_nt_sg = cetype_sg

                # sg_all_embs are already prepared for the subgraph sg and on self.device
                edge_logits_cetype = self.forward(sg, sg_all_embs, s_nodes_local_sg, d_nodes_local_sg, src_nt_sg, dst_nt_sg)
                
                if edge_logits_cetype.numel() > 0: # Only process if there are edges/logits
                    all_edge_logits_sg_list.append(edge_logits_cetype)

                    u = torch.rand(edge_logits_cetype.shape, device=self.device)
                    edge_mask_probs_cetype = torch.sigmoid((edge_logits_cetype + torch.log(u) - torch.log(1 - u)) / self.temp)
                    all_edge_masks_sg_list.append(edge_mask_probs_cetype.squeeze(-1))

                    # For constructing masked_graph using global IDs
                    global_s_nodes_cetype = sg.nodes[src_nt_sg].data[dgl.NID][s_nodes_local_sg]
                    global_d_nodes_cetype = sg.nodes[dst_nt_sg].data[dgl.NID][d_nodes_local_sg]
                    
                    # Bernoulli sample based on probabilities for this cetype
                    selected_edges_indices_cetype = torch.bernoulli(edge_mask_probs_cetype.squeeze(-1)).bool()
                    
                    data_dict_for_masked_graph[cetype_sg] = (global_s_nodes_cetype[selected_edges_indices_cetype],
                                                             global_d_nodes_cetype[selected_edges_indices_cetype])
                else: # No edges of this type in sg, or forward returned empty
                    data_dict_for_masked_graph[cetype_sg] = (torch.tensor([], dtype=torch.long, device=self.device),
                                                             torch.tensor([], dtype=torch.long, device=self.device))


            if not all_edge_logits_sg_list:
                print(f"No edge logits computed for any cetype in subgraph for node {node_idx_to_explain}. Skipping epoch.")
                continue
                
            combined_all_edge_logits = torch.cat(all_edge_logits_sg_list, dim=0)
            
            masked_graph = dgl.heterograph(data_dict_for_masked_graph, num_nodes_dict=num_nodes_dict_original).to(self.device)
            
            all_embs_global_masked_device = {k: v.to(masked_graph.device) for k, v in all_embs_global.items()}
            masked_node_pred = target_pred_func(masked_graph, all_embs_global_masked_device, node_idx_to_explain, node_ntype)

            if masked_node_pred is None:
                print(f"Warning: Masked prediction for node {node_idx_to_explain} is None. Using zeros for loss.")
                # Try to get output dimension for zeros
                out_dim_pred = original_node_pred.shape[-1] if original_node_pred is not None else 1
                masked_node_pred = torch.zeros((1, out_dim_pred), device=self.device) # Assume batch size 1

            masked_node_pred = masked_node_pred.to(self.device)
            
            loss = self._loss(combined_all_edge_logits, masked_node_pred, original_node_pred)
            
            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Loss is {loss.item()}. Skipping backward pass for this epoch.")
                continue

            loss.backward()
            optimizer.step()
            
            if epoch % max(1, (self.epochs // 10)) == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")
        
        print(f"PGExplainer training finished for node {node_idx_to_explain}.")


    def explain_node(self,
                     original_graph: dgl.DGLGraph,
                     node_idx_to_explain: int, # This is the GLOBAL ID in original_graph
                     node_ntype: str,
                     all_embs_global: Dict[str, torch.Tensor], # Embeddings for the original_graph
                     k_hop: int = 1,
                     top_k_edges_per_etype: int = 5) -> Optional[Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]]:
        """
        Explains a single node's prediction by returning the most important edges in its k-hop subgraph.

        Args:
            original_graph: The original DGL graph.
            node_idx_to_explain: The global ID of the node in original_graph to explain.
            node_ntype: The node type of the node to explain.
            all_embs_global: Dictionary of global node embeddings for the original_graph.
            k_hop: The number of hops for the subgraph around the node.
            top_k_edges_per_etype: The number of most important edges to return per edge type.

        Returns:
            A tuple containing:
            - explanation_per_etype: Dict mapping canonical etype string to a tuple of 
              (top_k_edge_masks, top_k_src_ids_local, top_k_dst_ids_local).
              Returns None if explanation fails.
            - subgraph_node_map_to_original_graph: Dict mapping ntype to a tensor of original global IDs
              corresponding to the nodes in the k-hop subgraph. Returns None if explanation fails.
            Returns None if explanation fails for any reason.
        """
        if not (0 <= node_idx_to_explain < original_graph.num_nodes(node_ntype)):
            self.logger.error(f"Node to explain ({node_ntype}, {node_idx_to_explain}) is out of bounds for original_graph.")
            return None

        self.eval() # Ensure explainer is in eval mode

        # 1. Extract k-hop subgraph around the node_idx_to_explain
        #    subgraph_node_map_to_original_graph maps new node IDs in khop_subgraph back to original graph IDs
        try:
            khop_subgraph, subgraph_node_map_to_original_graph = dgl.khop_subgraph(
                original_graph, 
                {node_ntype: torch.tensor([node_idx_to_explain], device=self.device)},
                k=k_hop,
                relabel_nodes=True # Essential for PGExplainer's assumptions of dense local IDs
            )
        except Exception as e_khop:
            self.logger.error(f"Error during dgl.khop_subgraph for node ({node_ntype}, {node_idx_to_explain}): {e_khop}", exc_info=True)
            return None
        
        if khop_subgraph is None or khop_subgraph.number_of_nodes() == 0:
            self.logger.warning(f"k-hop subgraph for node ({node_ntype}, {node_idx_to_explain}) is empty. Cannot explain.")
            return None
        
        # The node_idx_to_explain in the khop_subgraph will be 0 for its ntype if it's the only seed node of that type.
        # We need to find its new local ID in the khop_subgraph for that ntype.
        # subgraph_node_map_to_original_graph[node_ntype] contains the original IDs of nodes of type node_ntype in the subgraph.
        # The local ID of node_idx_to_explain is where it appears in this tensor.
        local_node_id_in_khop_indices = (subgraph_node_map_to_original_graph[node_ntype] == node_idx_to_explain).nonzero(as_tuple=True)[0]
        if local_node_id_in_khop_indices.nelement() == 0:
            self.logger.error(f"Could not find original node ({node_ntype}, {node_idx_to_explain}) in the subgraph_node_map. This is unexpected.")
            return None
        local_node_id_in_khop = local_node_id_in_khop_indices.item()

        self.logger.info(f"Explaining node original_id={node_idx_to_explain} (local_id_in_khop={local_node_id_in_khop}, type={node_ntype}) with k={k_hop} hop subgraph ({khop_subgraph.number_of_nodes()} total nodes). Map keys: {subgraph_node_map_to_original_graph.keys()}")

        # 2. Get embeddings for the k-hop subgraph from all_embs_global
        subgraph_embs = {}
        for ntype_khop, global_ids_in_subgraph in subgraph_node_map_to_original_graph.items():
            if ntype_khop in all_embs_global and all_embs_global[ntype_khop] is not None:
                subgraph_embs[ntype_khop] = all_embs_global[ntype_khop][global_ids_in_subgraph].to(self.device)
            else:
                self.logger.warning(f"Embeddings for ntype '{ntype_khop}' not found in all_embs_global or are None. Subgraph might be missing features for this type.")
                # PGExplainer expects all ntypes in the subgraph to have embeddings. If not, it might fail later.
                # Create zero embeddings if missing, assuming concatenated_feature_dim is based on sum of two such embeddings.
                # This is a patch; ideally, all_embs_global should be complete for nodes in the subgraph.
                # The dimension for these zero embeddings needs to be consistent with what PGExplainer expects.
                # This is tricky as PGExplainer's concatenated_feature_dim is about edge embeddings.
                # For node embeddings themselves, it just passes them through.
                # Let's assume the target dimension is `all_embs_global[some_valid_ntype].shape[1]`
                if subgraph_embs: # if at least one other ntype has embeddings
                    fallback_dim = next(iter(subgraph_embs.values())).shape[1]
                else: # No embeddings at all, this is problematic
                    self.logger.error(f"No embeddings available in subgraph_embs to determine fallback dimension for '{ntype_khop}'. PGExplainer will likely fail.")
                    fallback_dim = 1 # A wild guess, likely to cause errors

                subgraph_embs[ntype_khop] = torch.zeros((len(global_ids_in_subgraph), fallback_dim), device=self.device)
                self.logger.info(f"  Created zero placeholder embeddings for '{ntype_khop}' in subgraph (shape: {subgraph_embs[ntype_khop].shape})")

        # 3. Iterate over canonical edge types in the k-hop subgraph to get edge masks
        explanation_per_etype: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        if not khop_subgraph.canonical_etypes:
            self.logger.warning(f"k-hop subgraph for node ({node_ntype}, {node_idx_to_explain}) has no canonical_etypes. Returning empty explanation.")
            return {}, subgraph_node_map_to_original_graph # Return empty dict for explanations but valid map

        for cetype_khop_tuple in khop_subgraph.canonical_etypes:
            src_ntype_khop, etype_str_khop, dst_ntype_khop = cetype_khop_tuple
            num_edges_khop = khop_subgraph.num_edges(cetype_khop_tuple)

            if num_edges_khop == 0:
                continue
            
            # Get all edges for this canonical type in the k-hop subgraph
            edges_src_ids_local_khop, edges_dst_ids_local_khop = khop_subgraph.edges(etype=cetype_khop_tuple)

            # Get edge importance masks using the explainer's forward method
            edge_mask_for_etype = self.forward(
                khop_subgraph, 
                subgraph_embs, 
                edges_src_ids_local_khop, 
                edges_dst_ids_local_khop, 
                src_ntype_khop, 
                dst_ntype_khop
            )

            if edge_mask_for_etype is None or edge_mask_for_etype.nelement() == 0:
                continue
            
            # edge_mask_for_etype is of shape [num_edges_khop_for_etype]
            # Select top_k edges by mask value
            actual_top_k = min(top_k_edges_per_etype, len(edge_mask_for_etype))
            if actual_top_k <= 0:
                continue

            try:
                top_k_scores, top_k_indices = torch.topk(edge_mask_for_etype, k=actual_top_k)
            except RuntimeError as e_topk:
                 self.logger.warning(f"  torch.topk failed for etype {etype_str_khop} (mask_shape: {edge_mask_for_etype.shape}, k={actual_top_k}): {e_topk}. Skipping this etype.")
                 continue

            top_k_src_ids_local = edges_src_ids_local_khop[top_k_indices]
            top_k_dst_ids_local = edges_dst_ids_local_khop[top_k_indices]
            
            explanation_per_etype[etype_str_khop] = (top_k_scores, top_k_src_ids_local, top_k_dst_ids_local)

        if not explanation_per_etype:
            self.logger.warning(f"No explanations generated for any edge type in the k-hop subgraph of node ({node_ntype}, {node_idx_to_explain}).")
            # Still return the map, as the subgraph itself might be informative (e.g., if it was empty of certain edges)
            return {}, subgraph_node_map_to_original_graph 
            
        return explanation_per_etype, subgraph_node_map_to_original_graph

# Example Usage (conceptual, needs actual HGMAE model and data)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_data = {
        ('word', 'rev_defined_by', 'definition'): (torch.tensor([0, 1]), torch.tensor([0, 0])),
        ('word', 'related_to', 'word'): (torch.tensor([0, 1]), torch.tensor([1, 0]))
    }
    num_nodes_dict = {'word': 2, 'definition': 1}
    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict).to(device)

    emb_dim = 16
    all_embs = {
        'word': torch.randn(g.num_nodes('word'), emb_dim).to(device),
        'definition': torch.randn(g.num_nodes('definition'), emb_dim).to(device)
    }
    
    class DummyGNNHead(nn.Module):
        def __init__(self, out_classes=3):
            super().__init__()
            self.out_classes = out_classes
        def forward(self, graph, all_embs_for_pred, node_id, node_type):
            if node_type not in graph.ntypes or node_id >= graph.num_nodes(node_type):
                return torch.zeros(1, self.out_classes, device=device)
            return torch.randn(1, self.out_classes, device=device)

    dummy_prediction_model = DummyGNNHead(out_classes=3).to(device)

    pg_explainer = PGExplainer(model_to_explain=dummy_prediction_model,
                               concatenated_feature_dim=emb_dim * 2,
                               hidden_dim=32, epochs=5, lr=0.01, device=device)
    
    node_to_explain_global_id = 0
    node_to_explain_ntype = 'word'

    def example_target_pred_func(current_graph, current_all_embs, target_node_global_id, target_node_ntype):
        # This function receives the graph (original or masked) and corresponding embeddings.
        # It should return the prediction for target_node_global_id (of type target_node_ntype).
        # The dummy_prediction_model itself handles the node_id within its forward.
        return dummy_prediction_model(current_graph, current_all_embs, target_node_global_id, target_node_ntype)

    print(f"\nTraining PGExplainer for node {node_to_explain_global_id} (type {node_to_explain_ntype})...")
    pg_explainer.train_explainer_node(
        original_graph=g,
        node_idx_to_explain=node_to_explain_global_id,
        node_ntype=node_to_explain_ntype,
        all_embs_global=all_embs,
        target_pred_func=example_target_pred_func,
        k_hop=1
    )
    
    print(f"\nGetting explanation for node {node_to_explain_global_id} (type {node_to_explain_ntype})...")
    explanation_results_dict, subgraph_node_map_to_original_graph = pg_explainer.explain_node(
        original_graph=g,
        node_idx_to_explain=node_to_explain_global_id,
        node_ntype=node_to_explain_ntype,
        all_embs_global=all_embs,
        k_hop=1,
        top_k_edges_per_etype=2
    )
    
    if explanation_results_dict:
        print("\n--- Explanation Results ---")
        for etype_str, (exp_src, exp_dst, exp_importance) in explanation_results_dict.items():
            print(f"  Edge Type: {etype_str}")
            if exp_src.numel() == 0:
                print("    No important edges found for this type.")
                continue
            for i in range(len(exp_src)):
                print(f"    - Edge ({exp_src[i].item()} -> {exp_dst[i].item()}), Importance: {exp_importance[i].item():.4f}")
    else:
        print("No explanation generated.") 