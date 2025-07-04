"""
Graph construction for Philippine Lexicon GNN.
Builds a PyG HeteroData object from DB or JSON.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
from torch_geometric.data import HeteroData
from .utils import char_cnn_embed, normalize_numeric

NodeTypes = ["Word", "Morpheme", "Form", "Sense", "Language"]
EdgeTypes = [
    ("Word", "HAS_FORM", "Form"),
    ("Form", "OF_WORD", "Word"),
    ("Word", "HAS_SENSE", "Sense"),
    ("Word", "DERIVED_FROM", "Word"),
    ("Word", "HAS_AFFIX", "Morpheme"),
    ("Word", "RELATED", "Word"),
    ("Word", "SHARES_PHONOLOGY", "Word"),
    ("Word", "SHARES_ETYMOLOGY", "Word"),
]

def build_hetero_graph(raw: Dict[str, Any], device: Optional[torch.device] = None) -> HeteroData:
    """
    Converts raw node/edge dicts to a PyG HeteroData object.
    All string features are embedded via char-CNN.
    """
    data = HeteroData()
    
    # --- Nodes ---
    # Words
    word_lemmas = [w["lemma"] for w in raw["words"]]
    if word_lemmas:
        data["Word"].x = char_cnn_embed(word_lemmas)
        data["Word"].freq = torch.tensor([normalize_numeric(w.get("frequency", 1.0), 0.0, 1000.0) 
                                         for w in raw["words"]], dtype=torch.float32).unsqueeze(1)
        data["Word"].node_id = torch.tensor([w["id"] for w in raw["words"]], dtype=torch.long)
    
    # Morphemes
    morpheme_txts = [m["morpheme_text"] for m in raw.get("morphemes", [])]
    if morpheme_txts:
        data["Morpheme"].x = char_cnn_embed(morpheme_txts)
        data["Morpheme"].node_id = torch.tensor([m.get("id", i) for i, m in enumerate(raw.get("morphemes", []))], 
                                                dtype=torch.long)
    
    # Forms
    form_txts = [f["form"] for f in raw.get("forms", [])]
    if form_txts:
        data["Form"].x = char_cnn_embed(form_txts)
        data["Form"].node_id = torch.tensor([f["id"] for f in raw.get("forms", [])], dtype=torch.long)
    
    # Senses
    sense_txts = [s["definition_text"] for s in raw.get("senses", [])]
    if sense_txts:
        data["Sense"].x = char_cnn_embed(sense_txts)
        data["Sense"].node_id = torch.tensor([s["id"] for s in raw.get("senses", [])], dtype=torch.long)
    
    # Languages
    lang_txts = [l["language_code"] for l in raw.get("languages", [])]
    if lang_txts:
        data["Language"].x = char_cnn_embed(lang_txts)
        data["Language"].node_id = torch.tensor([l["id"] for l in raw.get("languages", [])], dtype=torch.long)
    
    # --- Create ID mappings ---
    # Map original IDs to new indices
    word_id_map = {w["id"]: i for i, w in enumerate(raw["words"])}
    form_id_map = {f["id"]: i for i, f in enumerate(raw.get("forms", []))}
    morpheme_id_map = {m.get("id", i): i for i, m in enumerate(raw.get("morphemes", []))}
    sense_id_map = {s["id"]: i for i, s in enumerate(raw.get("senses", []))}
    
    # --- Edges ---
    def edge_index(pairs: List[Tuple[int, int]], src_map: Dict, dst_map: Dict) -> torch.Tensor:
        """Convert ID pairs to edge indices using mapping."""
        mapped_pairs = []
        for src, dst in pairs:
            if src in src_map and dst in dst_map:
                mapped_pairs.append((src_map[src], dst_map[dst]))
        if mapped_pairs:
            return torch.tensor(mapped_pairs, dtype=torch.long).t().contiguous()
        else:
            return torch.empty((2, 0), dtype=torch.long)
    
    # HAS_FORM
    has_form_pairs = [(e["word_id"], e["form_id"]) for e in raw.get("has_form", [])]
    data["Word", "HAS_FORM", "Form"].edge_index = edge_index(has_form_pairs, word_id_map, form_id_map)
    
    # OF_WORD
    of_word_pairs = [(e["form_id"], e["word_id"]) for e in raw.get("of_word", [])]
    data["Form", "OF_WORD", "Word"].edge_index = edge_index(of_word_pairs, form_id_map, word_id_map)
    
    # HAS_SENSE
    has_sense_pairs = [(e["word_id"], e["definition_id"]) for e in raw.get("has_sense", [])]
    data["Word", "HAS_SENSE", "Sense"].edge_index = edge_index(has_sense_pairs, word_id_map, sense_id_map)
    
    # DERIVED_FROM
    derived_pairs = [(e["from_word_id"], e["to_word_id"]) for e in raw.get("relations", []) 
                     if e["relation_type"] == "derived_from"]
    data["Word", "DERIVED_FROM", "Word"].edge_index = edge_index(derived_pairs, word_id_map, word_id_map)
    
    # HAS_AFFIX
    has_affix_pairs = [(e["word_id"], e["morpheme_id"]) for e in raw.get("has_affix", [])]
    data["Word", "HAS_AFFIX", "Morpheme"].edge_index = edge_index(has_affix_pairs, word_id_map, morpheme_id_map)
    
    # RELATED (multi-relation: encode type as edge_attr)
    rel_pairs = [(e["from_word_id"], e["to_word_id"], e["relation_type"]) 
                 for e in raw.get("relations", []) 
                 if e["relation_type"].startswith("related_")]
    if rel_pairs:
        idx_pairs = [(word_id_map[a], word_id_map[b]) for a, b, _ in rel_pairs 
                     if a in word_id_map and b in word_id_map]
        rel_types = [r for a, b, r in rel_pairs if a in word_id_map and b in word_id_map]
        if idx_pairs:
            data["Word", "RELATED", "Word"].edge_index = torch.tensor(idx_pairs, dtype=torch.long).t().contiguous()
            data["Word", "RELATED", "Word"].edge_type = torch.tensor([hash(r) % 100 for r in rel_types], dtype=torch.long)
    
    # SHARES_PHONOLOGY
    phon_pairs = [(e["word1_id"], e["word2_id"]) for e in raw.get("shares_phon", [])]
    data["Word", "SHARES_PHONOLOGY", "Word"].edge_index = edge_index(phon_pairs, word_id_map, word_id_map)
    
    # SHARES_ETYMOLOGY
    etym_pairs = [(e["word1_id"], e["word2_id"]) for e in raw.get("shares_etym", [])]
    data["Word", "SHARES_ETYMOLOGY", "Word"].edge_index = edge_index(etym_pairs, word_id_map, word_id_map)
    
    # Add reverse edges for symmetric relations
    for edge_type in ["SHARES_PHONOLOGY", "SHARES_ETYMOLOGY"]:
        key = ("Word", edge_type, "Word")
        if key in data.edge_types and data[key].edge_index.size(1) > 0:
            edge_idx = data[key].edge_index
            reverse_idx = torch.stack([edge_idx[1], edge_idx[0]])
            data[key].edge_index = torch.cat([edge_idx, reverse_idx], dim=1)
    
    if device:
        data = data.to(device)
    
    return data

def split_edges(data: HeteroData, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[HeteroData, HeteroData, HeteroData]:
    """
    Split edges into train/val/test sets.
    Returns train_data, val_data, test_data.
    """
    import random
    
    train_data = HeteroData()
    val_data = HeteroData()
    test_data = HeteroData()
    
    # Copy node features
    for node_type in data.node_types:
        for key, value in data[node_type].items():
            train_data[node_type][key] = value
            val_data[node_type][key] = value
            test_data[node_type][key] = value
    
    # Split edges
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            continue
        
        # Shuffle edges
        perm = torch.randperm(num_edges)
        edge_index = edge_index[:, perm]
        
        # Calculate split points
        train_size = int(train_ratio * num_edges)
        val_size = int(val_ratio * num_edges)
        
        # Split
        train_edges = edge_index[:, :train_size]
        val_edges = edge_index[:, train_size:train_size + val_size]
        test_edges = edge_index[:, train_size + val_size:]
        
        # Assign to data objects
        train_data[edge_type].edge_index = train_edges
        val_data[edge_type].edge_index = val_edges
        test_data[edge_type].edge_index = test_edges
        
        # Copy edge attributes if they exist
        if hasattr(data[edge_type], 'edge_type'):
            edge_types = data[edge_type].edge_type[perm]
            train_data[edge_type].edge_type = edge_types[:train_size]
            val_data[edge_type].edge_type = edge_types[train_size:train_size + val_size]
            test_data[edge_type].edge_type = edge_types[train_size + val_size:]
    
    return train_data, val_data, test_data 