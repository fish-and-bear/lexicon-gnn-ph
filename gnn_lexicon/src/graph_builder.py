"""
Graph construction for Philippine Lexicon GNN.
Builds a PyG HeteroData object from DB or JSON.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
from torch_geometric.data import HeteroData
from .utils import char_cnn_embed, normalize_numeric

NodeTypes = [
    "Word", "Form", "Definition", "PartOfSpeech", "Etymology", 
    "Pronunciation", "WordTemplate", "DefinitionExample", 
    "DefinitionCategory", "DefinitionLink", "Language"
]
EdgeTypes = [
    # Word-Form relationships
    ("Word", "HAS_FORM", "Form"),
    ("Form", "OF_WORD", "Word"),
    
    # Word-Definition relationships
    ("Word", "HAS_DEFINITION", "Definition"),
    
    # Word-PartOfSpeech relationships
    ("Word", "HAS_PART_OF_SPEECH", "PartOfSpeech"),
    
    # Word-Etymology relationships
    ("Word", "HAS_ETYMOLOGY", "Etymology"),
    
    # Word-Pronunciation relationships
    ("Word", "HAS_PRONUNCIATION", "Pronunciation"),
    
    # Word-Template relationships
    ("Word", "HAS_TEMPLATE", "WordTemplate"),
    
    # Definition-Example relationships
    ("Definition", "HAS_EXAMPLE", "DefinitionExample"),
    
    # Definition-Category relationships
    ("Definition", "HAS_CATEGORY", "DefinitionCategory"),
    
    # Definition-Link relationships
    ("Definition", "HAS_LINK", "DefinitionLink"),
    
    # Word-Language relationships
    ("Word", "IN_LANGUAGE", "Language"),
    
    # Root word relationships
    ("Word", "ROOT_OF", "Word"),
    
    # Semantic relationships (Word to Word)
    ("Word", "SYNONYM_OF", "Word"),
    ("Word", "ANTONYM_OF", "Word"),
    ("Word", "RELATED_TO", "Word"),
    ("Word", "DERIVED_FROM", "Word"),
    ("Word", "COGNATE_OF", "Word"),
    ("Word", "TRANSLATION_OF", "Word"),
    ("Word", "SEE_ALSO", "Word"),
    ("Word", "VARIANT_OF", "Word"),
    ("Word", "DOUBLET_OF", "Word"),
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
        data["Word"].node_id = torch.tensor([w["id"] for w in raw["words"]], dtype=torch.long)
        # Add additional word features
        if "is_proper_noun" in raw["words"][0]:
            data["Word"].is_proper_noun = torch.tensor([w.get("is_proper_noun", False) for w in raw["words"]], dtype=torch.bool)
        if "is_abbreviation" in raw["words"][0]:
            data["Word"].is_abbreviation = torch.tensor([w.get("is_abbreviation", False) for w in raw["words"]], dtype=torch.bool)
        if "is_initialism" in raw["words"][0]:
            data["Word"].is_initialism = torch.tensor([w.get("is_initialism", False) for w in raw["words"]], dtype=torch.bool)
    
    # Word Forms
    if raw["forms"]:
        form_texts = [f["form"] for f in raw["forms"]]
        data["Form"].x = char_cnn_embed(form_texts)
        data["Form"].node_id = torch.tensor([f["id"] for f in raw["forms"]], dtype=torch.long)
        if "is_canonical" in raw["forms"][0]:
            data["Form"].is_canonical = torch.tensor([f.get("is_canonical", False) for f in raw["forms"]], dtype=torch.bool)
        if "is_primary" in raw["forms"][0]:
            data["Form"].is_primary = torch.tensor([f.get("is_primary", False) for f in raw["forms"]], dtype=torch.bool)
    
    # Definitions
    if raw["definitions"]:
        def_texts = [d["definition_text"] for d in raw["definitions"]]
        data["Definition"].x = char_cnn_embed(def_texts)
        data["Definition"].node_id = torch.tensor([d["id"] for d in raw["definitions"]], dtype=torch.long)
        if "original_pos" in raw["definitions"][0]:
            # Create POS embeddings
            pos_texts = [d.get("original_pos", "unknown") for d in raw["definitions"]]
            data["Definition"].pos_embedding = char_cnn_embed(pos_texts)
    
    # Parts of Speech
    if raw["parts_of_speech"]:
        pos_codes = [p["code"] for p in raw["parts_of_speech"]]
        data["PartOfSpeech"].x = char_cnn_embed(pos_codes)
        data["PartOfSpeech"].node_id = torch.tensor([p["id"] for p in raw["parts_of_speech"]], dtype=torch.long)
    
    # Etymologies
    if raw["etymologies"]:
        etym_texts = [e["etymology_text"] for e in raw["etymologies"]]
        data["Etymology"].x = char_cnn_embed(etym_texts)
        data["Etymology"].node_id = torch.tensor([e["id"] for e in raw["etymologies"]], dtype=torch.long)
    
    # Pronunciations
    if raw["pronunciations"]:
        pron_values = [p["value"] for p in raw["pronunciations"]]
        data["Pronunciation"].x = char_cnn_embed(pron_values)
        data["Pronunciation"].node_id = torch.tensor([p["id"] for p in raw["pronunciations"]], dtype=torch.long)
        if "type" in raw["pronunciations"][0]:
            pron_types = [p.get("type", "unknown") for p in raw["pronunciations"]]
            data["Pronunciation"].type_embedding = char_cnn_embed(pron_types)
    
    # Word Templates
    if raw["word_templates"]:
        template_names = [t["template_name"] for t in raw["word_templates"]]
        data["WordTemplate"].x = char_cnn_embed(template_names)
        data["WordTemplate"].node_id = torch.tensor([t["id"] for t in raw["word_templates"]], dtype=torch.long)
    
    # Definition Examples
    if raw["definition_examples"]:
        example_texts = [ex["example_text"] for ex in raw["definition_examples"]]
        data["DefinitionExample"].x = char_cnn_embed(example_texts)
        data["DefinitionExample"].node_id = torch.tensor([ex["id"] for ex in raw["definition_examples"]], dtype=torch.long)
    
    # Definition Categories
    if raw["definition_categories"]:
        category_names = [cat["category_name"] for cat in raw["definition_categories"]]
        data["DefinitionCategory"].x = char_cnn_embed(category_names)
        data["DefinitionCategory"].node_id = torch.tensor([cat["id"] for cat in raw["definition_categories"]], dtype=torch.long)
    
    # Definition Links
    if raw["definition_links"]:
        link_texts = [link["link_text"] for link in raw["definition_links"]]
        data["DefinitionLink"].x = char_cnn_embed(link_texts)
        data["DefinitionLink"].node_id = torch.tensor([link["id"] for link in raw["definition_links"]], dtype=torch.long)
    
    # Languages
    if raw["languages"]:
        lang_codes = [l["code"] for l in raw["languages"]]
        data["Language"].x = char_cnn_embed(lang_codes)
        data["Language"].node_id = torch.tensor([l["id"] for l in raw["languages"]], dtype=torch.long)
    
    # --- Edges ---
    # Create word-to-word mappings
    word_to_idx = {w["id"]: i for i, w in enumerate(raw["words"])}
    form_to_idx = {f["id"]: i for i, f in enumerate(raw["forms"])}
    def_to_idx = {d["id"]: i for i, d in enumerate(raw["definitions"])}
    pos_to_idx = {p["id"]: i for i, p in enumerate(raw["parts_of_speech"])}
    etym_to_idx = {e["id"]: i for i, e in enumerate(raw["etymologies"])}
    pron_to_idx = {p["id"]: i for i, p in enumerate(raw["pronunciations"])}
    template_to_idx = {t["id"]: i for i, t in enumerate(raw["word_templates"])}
    example_to_idx = {ex["id"]: i for i, ex in enumerate(raw["definition_examples"])}
    category_to_idx = {cat["id"]: i for i, cat in enumerate(raw["definition_categories"])}
    link_to_idx = {link["id"]: i for i, link in enumerate(raw["definition_links"])}
    lang_to_idx = {l["id"]: i for i, l in enumerate(raw["languages"])}
    
    # Word-Form relationships
    if raw["has_form"]:
        has_form_edges = []
        for edge in raw["has_form"]:
            if edge["word_id"] in word_to_idx and edge["form_id"] in form_to_idx:
                has_form_edges.append([word_to_idx[edge["word_id"]], form_to_idx[edge["form_id"]]])
        if has_form_edges:
            data["Word", "HAS_FORM", "Form"].edge_index = torch.tensor(has_form_edges, dtype=torch.long).t()
    
    if raw["of_word"]:
        of_word_edges = []
        for edge in raw["of_word"]:
            if edge["form_id"] in form_to_idx and edge["word_id"] in word_to_idx:
                of_word_edges.append([form_to_idx[edge["form_id"]], word_to_idx[edge["word_id"]]])
        if of_word_edges:
            data["Form", "OF_WORD", "Word"].edge_index = torch.tensor(of_word_edges, dtype=torch.long).t()
    
    # Word-Definition relationships
    if raw["has_definition"]:
        has_def_edges = []
        for edge in raw["has_definition"]:
            if edge["word_id"] in word_to_idx and edge["definition_id"] in def_to_idx:
                has_def_edges.append([word_to_idx[edge["word_id"]], def_to_idx[edge["definition_id"]]])
        if has_def_edges:
            data["Word", "HAS_DEFINITION", "Definition"].edge_index = torch.tensor(has_def_edges, dtype=torch.long).t()
    
    # Word-PartOfSpeech relationships
    if raw["has_pos"]:
        has_pos_edges = []
        for edge in raw["has_pos"]:
            if edge["word_id"] in word_to_idx and edge["pos_id"] in pos_to_idx:
                has_pos_edges.append([word_to_idx[edge["word_id"]], pos_to_idx[edge["pos_id"]]])
        if has_pos_edges:
            data["Word", "HAS_PART_OF_SPEECH", "PartOfSpeech"].edge_index = torch.tensor(has_pos_edges, dtype=torch.long).t()
    
    # Word-Etymology relationships
    if raw["has_etymology"]:
        has_etym_edges = []
        for edge in raw["has_etymology"]:
            if edge["word_id"] in word_to_idx and edge["etymology_id"] in etym_to_idx:
                has_etym_edges.append([word_to_idx[edge["word_id"]], etym_to_idx[edge["etymology_id"]]])
        if has_etym_edges:
            data["Word", "HAS_ETYMOLOGY", "Etymology"].edge_index = torch.tensor(has_etym_edges, dtype=torch.long).t()
    
    # Word-Pronunciation relationships
    if raw["has_pronunciation"]:
        has_pron_edges = []
        for edge in raw["has_pronunciation"]:
            if edge["word_id"] in word_to_idx and edge["pronunciation_id"] in pron_to_idx:
                has_pron_edges.append([word_to_idx[edge["word_id"]], pron_to_idx[edge["pronunciation_id"]]])
        if has_pron_edges:
            data["Word", "HAS_PRONUNCIATION", "Pronunciation"].edge_index = torch.tensor(has_pron_edges, dtype=torch.long).t()
    
    # Word-Template relationships
    if raw["has_template"]:
        has_template_edges = []
        for edge in raw["has_template"]:
            if edge["word_id"] in word_to_idx and edge["template_id"] in template_to_idx:
                has_template_edges.append([word_to_idx[edge["word_id"]], template_to_idx[edge["template_id"]]])
        if has_template_edges:
            data["Word", "HAS_TEMPLATE", "WordTemplate"].edge_index = torch.tensor(has_template_edges, dtype=torch.long).t()
    
    # Definition-Example relationships
    if raw["has_example"]:
        has_example_edges = []
        for edge in raw["has_example"]:
            if edge["definition_id"] in def_to_idx and edge["example_id"] in example_to_idx:
                has_example_edges.append([def_to_idx[edge["definition_id"]], example_to_idx[edge["example_id"]]])
        if has_example_edges:
            data["Definition", "HAS_EXAMPLE", "DefinitionExample"].edge_index = torch.tensor(has_example_edges, dtype=torch.long).t()
    
    # Definition-Category relationships
    if raw["has_category"]:
        has_category_edges = []
        for edge in raw["has_category"]:
            if edge["definition_id"] in def_to_idx and edge["category_id"] in category_to_idx:
                has_category_edges.append([def_to_idx[edge["definition_id"]], category_to_idx[edge["category_id"]]])
        if has_category_edges:
            data["Definition", "HAS_CATEGORY", "DefinitionCategory"].edge_index = torch.tensor(has_category_edges, dtype=torch.long).t()
    
    # Definition-Link relationships
    if raw["has_link"]:
        has_link_edges = []
        for edge in raw["has_link"]:
            if edge["definition_id"] in def_to_idx and edge["link_id"] in link_to_idx:
                has_link_edges.append([def_to_idx[edge["definition_id"]], link_to_idx[edge["link_id"]]])
        if has_link_edges:
            data["Definition", "HAS_LINK", "DefinitionLink"].edge_index = torch.tensor(has_link_edges, dtype=torch.long).t()
    
    # Word-Language relationships
    if raw["in_language"]:
        in_lang_edges = []
        for edge in raw["in_language"]:
            if edge["word_id"] in word_to_idx and edge["language_id"] in lang_to_idx:
                in_lang_edges.append([word_to_idx[edge["word_id"]], lang_to_idx[edge["language_id"]]])
        if in_lang_edges:
            data["Word", "IN_LANGUAGE", "Language"].edge_index = torch.tensor(in_lang_edges, dtype=torch.long).t()
    
    # Root word relationships (self-referential)
    if raw["root_of"]:
        root_edges = []
        for edge in raw["root_of"]:
            if edge["root_id"] in word_to_idx and edge["word_id"] in word_to_idx:
                root_edges.append([word_to_idx[edge["root_id"]], word_to_idx[edge["word_id"]]])
        if root_edges:
            data["Word", "ROOT_OF", "Word"].edge_index = torch.tensor(root_edges, dtype=torch.long).t()
    
    # Semantic relationships (Word to Word)
    semantic_relations = [
        ("synonym_of", raw["synonym_of"]),
        ("antonym_of", raw["antonym_of"]),
        ("related_to", raw["related_to"]),
        ("derived_from", raw["derived_from"]),
        ("cognate_of", raw["cognate_of"]),
        ("translation_of", raw["translation_of"]),
        ("see_also", raw["see_also"]),
        ("variant_of", raw["variant_of"]),
        ("doublet_of", raw["doublet_of"]),
    ]
    
    for rel_name, rel_edges in semantic_relations:
        if rel_edges:
            edges = []
            for edge in rel_edges:
                if edge["word1_id"] in word_to_idx and edge["word2_id"] in word_to_idx:
                    edges.append([word_to_idx[edge["word1_id"]], word_to_idx[edge["word2_id"]]])
            if edges:
                data["Word", rel_name.upper(), "Word"].edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Move to device if specified
    if device is not None:
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