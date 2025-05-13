"""
Lexical Graph Builder for Filipino language resources.

This module constructs a heterogeneous graph from the lexical database,
creating a rich knowledge graph structure for GNN processing.
"""

import logging
import pandas as pd
import numpy as np
import torch
import dgl
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class LexicalGraphBuilder:
    """
    Builds a heterogeneous graph from lexical database contents, 
    optimized for low-resource Philippine languages.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 target_languages: Optional[List[str]] = None):
        """
        Initialize the graph builder.
        
        Args:
            config: Configuration dictionary
            target_languages: Optional list of language codes to focus on
        """
        self.config = config
        self.target_languages = target_languages or config.get('data', {}).get('target_languages', [])
        
        # Node type mappings
        self.word_id_to_node = {}
        self.def_id_to_node = {}
        self.etym_id_to_node = {}
        
        # Reverse mappings
        self.node_to_word_id = {}
        self.node_to_def_id = {}
        self.node_to_etym_id = {}
        
        # Track relation statistics
        self.relation_counts = defaultdict(int)

    def build_graph(self,
                    lemmas_df: pd.DataFrame,
                    relations_df: pd.DataFrame,
                    definitions_df: pd.DataFrame,
                    etymologies_df: pd.DataFrame,
                    pronunciations_df: Optional[pd.DataFrame] = None,
                    word_forms_df: Optional[pd.DataFrame] = None,
                    affixations_df: Optional[pd.DataFrame] = None,
                    definition_links_df: Optional[pd.DataFrame] = None) -> dgl.DGLGraph:
        """
        Build a heterogeneous graph from the lexical data.
        
        Args:
            lemmas_df: DataFrame with word lemma information
            relations_df: DataFrame with word-to-word relations
            definitions_df: DataFrame with word definitions
            etymologies_df: DataFrame with etymology information
            pronunciations_df: Optional DataFrame with pronunciation information
            word_forms_df: Optional DataFrame with word form variations
            affixations_df: Optional DataFrame with affixation information
            definition_links_df: Optional DataFrame with definition relations
            
        Returns:
            A DGL heterogeneous graph
        """
        logger.info("Building heterogeneous graph from lexical data")
        
        # Filter by target languages if specified
        if self.target_languages:
            logger.info(f"Filtering to languages: {self.target_languages}")
            lemmas_df = lemmas_df[lemmas_df['language_code'].isin(self.target_languages)]
            logger.info(f"Filtered to {len(lemmas_df)} lemmas")
            
            # Filter related tables based on the filtered lemmas
            word_ids = set(lemmas_df['id'])
            definitions_df = definitions_df[definitions_df['word_id'].isin(word_ids)]
            etymologies_df = etymologies_df[etymologies_df['word_id'].isin(word_ids)]
            relations_df = relations_df[
                (relations_df['from_word_id'].isin(word_ids)) & 
                (relations_df['to_word_id'].isin(word_ids))
            ]
            
            if pronunciations_df is not None:
                pronunciations_df = pronunciations_df[pronunciations_df['word_id'].isin(word_ids)]
            
            if word_forms_df is not None:
                word_forms_df = word_forms_df[word_forms_df['word_id'].isin(word_ids)]
            
            if affixations_df is not None:
                affixations_df = affixations_df[
                    (affixations_df['root_word_id'].isin(word_ids)) & 
                    (affixations_df['affixed_word_id'].isin(word_ids))
                ]
        
        # Create node ID mappings
        self._create_node_mappings(lemmas_df, definitions_df, etymologies_df)
        
        # Create heterogeneous graph data structure
        graph_data = {}
        
        # Add word-to-word relations (heterogeneous edge types)
        logger.info("Adding word-to-word relations")
        self._add_word_relations(graph_data, relations_df)
        
        # Add word-to-definition relations
        logger.info("Adding word-to-definition relations")
        self._add_definition_relations(graph_data, definitions_df)
        
        # Add word-to-etymology relations
        logger.info("Adding word-to-etymology relations")
        self._add_etymology_relations(graph_data, etymologies_df)
        
        # Add affixations if available
        if affixations_df is not None and not affixations_df.empty:
            logger.info("Adding affixation relations")
            self._add_affixation_relations(graph_data, affixations_df)
        
        # Create the heterogeneous graph
        hg = dgl.heterograph(graph_data)
        
        # Log graph statistics
        self._log_graph_statistics(hg)
        
        return hg
    
    def _create_node_mappings(self, 
                            lemmas_df: pd.DataFrame, 
                            definitions_df: pd.DataFrame,
                            etymologies_df: pd.DataFrame):
        """
        Create mappings between database IDs and node indices.
        
        Args:
            lemmas_df: DataFrame with word lemmas
            definitions_df: DataFrame with definitions
            etymologies_df: DataFrame with etymologies
        """
        # Create word node mappings
        for i, row in lemmas_df.iterrows():
            word_id = row['id']
            node_id = len(self.word_id_to_node)
            self.word_id_to_node[word_id] = node_id
            self.node_to_word_id[node_id] = word_id
        
        # Create definition node mappings
        for i, row in definitions_df.iterrows():
            def_id = row['id']
            node_id = len(self.def_id_to_node)
            self.def_id_to_node[def_id] = node_id
            self.node_to_def_id[node_id] = def_id
        
        # Create etymology node mappings
        for i, row in etymologies_df.iterrows():
            etym_id = row['id']
            node_id = len(self.etym_id_to_node)
            self.etym_id_to_node[etym_id] = node_id
            self.node_to_etym_id[node_id] = etym_id
        
        logger.info(f"Created mappings for {len(self.word_id_to_node)} words, "
                   f"{len(self.def_id_to_node)} definitions, and {len(self.etym_id_to_node)} etymologies")
    
    def _add_word_relations(self, graph_data: Dict, relations_df: pd.DataFrame):
        """
        Add word-to-word relations to the graph data.
        
        Args:
            graph_data: Graph data dictionary
            relations_df: DataFrame with relations
        """
        # Group by relation type for efficiency
        grouped = relations_df.groupby('relation_type')
        
        for rel_type, group in grouped:
            src_ids = []
            dst_ids = []
            
            for _, row in group.iterrows():
                from_id = row['from_word_id']
                to_id = row['to_word_id']
                
                # Skip if nodes are not in our mappings
                if from_id not in self.word_id_to_node or to_id not in self.word_id_to_node:
                    continue
                
                src_node = self.word_id_to_node[from_id]
                dst_node = self.word_id_to_node[to_id]
                
                src_ids.append(src_node)
                dst_ids.append(dst_node)
                
                # Track relation statistics
                self.relation_counts[rel_type] += 1
            
            if src_ids:
                # Create edge type (word, relation_type, word)
                edge_type = ('word', rel_type, 'word')
                graph_data[edge_type] = (torch.tensor(src_ids), torch.tensor(dst_ids))
                
                logger.debug(f"Added {len(src_ids)} edges for relation type {rel_type}")
    
    def _add_definition_relations(self, graph_data: Dict, definitions_df: pd.DataFrame):
        """
        Add word-to-definition relations to the graph data.
        
        Args:
            graph_data: Graph data dictionary
            definitions_df: DataFrame with definitions
        """
        src_ids = []
        dst_ids = []
        def_to_word_src = []
        def_to_word_dst = []
        
        for _, row in definitions_df.iterrows():
            word_id = row['word_id']
            def_id = row['id']
            
            # Skip if nodes are not in our mappings
            if word_id not in self.word_id_to_node or def_id not in self.def_id_to_node:
                continue
            
            word_node = self.word_id_to_node[word_id]
            def_node = self.def_id_to_node[def_id]
            
            # Word -> Definition edge
            src_ids.append(word_node)
            dst_ids.append(def_node)
            
            # Definition -> Word edge (bidirectional)
            def_to_word_src.append(def_node)
            def_to_word_dst.append(word_node)
        
        if src_ids:
            # Create edge types
            graph_data[('word', 'has_definition', 'definition')] = (torch.tensor(src_ids), torch.tensor(dst_ids))
            graph_data[('definition', 'defines', 'word')] = (torch.tensor(def_to_word_src), torch.tensor(def_to_word_dst))
            
            logger.debug(f"Added {len(src_ids)} word-definition edges")
    
    def _add_etymology_relations(self, graph_data: Dict, etymologies_df: pd.DataFrame):
        """
        Add word-to-etymology relations to the graph data.
        
        Args:
            graph_data: Graph data dictionary
            etymologies_df: DataFrame with etymologies
        """
        src_ids = []
        dst_ids = []
        etym_to_word_src = []
        etym_to_word_dst = []
        
        for _, row in etymologies_df.iterrows():
            word_id = row['word_id']
            etym_id = row['id']
            
            # Skip if nodes are not in our mappings
            if word_id not in self.word_id_to_node or etym_id not in self.etym_id_to_node:
                continue
            
            word_node = self.word_id_to_node[word_id]
            etym_node = self.etym_id_to_node[etym_id]
            
            # Word -> Etymology edge
            src_ids.append(word_node)
            dst_ids.append(etym_node)
            
            # Etymology -> Word edge (bidirectional)
            etym_to_word_src.append(etym_node)
            etym_to_word_dst.append(word_node)
        
        if src_ids:
            # Create edge types
            graph_data[('word', 'has_etymology', 'etymology')] = (torch.tensor(src_ids), torch.tensor(dst_ids))
            graph_data[('etymology', 'is_etymology_of', 'word')] = (torch.tensor(etym_to_word_src), torch.tensor(etym_to_word_dst))
            
            logger.debug(f"Added {len(src_ids)} word-etymology edges")
    
    def _add_affixation_relations(self, graph_data: Dict, affixations_df: pd.DataFrame):
        """
        Add affixation relations to the graph data.
        
        Args:
            graph_data: Graph data dictionary
            affixations_df: DataFrame with affixations
        """
        # Group by affixation type for efficiency
        grouped = affixations_df.groupby('affix_type')
        
        for affix_type, group in grouped:
            src_ids = []
            dst_ids = []
            
            for _, row in group.iterrows():
                root_id = row['root_word_id']
                affixed_id = row['affixed_word_id']
                
                # Skip if nodes are not in our mappings
                if root_id not in self.word_id_to_node or affixed_id not in self.word_id_to_node:
                    continue
                
                root_node = self.word_id_to_node[root_id]
                affixed_node = self.word_id_to_node[affixed_id]
                
                src_ids.append(root_node)
                dst_ids.append(affixed_node)
                
                # Track relation statistics
                self.relation_counts[f"affix_{affix_type}"] += 1
            
            if src_ids:
                # Create edge type (word, affix_type, word)
                edge_type = ('word', f"affix_{affix_type}", 'word')
                graph_data[edge_type] = (torch.tensor(src_ids), torch.tensor(dst_ids))
                
                logger.debug(f"Added {len(src_ids)} edges for affixation type {affix_type}")
    
    def _log_graph_statistics(self, graph: dgl.DGLGraph):
        """
        Log statistics about the constructed graph.
        
        Args:
            graph: The constructed heterogeneous graph
        """
        logger.info("Heterogeneous graph statistics:")
        logger.info(f"Node types: {graph.ntypes}")
        logger.info(f"Edge types: {graph.etypes}")
        logger.info(f"Canonical edge types: {graph.canonical_etypes}")
        
        for ntype in graph.ntypes:
            logger.info(f"Nodes of type '{ntype}': {graph.num_nodes(ntype)}")
        
        for etype in graph.canonical_etypes:
            logger.info(f"Edges of type '{etype}': {graph.num_edges(etype)}")
        
        logger.info("Relation counts:")
        for rel_type, count in sorted(self.relation_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {rel_type}: {count}")
    
    def get_node_mappings(self) -> Dict[str, Dict]:
        """
        Get the mappings between database IDs and node indices.
        
        Returns:
            Dictionary with node mappings
        """
        return {
            'word_id_to_node': self.word_id_to_node,
            'def_id_to_node': self.def_id_to_node,
            'etym_id_to_node': self.etym_id_to_node,
            'node_to_word_id': self.node_to_word_id,
            'node_to_def_id': self.node_to_def_id,
            'node_to_etym_id': self.node_to_etym_id
        }
    
    def save_node_mappings(self, filepath: str):
        """
        Save node mappings to a JSON file.
        
        Args:
            filepath: Path to save the mappings
        """
        # Convert dictionaries with integer keys to use string keys for JSON serialization
        serializable_mappings = {
            'word_id_to_node': {str(k): v for k, v in self.word_id_to_node.items()},
            'def_id_to_node': {str(k): v for k, v in self.def_id_to_node.items()},
            'etym_id_to_node': {str(k): v for k, v in self.etym_id_to_node.items()},
            'node_to_word_id': {str(k): v for k, v in self.node_to_word_id.items()},
            'node_to_def_id': {str(k): v for k, v in self.node_to_def_id.items()},
            'node_to_etym_id': {str(k): v for k, v in self.node_to_etym_id.items()},
            'relation_counts': {k: v for k, v in self.relation_counts.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_mappings, f, indent=2)
        
        logger.info(f"Node mappings saved to {filepath}")
    
    def load_node_mappings(self, filepath: str):
        """
        Load node mappings from a JSON file.
        
        Args:
            filepath: Path to the mappings file
        """
        with open(filepath, 'r') as f:
            mappings = json.load(f)
        
        self.word_id_to_node = {int(k): v for k, v in mappings['word_id_to_node'].items()}
        self.def_id_to_node = {int(k): v for k, v in mappings['def_id_to_node'].items()}
        self.etym_id_to_node = {int(k): v for k, v in mappings['etym_id_to_node'].items()}
        self.node_to_word_id = {int(k): v for k, v in mappings['node_to_word_id'].items()}
        self.node_to_def_id = {int(k): v for k, v in mappings['node_to_def_id'].items()}
        self.node_to_etym_id = {int(k): v for k, v in mappings['node_to_etym_id'].items()}
        
        if 'relation_counts' in mappings:
            self.relation_counts = defaultdict(int, mappings['relation_counts'])
        
        logger.info(f"Node mappings loaded from {filepath}") 