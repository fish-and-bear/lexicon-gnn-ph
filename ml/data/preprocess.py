"""
Preprocessing utilities for lexical data features.

This module contains functions for:
1. Text feature extraction (character n-grams, multilingual embeddings)
2. Feature normalization and transformation
3. Train/validation/test split creation with stratification
4. Negative sampling for link prediction
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import random
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# Import optional dependencies if installed
try:
    from transformers import XLMRobertaModel, XLMRobertaTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    
try:
    import fasttext
    import fasttext.util
    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False

# Setup logging
logger = logging.getLogger(__name__)

class LexicalFeatureExtractor:
    """Extract features from lexical data."""
    
    def __init__(self, 
                 use_xlmr: bool = True, 
                 use_fasttext: bool = True,
                 use_char_ngrams: bool = True,
                 use_phonetic_features: bool = True,
                 use_etymology_features: bool = True,
                 use_baybayin_features: bool = True,
                 normalize_features: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            use_xlmr: Whether to use XLM-RoBERTa embeddings
            use_fasttext: Whether to use fastText embeddings
            use_char_ngrams: Whether to use character n-grams
            use_phonetic_features: Whether to use phonetic similarity features
            use_etymology_features: Whether to use etymology features
            use_baybayin_features: Whether to use Baybayin script features
            normalize_features: Whether to normalize features
        """
        self.use_xlmr = use_xlmr and HAS_TRANSFORMERS
        self.use_fasttext = use_fasttext and HAS_FASTTEXT
        self.use_char_ngrams = use_char_ngrams
        self.use_phonetic_features = use_phonetic_features
        self.use_etymology_features = use_etymology_features
        self.use_baybayin_features = use_baybayin_features
        self.normalize_features = normalize_features
        
        # Initialize models if needed
        self.xlmr_model = None
        self.xlmr_tokenizer = None
        self.fasttext_model = None
        
        # Cache for phonetic encoders
        self.phonetic_encoders = {}
        
        # Initialize language-specific resources
        self.init_language_resources()
        
        if self.use_xlmr:
            try:
                logger.info("Loading XLM-RoBERTa model...")
                self.xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
                self.xlmr_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
                logger.info("XLM-RoBERTa model loaded")
            except Exception as e:
                logger.warning(f"Failed to load XLM-RoBERTa: {e}")
                self.use_xlmr = False
        
        if self.use_fasttext:
            try:
                logger.info("Loading fastText model...")
                # Download if needed (uncomment as needed)
                # fasttext.util.download_model('tl', if_exists='ignore')
                # self.fasttext_model = fasttext.load_model('cc.tl.300.bin')
                logger.info("Using dummy fastText model for demo")
                self.use_fasttext = False  # Disable for demo
            except Exception as e:
                logger.warning(f"Failed to load fastText: {e}")
                self.use_fasttext = False
    
    def init_language_resources(self):
        """Initialize language-specific resources."""
        # Philippine languages phonetic patterns
        self.ph_vowels = set('aeiou')
        self.ph_consonants = set('bcdfghjklmnpqrstvwxyz')
        
        # Baybayin character mappings (simplified)
        self.latin_to_baybayin = {
            'a': 'ᜀ', 'e': 'ᜁ', 'i': 'ᜁ', 'o': 'ᜂ', 'u': 'ᜂ',
            'ba': 'ᜊ', 'ka': 'ᜃ', 'da': 'ᜇ', 'ga': 'ᜄ',
            'ha': 'ᜑ', 'la': 'ᜎ', 'ma': 'ᜋ', 'na': 'ᜈ',
            'pa': 'ᜉ', 'sa': 'ᜐ', 'ta': 'ᜆ', 'wa': 'ᜏ', 'ya': 'ᜌ'
        }
    
    def extract_phonetic_features(self, texts: List[str], lang_codes: List[str]) -> torch.Tensor:
        """
        Extract phonetic features from text.
        
        Args:
            texts: List of strings to extract features from
            lang_codes: List of language codes
            
        Returns:
            Tensor of phonetic features
        """
        num_texts = len(texts)
        feature_dim = 20  # Adjust as needed
        features = torch.zeros(num_texts, feature_dim)
        
        for i, (text, lang) in enumerate(zip(texts, lang_codes)):
            if not text or pd.isna(text):
                continue
                
            # Get normalized text (lowercase, without punctuation)
            norm_text = ''.join(c.lower() for c in text if c.isalpha() or c.isspace())
            
            # Extract features (more can be added for production)
            vowel_count = sum(1 for c in norm_text if c in self.ph_vowels)
            consonant_count = sum(1 for c in norm_text if c in self.ph_consonants)
            word_length = len(norm_text)
            if word_length > 0:
                vowel_ratio = vowel_count / word_length
                consonant_ratio = consonant_count / word_length
            else:
                vowel_ratio = consonant_ratio = 0
                
            # Count specific phonetic patterns in Philippine languages
            features[i, 0] = float(vowel_count)
            features[i, 1] = float(consonant_count)
            features[i, 2] = float(vowel_ratio)
            features[i, 3] = float(consonant_ratio)
            
            # Count specific patterns by language
            if lang == 'tl':  # Tagalog
                # Count ng digraphs (common in Tagalog)
                ng_count = norm_text.count('ng')
                features[i, 4] = float(ng_count)
                
                # Count presence of stress markers (important in Tagalog)
                features[i, 5] = float(1 if "'" in text else 0)
            
            elif lang == 'ceb':  # Cebuano
                # Count y vowels (more common than in Tagalog)
                y_count = norm_text.count('y')
                features[i, 6] = float(y_count)
        
        return features
    
    def extract_baybayin_features(self, baybayin_texts: List[str], romanized_texts: List[str]) -> torch.Tensor:
        """
        Extract features from Baybayin script texts.
        
        Args:
            baybayin_texts: List of Baybayin script texts
            romanized_texts: List of romanized texts
            
        Returns:
            Tensor of Baybayin features
        """
        num_texts = len(baybayin_texts)
        feature_dim = 10  # Adjust as needed
        features = torch.zeros(num_texts, feature_dim)
        
        for i, (baybayin, roman) in enumerate(zip(baybayin_texts, romanized_texts)):
            # Skip empty texts
            if (not baybayin or pd.isna(baybayin)) and (not roman or pd.isna(roman)):
                continue
            
            has_baybayin = baybayin is not None and not pd.isna(baybayin) and len(baybayin) > 0
            has_roman = roman is not None and not pd.isna(roman) and len(roman) > 0
            
            # Basic features
            features[i, 0] = 1.0 if has_baybayin else 0.0
            features[i, 1] = 1.0 if has_roman else 0.0
            features[i, 2] = 1.0 if (has_baybayin and has_roman) else 0.0
            
            # Extract more features if Baybayin text is available
            if has_baybayin:
                # Count common Baybayin syllables
                features[i, 3] = float(sum(1 for c in baybayin if c in 'ᜀᜁᜂᜃᜄᜅ'))
                
                # Calculate basic length features
                features[i, 4] = float(len(baybayin))
                
                if has_roman:
                    # Calculate ratio of Baybayin to Roman lengths
                    features[i, 5] = float(len(baybayin)) / max(1, float(len(roman)))
        
        return features
    
    def extract_etymology_features(self, etymologies: Dict[int, List[Dict]]) -> torch.Tensor:
        """
        Extract features from etymology data.
        
        Args:
            etymologies: Dictionary mapping word IDs to lists of etymology dictionaries
            
        Returns:
            Tensor of etymology features
        """
        word_ids = sorted(etymologies.keys())
        num_words = len(word_ids)
        feature_dim = 15  # Adjust as needed
        features = torch.zeros(num_words, feature_dim)
        
        # Language family groups
        austronesian = {'tl', 'ceb', 'ilo', 'hil', 'bik', 'pag', 'war'}
        indo_european = {'en', 'es', 'fr', 'de', 'la', 'pt', 'it', 'nl'}
        sino_tibetan = {'zh', 'yue', 'nan'}
        
        for i, word_id in enumerate(word_ids):
            etym_list = etymologies.get(word_id, [])
            if not etym_list:
                continue
            
            # Basic feature: has etymology
            features[i, 0] = 1.0
            
            # Count number of etymologies
            features[i, 1] = float(len(etym_list))
            
            # Track language families
            has_austronesian = False
            has_indo_european = False
            has_sino_tibetan = False
            
            # Track language codes
            language_codes = []
            
            for etym in etym_list:
                # Extract language codes
                if 'language_codes' in etym and etym['language_codes']:
                    try:
                        if isinstance(etym['language_codes'], str):
                            langs = json.loads(etym['language_codes'])
                            if isinstance(langs, list):
                                language_codes.extend(langs)
                        elif isinstance(etym['language_codes'], list):
                            language_codes.extend(etym['language_codes'])
                    except:
                        pass
            
            # Check language families
            for lang in language_codes:
                if lang in austronesian:
                    has_austronesian = True
                elif lang in indo_european:
                    has_indo_european = True
                elif lang in sino_tibetan:
                    has_sino_tibetan = True
            
            # Set language family features
            features[i, 2] = 1.0 if has_austronesian else 0.0
            features[i, 3] = 1.0 if has_indo_european else 0.0
            features[i, 4] = 1.0 if has_sino_tibetan else 0.0
            
            # Count distinct source languages
            features[i, 5] = float(len(set(language_codes)))
            
            # Check for specific patterns like Spanish influence, etc.
            features[i, 6] = 1.0 if 'es' in language_codes else 0.0  # Spanish
            features[i, 7] = 1.0 if 'en' in language_codes else 0.0  # English
            features[i, 8] = 1.0 if 'zh' in language_codes else 0.0  # Chinese
            features[i, 9] = 1.0 if 'sn' in language_codes else 0.0  # Sanskrit
            
            # Try to extract year/century information
            # This is simplified - would need more sophisticated analysis in production
            etymology_text = " ".join([e.get('etymology_text', '') for e in etym_list if 'etymology_text' in e])
            has_date_info = any(str(year) in etymology_text for year in range(1500, 2024))
            features[i, 10] = 1.0 if has_date_info else 0.0
        
        return features
    
    def extract_char_ngrams(self, texts: List[str], n: int = 3) -> torch.Tensor:
        """
        Extract character n-gram features.
        
        Args:
            texts: List of strings to extract features from
            n: Size of n-grams (default: 3)
            
        Returns:
            Tensor of character n-gram features
        """
        # Create a vocabulary of character n-grams
        ngram_counter = Counter()
        for text in texts:
            if not text or pd.isna(text):
                continue
            padded = f"#{text}#"  # Add start/end markers
            for i in range(len(padded) - n + 1):
                ngram_counter[padded[i:i+n]] += 1
        
        # Keep most common n-grams (limit vocabulary size)
        max_ngrams = 10000
        vocab = {ngram: i for i, (ngram, _) in enumerate(ngram_counter.most_common(max_ngrams))}
        
        # Create feature vectors
        features = torch.zeros(len(texts), len(vocab))
        for i, text in enumerate(texts):
            if not text or pd.isna(text):
                continue
            padded = f"#{text}#"  # Add start/end markers
            for j in range(len(padded) - n + 1):
                ngram = padded[j:j+n]
                if ngram in vocab:
                    features[i, vocab[ngram]] += 1
        
        # Apply TF-IDF transformation (simplified)
        if len(texts) > 0:
            idf = torch.log(len(texts) / (1 + features.sum(dim=0)))
            features = features * idf
        
        logger.info(f"Created character {n}-gram features with {len(vocab)} dimensions")
        return features
    
    def extract_xlmr_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Extract XLM-RoBERTa embeddings.
        
        Args:
            texts: List of strings to extract features from
            
        Returns:
            Tensor of XLM-RoBERTa embeddings
        """
        if not self.use_xlmr or not self.xlmr_model:
            # Return dummy embeddings if XLM-RoBERTa is not available
            logger.warning("XLM-RoBERTa not available, using random embeddings")
            return torch.randn(len(texts), 768)
        
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                # Replace None/NaN values with empty strings
                batch_texts = [text if text and not pd.isna(text) else "" for text in batch_texts]
                
                inputs = self.xlmr_tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                )
                
                outputs = self.xlmr_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # Use [CLS] token
                embeddings.append(batch_embeddings)
        
        embeddings = torch.cat(embeddings, dim=0)
        logger.info(f"Created XLM-RoBERTa embeddings with shape {embeddings.shape}")
        return embeddings
    
    def extract_fasttext_embeddings(self, texts: List[str], lang_codes: Optional[List[str]] = None) -> torch.Tensor:
        """
        Extract fastText embeddings.
        
        Args:
            texts: List of strings to extract features from
            lang_codes: List of language codes (optional)
            
        Returns:
            Tensor of fastText embeddings
        """
        if not self.use_fasttext or not self.fasttext_model:
            # Return dummy embeddings if fastText is not available
            logger.warning("fastText not available, using random embeddings")
            return torch.randn(len(texts), 300)
        
        embedding_dim = 300  # Standard fastText dimension
        embeddings = torch.zeros(len(texts), embedding_dim)
        
        for i, text in enumerate(texts):
            if not text or pd.isna(text):
                embeddings[i] = torch.zeros(embedding_dim)
                continue
            
            # In a real implementation, language-specific models would be used
            # based on the lang_codes parameter
            vec = self.fasttext_model.get_word_vector(text.lower())
            embeddings[i] = torch.tensor(vec)
        
        logger.info(f"Created fastText embeddings with shape {embeddings.shape}")
        return embeddings
    
    def extract_pronunciation_features(self, pronunciations_df: pd.DataFrame) -> Dict[int, torch.Tensor]:
        """
        Extract features from pronunciation data.
        
        Args:
            pronunciations_df: DataFrame with pronunciation data
            
        Returns:
            Dictionary mapping word IDs to pronunciation feature tensors
        """
        # Group pronunciations by word_id
        word_to_prons = defaultdict(list)
        for _, row in pronunciations_df.iterrows():
            word_id = row['word_id']
            pron_type = row['type']
            value = row['value']
            
            word_to_prons[word_id].append({
                'type': pron_type,
                'value': value,
                'tags': row['tags'] if 'tags' in row and not pd.isna(row['tags']) else None
            })
        
        # Generate features for each word
        feature_dim = 10  # Adjust as needed
        features = {}
        
        for word_id, prons in word_to_prons.items():
            word_features = torch.zeros(feature_dim)
            
            # Basic feature: number of pronunciations
            word_features[0] = float(len(prons))
            
            # Features by type
            has_ipa = any(p['type'] == 'ipa' for p in prons)
            has_respelling = any(p['type'] == 'respelling_guide' for p in prons)
            
            word_features[1] = 1.0 if has_ipa else 0.0
            word_features[2] = 1.0 if has_respelling else 0.0
            
            # For IPA pronunciation, extract basic phonetic features
            ipa_prons = [p['value'] for p in prons if p['type'] == 'ipa']
            if ipa_prons:
                ipa_text = ipa_prons[0]
                # Example features (simplified for this example)
                vowel_count = sum(1 for c in ipa_text if c in 'aeiouəɪʊɔæɑɒɛɜːʌʏʉ')
                consonant_count = sum(1 for c in ipa_text if c not in 'aeiouəɪʊɔæɑɒɛɜːʌʏʉ ')
                
                word_features[3] = float(vowel_count)
                word_features[4] = float(consonant_count)
                word_features[5] = float(len(ipa_text))
                
                # Detect presence of specific Philippine language sounds
                word_features[6] = 1.0 if 'ŋ' in ipa_text else 0.0  # velar nasal (common in PH langs)
                
            features[word_id] = word_features
            
        logger.info(f"Created pronunciation features for {len(features)} words")
        return features
    
    def extract_all_features(self, lemmas: pd.DataFrame, 
                              definitions_df: Optional[pd.DataFrame] = None,
                              etymologies_df: Optional[pd.DataFrame] = None,
                              pronunciations_df: Optional[pd.DataFrame] = None,
                              word_forms_df: Optional[pd.DataFrame] = None) -> Dict[str, torch.Tensor]:
        """
        Extract all features from lexical data.
        
        Args:
            lemmas: DataFrame with lemma information
            definitions_df: DataFrame with definition information (optional)
            etymologies_df: DataFrame with etymology information (optional)
            pronunciations_df: DataFrame with pronunciation information (optional)
            word_forms_df: DataFrame with word form information (optional)
            
        Returns:
            Dictionary of feature tensors
        """
        features = {}
        
        # Get lists of text and language codes
        texts = lemmas['lemma'].fillna('').tolist()
        normalized_texts = lemmas['normalized_lemma'].fillna('').tolist()
        lang_codes = lemmas['language_code'].fillna('unknown').tolist()
        
        # Basic language and boolean features are assumed to be preprocessed by the adapter
        
        # Extract text-based features
        if self.use_char_ngrams:
            trigram_features = self.extract_char_ngrams(normalized_texts, n=3)
            fourgram_features = self.extract_char_ngrams(normalized_texts, n=4)
            features['char_trigram'] = trigram_features
            features['char_fourgram'] = fourgram_features
        
        if self.use_xlmr:
            xlmr_features = self.extract_xlmr_embeddings(texts)
            features['xlmr'] = xlmr_features
        
        if self.use_fasttext:
            fasttext_features = self.extract_fasttext_embeddings(texts, lang_codes)
            features['fasttext'] = fasttext_features
        
        # Extract phonetic features
        if self.use_phonetic_features:
            phonetic_features = self.extract_phonetic_features(texts, lang_codes)
            features['phonetic'] = phonetic_features
            
            # Process pronunciations if available
            if pronunciations_df is not None and len(pronunciations_df) > 0:
                pron_features_dict = self.extract_pronunciation_features(pronunciations_df)
                
                # Convert dict to tensor, filling with zeros for missing words
                pron_features = torch.zeros(len(lemmas), pron_features_dict[next(iter(pron_features_dict))].shape[0])
                for i, row in enumerate(lemmas.to_dict('records')):
                    word_id = row['id']
                    if word_id in pron_features_dict:
                        pron_features[i] = pron_features_dict[word_id]
                
                features['pronunciation'] = pron_features
        
        # Extract Baybayin script features
        if self.use_baybayin_features:
            baybayin_texts = lemmas['baybayin_form'].tolist()
            romanized_texts = lemmas['romanized_form'].tolist()
            
            baybayin_features = self.extract_baybayin_features(baybayin_texts, romanized_texts)
            features['baybayin'] = baybayin_features
        
        # Extract etymology features
        if self.use_etymology_features and etymologies_df is not None and len(etymologies_df) > 0:
            # Group etymologies by word_id
            etymology_dict = defaultdict(list)
            for _, row in etymologies_df.iterrows():
                word_id = row['word_id']
                if pd.isna(word_id):
                    continue
                    
                etymology_dict[word_id].append(row.to_dict())
            
            # Convert to tensor format
            word_ids = [row['id'] for _, row in lemmas.iterrows()]
            etym_features = self.extract_etymology_features({wid: etymology_dict.get(wid, []) for wid in word_ids})
            features['etymology'] = etym_features
        
        # Feature normalization
        if self.normalize_features:
            for feat_name, feat_tensor in features.items():
                # Apply L2 normalization
                if torch.norm(feat_tensor).item() > 0:
                    normalized = feat_tensor / (torch.norm(feat_tensor, dim=1, keepdim=True) + 1e-8)
                    features[feat_name] = normalized
        
        return features

def create_train_test_split(graph, test_size=0.1, val_size=0.1, stratify_by='relation_type', seed=42):
    """
    Create train/validation/test splits for link prediction.
    
    Args:
        graph: DGL heterogeneous graph
        test_size: Proportion of edges for testing
        val_size: Proportion of edges for validation
        stratify_by: How to stratify the splits ('relation_type' or 'language')
        seed: Random seed
        
    Returns:
        Dictionary with train/val/test edge masks for each relation type
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    edge_splits = {}
    
    # Process each relation type separately
    for rel in graph.etypes:
        num_edges = graph.num_edges(rel)
        
        if num_edges == 0:
            continue
        
        # Create a mask array for all edges
        all_edges = torch.arange(num_edges)
        
        # Split into train/val/test
        train_val, test = train_test_split(
            all_edges.numpy(), test_size=test_size, random_state=seed
        )
        
        train, val = train_test_split(
            train_val, test_size=val_size/(1-test_size), random_state=seed
        )
        
        # Create boolean masks
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        train_mask[torch.tensor(train)] = True
        val_mask[torch.tensor(val)] = True
        test_mask[torch.tensor(test)] = True
        
        edge_splits[rel] = {
            'train': train_mask,
            'val': val_mask,
            'test': test_mask
        }
        
        logger.info(f"Relation '{rel}': {train_mask.sum()} train, {val_mask.sum()} val, {test_mask.sum()} test edges")
    
    return edge_splits

def generate_negative_samples(graph, num_samples, exclude_existing=True):
    """
    Generate negative samples for link prediction.
    
    Args:
        graph: DGL heterogeneous graph
        num_samples: Number of negative samples per relation type
        exclude_existing: Whether to exclude existing edges
        
    Returns:
        Dictionary with negative sample edges for each relation type
    """
    negative_samples = {}
    
    # Process each relation type separately
    for rel in graph.etypes:
        if "rev" in rel:  # Skip reverse relations
            continue
        
        # Get existing edges as a set for efficient lookup
        src, dst = graph.edges(etype=rel)
        existing_edges = set(zip(src.tolist(), dst.tolist()))
        num_nodes = graph.num_nodes()
        
        # Find negative samples
        neg_samples = []
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(neg_samples) < num_samples and attempts < max_attempts:
            src_id = random.randint(0, num_nodes - 1)
            dst_id = random.randint(0, num_nodes - 1)
            
            if src_id == dst_id:  # Skip self-loops
                continue
            
            if exclude_existing and (src_id, dst_id) in existing_edges:
                continue
            
            neg_samples.append((src_id, dst_id))
            attempts += 1
        
        negative_samples[rel] = torch.tensor(neg_samples)
        
        logger.info(f"Generated {len(neg_samples)} negative samples for relation '{rel}'")
    
    return negative_samples

def run_preprocessing(config_path=None):
    """
    Run the full preprocessing pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Preprocessed graph and features
    """
    from .db_adapter import LexicalDatabaseAdapter
    
    # Initialize adapter and extract graph
    adapter = LexicalDatabaseAdapter(config_path)
    
    # Fetch all necessary data
    lemmas_df = adapter.fetch_lemmas()
    relations_df = adapter.fetch_relations()
    definitions_df = adapter.fetch_definitions()
    etymologies_df = adapter.fetch_etymologies()
    pronunciations_df = adapter.fetch_pronunciations()
    word_forms_df = adapter.fetch_word_forms()
    
    # Try to fetch optional data
    try:
        definition_examples_df = adapter.fetch_definition_examples()
    except:
        definition_examples_df = pd.DataFrame()
    
    try:
        definition_categories_df = adapter.fetch_definition_categories()
    except:
        definition_categories_df = pd.DataFrame()
    
    logger.info("Building heterogeneous graph...")
    graph = adapter.build_heterogeneous_graph()
    
    # Extract features
    logger.info("Extracting features...")
    feature_extractor = LexicalFeatureExtractor()
    features = feature_extractor.extract_all_features(
        lemmas_df, 
        definitions_df, 
        etymologies_df, 
        pronunciations_df, 
        word_forms_df
    )
    
    # Add features to graph
    for feat_name, feat_tensor in features.items():
        graph.ndata[feat_name] = feat_tensor
    
    # Create train/val/test splits
    logger.info("Creating data splits...")
    edge_splits = create_train_test_split(graph)
    
    # Generate negative samples
    logger.info("Generating negative samples...")
    negative_samples = generate_negative_samples(graph, 1000)
    
    # Get node labels (e.g., POS tags)
    logger.info("Extracting node labels...")
    node_labels = adapter.get_node_labels()
    for label_name, label_tensor in node_labels.items():
        graph.ndata[label_name] = label_tensor
    
    # Close the database connection
    adapter.close()
    
    return {
        'graph': graph,
        'features': features,
        'edge_splits': edge_splits,
        'negative_samples': negative_samples,
        'node_labels': node_labels,
        'adapter': adapter  # Return the adapter for later use
    }

if __name__ == "__main__":
    # Run the preprocessing pipeline as a script
    result = run_preprocessing()
    print(f"Preprocessing complete. Graph has {result['graph'].num_nodes()} nodes and {result['graph'].num_edges()} edges.") 