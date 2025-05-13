"""
Feature extraction utilities for lexical data.

This module provides classes and functions for extracting features from lexical data
to be used in machine learning models, particularly for the heterogeneous graph neural network.
"""

import logging
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Set
import re
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Setup logging
logger = logging.getLogger(__name__)

class LexicalFeatureExtractor:
    """Extract features from lexical data."""
    
    def __init__(self, 
                 use_xlmr: bool = True, 
                 use_transformer_embeddings: bool = True,
                 use_char_ngrams: bool = True,
                 use_phonetic_features: bool = True,
                 use_etymology_features: bool = True,
                 use_baybayin_features: bool = True,
                 normalize_features: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            use_xlmr: Whether to use XLM-RoBERTa embeddings
            use_transformer_embeddings: Whether to use SentenceTransformer embeddings
            use_char_ngrams: Whether to use character n-grams
            use_phonetic_features: Whether to use phonetic similarity features
            use_etymology_features: Whether to use etymology features
            use_baybayin_features: Whether to use Baybayin script features
            normalize_features: Whether to normalize features
        """
        self.use_xlmr = use_xlmr
        self.use_transformer_embeddings = use_transformer_embeddings
        self.use_char_ngrams = use_char_ngrams
        self.use_phonetic_features = use_phonetic_features
        self.use_etymology_features = use_etymology_features
        self.use_baybayin_features = use_baybayin_features
        self.normalize_features = normalize_features
        
        # Initialize models if needed
        self.xlmr_model = None
        self.xlmr_tokenizer = None
        self.transformer_model = None
        self.transformer_tokenizer = None
        
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
        
        if self.use_transformer_embeddings:
            try:
                logger.info("Loading SentenceTransformer model...")
                self.transformer_model = SentenceTransformer("meedan/paraphrase-filipino-mpnet-base-v2")
                self.transformer_tokenizer = AutoTokenizer.from_pretrained("meedan/paraphrase-filipino-mpnet-base-v2")
                self.transformer_embedding_dim = self.transformer_model.get_sentence_embedding_dimension()
                logger.info(f"SentenceTransformer model loaded. Embedding dimension: {self.transformer_embedding_dim}")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}", exc_info=True)
                logger.warning("Disabling SentenceTransformer embeddings due to loading error.")
                self.use_transformer_embeddings = False
    
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
        austronesian = {'tl', 'ceb', 'ilo', 'hil', 'war', 'pam', 'bik', 'pag', 'krj'}
        sino_tibetan = {'zh', 'yue', 'wuu', 'hak', 'nan'}
        romance = {'es', 'pt', 'fr', 'it', 'ro', 'la'}
        germanic = {'en', 'de', 'nl', 'af', 'sv', 'no', 'da'}
        
        for i, word_id in enumerate(word_ids):
            if word_id not in etymologies:
                continue
                
            word_etyms = etymologies[word_id]
            
            # Count etymology sources by language family
            lang_counts = {
                'austronesian': 0,
                'sino_tibetan': 0,
                'romance': 0,
                'germanic': 0,
                'sanskrit': 0,
                'arabic': 0,
                'other': 0
            }
            
            for etym in word_etyms:
                lang_codes = etym.get('language_codes', [])
                if not lang_codes:
                    continue
                    
                for lang in lang_codes:
                    if lang in austronesian:
                        lang_counts['austronesian'] += 1
                    elif lang in sino_tibetan:
                        lang_counts['sino_tibetan'] += 1
                    elif lang in romance:
                        lang_counts['romance'] += 1
                    elif lang in germanic:
                        lang_counts['germanic'] += 1
                    elif lang == 'sa':
                        lang_counts['sanskrit'] += 1
                    elif lang == 'ar':
                        lang_counts['arabic'] += 1
                    else:
                        lang_counts['other'] += 1
            
            # Set features
            features[i, 0] = float(len(word_etyms))
            features[i, 1] = float(lang_counts['austronesian'])
            features[i, 2] = float(lang_counts['sino_tibetan'])
            features[i, 3] = float(lang_counts['romance'])
            features[i, 4] = float(lang_counts['germanic'])
            features[i, 5] = float(lang_counts['sanskrit'])
            features[i, 6] = float(lang_counts['arabic'])
            features[i, 7] = float(lang_counts['other'])
            
            # Feature: has mixed etymology (multiple language families)
            nonzero_families = sum(1 for count in lang_counts.values() if count > 0)
            features[i, 8] = float(nonzero_families > 1)
            
            # Feature: borrowed word (non-Austronesian source)
            has_foreign = sum(lang_counts[fam] for fam in ['sino_tibetan', 'romance', 'germanic', 'sanskrit', 'arabic', 'other']) > 0
            features[i, 9] = float(has_foreign)
        
        return features
    
    def extract_char_ngrams(self, texts: List[str], n: int = 3) -> torch.Tensor:
        """
        Extract character n-gram features.
        
        Args:
            texts: List of strings to extract features from
            n: n-gram size
            
        Returns:
            Tensor of character n-gram features
        """
        # First pass: count all n-grams to build vocabulary
        ngram_counts = {}
        for text in texts:
            if not text or pd.isna(text):
                continue
                
            # Normalize and pad text
            norm_text = ''.join(c.lower() for c in text if c.isalpha())
            padded = f"{'#'*(n-1)}{norm_text}{'#'*(n-1)}"
            
            # Count n-grams
            for i in range(len(padded) - n + 1):
                ngram = padded[i:i+n]
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        # Keep top k n-grams
        k = 500  # Adjust as needed
        top_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:k]
        ngram_vocab = {ngram: idx for idx, (ngram, _) in enumerate(top_ngrams)}
        
        # Second pass: create feature vectors
        num_texts = len(texts)
        features = torch.zeros(num_texts, len(ngram_vocab))
        
        for i, text in enumerate(texts):
            if not text or pd.isna(text):
                continue
                
            # Normalize and pad text
            norm_text = ''.join(c.lower() for c in text if c.isalpha())
            padded = f"{'#'*(n-1)}{norm_text}{'#'*(n-1)}"
            
            # Count n-grams in vocab
            text_ngrams = {}
            for j in range(len(padded) - n + 1):
                ngram = padded[j:j+n]
                if ngram in ngram_vocab:
                    text_ngrams[ngram] = text_ngrams.get(ngram, 0) + 1
            
            # Populate feature vector
            for ngram, count in text_ngrams.items():
                features[i, ngram_vocab[ngram]] = float(count)
            
            # Normalize by text length (optional)
            if norm_text:
                norm_factor = len(norm_text)
                features[i] /= max(1, norm_factor)
        
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
            logger.warning("XLM-RoBERTa not available, returning zero embeddings")
            return torch.zeros(len(texts), 768)  # Default XLM-R embedding size
            
        # Process in smaller batches to avoid OOM
        batch_size = 32
        num_texts = len(texts)
        embeddings = torch.zeros(num_texts, 768)
        
        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Filter out empty texts
            valid_indices = []
            valid_texts = []
            for j, text in enumerate(batch_texts):
                if text and not pd.isna(text):
                    valid_indices.append(j)
                    valid_texts.append(text)
            
            if not valid_texts:
                continue
                
            # Tokenize and get embeddings
            with torch.no_grad():
                inputs = self.xlmr_tokenizer(
                    valid_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=128
                )
                
                outputs = self.xlmr_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # CLS token embeddings
                
                # Store embeddings
                for j, idx in enumerate(valid_indices):
                    embeddings[i + idx] = batch_embeddings[j]
        
        return embeddings
    
    def extract_transformer_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Extract SentenceTransformer embeddings.

        Args:
            texts: List of sentences/texts to embed.
            batch_size: Batch size for encoding.

        Returns:
            Tensor of SentenceTransformer embeddings.
        """
        if not self.use_transformer_embeddings or not self.transformer_model:
            logger.warning("SentenceTransformer embeddings not available/enabled, returning zero embeddings.")
            # Determine embedding dim safely
            dim = self.transformer_embedding_dim if self.transformer_embedding_dim else 768 # Default fallback dim
            return torch.zeros(len(texts), dim)

        logger.info(f"Generating SentenceTransformer embeddings for {len(texts)} texts...")
        try:
            # Encode texts using the sentence-transformer model
            embeddings = self.transformer_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=True, # Show progress for larger lists
                batch_size=batch_size
            )
            logger.info(f"Created SentenceTransformer embeddings with shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating SentenceTransformer embeddings: {e}", exc_info=True)
            dim = self.transformer_embedding_dim if self.transformer_embedding_dim else 768 # Use loaded dim or fallback
            return torch.zeros(len(texts), dim) # Return zeros on error
    
    def extract_pronunciation_features(self, pronunciations_df: pd.DataFrame) -> Dict[int, torch.Tensor]:
        """
        Extract features from pronunciations data.
        
        Args:
            pronunciations_df: DataFrame containing pronunciations
            
        Returns:
            Dictionary mapping word IDs to pronunciation feature tensors
        """
        if pronunciations_df.empty:
            return {}
            
        # Group pronunciations by word_id
        word_pronunciations = {}
        feature_dim = 10  # Adjust as needed
        
        for _, row in pronunciations_df.iterrows():
            word_id = row.get('word_id')
            if word_id is None:
                continue
                
            if word_id not in word_pronunciations:
                word_pronunciations[word_id] = []
                
            word_pronunciations[word_id].append({
                'type': row.get('type'),
                'value': row.get('value')
            })
        
        # Extract features for each word
        pronunciation_features = {}
        for word_id, pronunciations in word_pronunciations.items():
            features = torch.zeros(feature_dim)
            
            # Count by type
            type_counts = {}
            for pron in pronunciations:
                pron_type = pron.get('type', 'unknown')
                type_counts[pron_type] = type_counts.get(pron_type, 0) + 1
                
            # Set features
            features[0] = float(len(pronunciations))
            features[1] = float(type_counts.get('ipa', 0))
            features[2] = float(type_counts.get('respelling_guide', 0))
            features[3] = float(type_counts.get('text', 0))
            
            # Additional features (add more as needed)
            has_ipa = type_counts.get('ipa', 0) > 0
            has_multiple_types = len(type_counts) > 1
            features[4] = float(has_ipa)
            features[5] = float(has_multiple_types)
            
            # Store features
            pronunciation_features[word_id] = features
            
        return pronunciation_features
    
    def extract_all_features(self, 
                           lemmas_df: pd.DataFrame, 
                           definitions_df: Optional[pd.DataFrame] = None,
                           etymologies_df: Optional[pd.DataFrame] = None,
                           pronunciations_df: Optional[pd.DataFrame] = None,
                           word_forms_df: Optional[pd.DataFrame] = None) -> Dict[str, torch.Tensor]:
        """
        Extract all features for words and other entities.
        
        Args:
            lemmas_df: DataFrame containing word lemmas
            definitions_df: DataFrame containing definitions
            etymologies_df: DataFrame containing etymologies
            pronunciations_df: DataFrame containing pronunciations
            word_forms_df: DataFrame containing word forms
            
        Returns:
            Dictionary mapping node types to feature tensors
        """
        logger.info("Extracting features for lexical data...")
        
        # 1. Extract word features
        word_features = []
        
        # Get basic word info
        word_texts = lemmas_df['lemma'].tolist()
        word_ids = lemmas_df['id'].tolist()
        lang_codes = lemmas_df['language_code'].tolist()
        
        # Baybayin features (if enabled)
        if self.use_baybayin_features:
            baybayin_texts = lemmas_df.get('baybayin_form', [None] * len(word_texts))
            romanized_texts = lemmas_df.get('romanized_form', [None] * len(word_texts))
            baybayin_features = self.extract_baybayin_features(baybayin_texts, romanized_texts)
            word_features.append(baybayin_features)
        
        # Character n-gram features (if enabled)
        if self.use_char_ngrams:
            ngram_features = self.extract_char_ngrams(word_texts, n=3)
            word_features.append(ngram_features)
        
        # Phonetic features (if enabled)
        if self.use_phonetic_features:
            phonetic_features = self.extract_phonetic_features(word_texts, lang_codes)
            word_features.append(phonetic_features)
        
        # XLM-RoBERTa embeddings (if enabled)
        if self.use_xlmr:
            xlmr_embeddings = self.extract_xlmr_embeddings(word_texts)
            word_features.append(xlmr_embeddings)
        
        # SentenceTransformer embeddings (if enabled)
        if self.use_transformer_embeddings:
            st_embeddings = self.extract_transformer_embeddings(word_texts)
            word_features.append(st_embeddings)
        
        # Etymology features (if enabled and data available)
        if self.use_etymology_features and etymologies_df is not None and not etymologies_df.empty:
            # Process etymologies
            word_etymologies = {}
            for _, row in etymologies_df.iterrows():
                word_id = row.get('word_id')
                if word_id is None:
                    continue
                    
                if word_id not in word_etymologies:
                    word_etymologies[word_id] = []
                    
                word_etymologies[word_id].append({
                    'etymology_text': row.get('etymology_text'),
                    'language_codes': row.get('language_codes'),
                    'normalized_components': row.get('normalized_components')
                })
            
            # Extract features from etymologies
            etymology_features = self.extract_etymology_features(word_etymologies)
            word_features.append(etymology_features)
        
        # Pronunciation features (if data available)
        if pronunciations_df is not None and not pronunciations_df.empty:
            pronunciation_features = self.extract_pronunciation_features(pronunciations_df)
            
            # Convert to tensor matching the word_ids order
            pron_tensor = torch.zeros(len(word_ids), pronunciation_features[next(iter(pronunciation_features))].size(0)) 
            for i, word_id in enumerate(word_ids):
                if word_id in pronunciation_features:
                    pron_tensor[i] = pronunciation_features[word_id]
                    
            word_features.append(pron_tensor)
        
        # Concatenate all word features
        if word_features:
            word_feature_tensor = torch.cat(word_features, dim=1)
            
            # Normalize if needed
            if self.normalize_features:
                word_feature_tensor = self._normalize_features(word_feature_tensor)
        else:
            # Fallback to dummy features if no extractions were done
            word_feature_tensor = torch.randn(len(word_ids), 10)
        
        # 2. Extract definition features if needed
        definition_features = None
        if definitions_df is not None and not definitions_df.empty:
            # Simple version: just extract from definition text
            def_texts = definitions_df['definition_text'].tolist()
            
            def_features = []
            
            # SentenceTransformer embeddings for definitions
            if self.use_transformer_embeddings:
                 st_def_embeddings = self.extract_transformer_embeddings(def_texts)
                 def_features.append(st_def_embeddings)

            # Concatenate and normalize
            if def_features:
                definition_features = torch.cat(def_features, dim=1)
                if self.normalize_features:
                    definition_features = self._normalize_features(definition_features)
            else:
                definition_features = torch.randn(len(def_texts), 10)
        
        # Return features by node type
        features = {
            'word': word_feature_tensor
        }
        
        if definition_features is not None:
            features['definition'] = definition_features
        
        return features
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features to have zero mean and unit variance.
        
        Args:
            features: Feature tensor [num_entities, feature_dim]
            
        Returns:
            Normalized feature tensor
        """
        # Calculate mean and std along each feature dimension
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-6)  # Avoid division by zero
        
        # Normalize
        normalized = (features - mean) / std
        
        # Handle NaNs from constant features
        normalized[normalized != normalized] = 0.0
        
        return normalized 