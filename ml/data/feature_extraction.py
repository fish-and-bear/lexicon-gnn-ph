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
from transformers import AutoTokenizer, AutoModel, XLMRobertaTokenizer, XLMRobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Setup logging
logger = logging.getLogger(__name__)

class LexicalFeatureExtractor:
    """Extract features from lexical data."""
    
    def __init__(self, 
                 use_xlmr: bool = True, 
                 use_transformer_embeddings: bool = True,
                 sentence_transformer_model_name: Optional[str] = None,
                 use_char_ngrams: bool = True,
                 use_phonetic_features: bool = True,
                 use_etymology_features: bool = True,
                 use_baybayin_features: bool = True,
                 normalize_features: bool = True,
                 **kwargs):
        """
        Initialize feature extractor.
        
        Args:
            use_xlmr: Whether to use XLM-RoBERTa embeddings
            use_transformer_embeddings: Whether to use SentenceTransformer embeddings
            sentence_transformer_model_name: Name of the SentenceTransformer model to use
            use_char_ngrams: Whether to use character n-grams
            use_phonetic_features: Whether to use phonetic similarity features
            use_etymology_features: Whether to use etymology features
            use_baybayin_features: Whether to use Baybayin script features
            normalize_features: Whether to normalize features
            **kwargs: Additional keyword arguments
        """
        self.use_xlmr = use_xlmr
        self.use_transformer_embeddings = use_transformer_embeddings
        self.sentence_transformer_model_name = sentence_transformer_model_name
        self.use_char_ngrams = use_char_ngrams
        self.use_phonetic_features = use_phonetic_features
        self.use_etymology_features = use_etymology_features
        self.use_baybayin_features = use_baybayin_features
        self.normalize_features = normalize_features
        
        # Initialize models if needed
        self.xlmr_model = None
        self.xlmr_tokenizer = None
        self.transformer_model = None
        self.transformer_embedding_dim = 0 # Initialize
        self.st_model_name_for_reload = None # Will be set if ST is used
        self.st_tokenizer = None # For direct tokenization checks
        self.st_vocab_size = None # For direct tokenization checks
        
        # Store model_original_feat_dims if provided, for placeholder creation
        self.model_original_feat_dims = kwargs.get('model_original_feat_dims', {})
        if self.model_original_feat_dims:
            logger.info(f"LexicalFeatureExtractor initialized with model_original_feat_dims: {self.model_original_feat_dims}")
        else:
            logger.warning("LexicalFeatureExtractor initialized WITHOUT model_original_feat_dims. Placeholders will be 1-dimensional.")
        
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
                model_name_to_load = self.sentence_transformer_model_name if self.sentence_transformer_model_name else "paraphrase-multilingual-MiniLM-L12-v2"
                self.st_model_name_for_reload = model_name_to_load
                logger.info(f"Attempting to load SentenceTransformer model: {model_name_to_load}")
                self.transformer_model = SentenceTransformer(model_name_to_load)
                
                # Get tokenizer and vocab size for diagnostics
                if hasattr(self.transformer_model, 'tokenizer'):
                    self.st_tokenizer = self.transformer_model.tokenizer
                    if hasattr(self.st_tokenizer, 'vocab_size'):
                        self.st_vocab_size = self.st_tokenizer.vocab_size
                        logger.info(f"SentenceTransformer's tokenizer ({type(self.st_tokenizer).__name__}) loaded with vocab_size: {self.st_vocab_size}")
                    else:
                        logger.warning("SentenceTransformer's tokenizer does not have 'vocab_size' attribute.")
                else:
                    logger.warning("SentenceTransformer model does not have 'tokenizer' attribute for diagnostics.")

                if torch.cuda.is_available():
                    logger.info("Moving SentenceTransformer model to CUDA.")
                    self.transformer_model = self.transformer_model.to("cuda")
                self.transformer_embedding_dim = self.transformer_model.get_sentence_embedding_dimension()
                logger.info(f"SentenceTransformer model '{model_name_to_load}' loaded. Embedding dimension: {self.transformer_embedding_dim}")

                # Explicitly configure tokenizer_args and sub-module max_seq_length for robust truncation
                try:
                    transformer_module = self.transformer_model._first_module() # Typically sentence_transformers.models.Transformer
                    if hasattr(transformer_module, 'tokenizer') and hasattr(transformer_module.tokenizer, 'model_max_length'):
                        desired_max_tok_len = transformer_module.tokenizer.model_max_length
                        # Fallback if model_max_length is somehow None or too large for typical RoBERTa
                        if desired_max_tok_len is None or desired_max_tok_len > 512: 
                            desired_max_tok_len = 512
                        
                        if not hasattr(transformer_module, 'tokenizer_args') or transformer_module.tokenizer_args is None: # Ensure tokenizer_args exists
                            transformer_module.tokenizer_args = {}
                        
                        if transformer_module.tokenizer_args.get('max_length') != desired_max_tok_len or \
                           transformer_module.tokenizer_args.get('truncation') is not True:
                            logger.info(f"LexicalFeatureExtractor: Explicitly setting tokenizer_args: max_length={desired_max_tok_len}, truncation=True for ST's Transformer module.")
                            transformer_module.tokenizer_args['max_length'] = desired_max_tok_len
                            transformer_module.tokenizer_args['truncation'] = True
                        else:
                            logger.info(f"LexicalFeatureExtractor: ST's Transformer module tokenizer_args already appropriately set: {transformer_module.tokenizer_args}")

                        # Ensure the Transformer sub-module's own max_seq_length (post-tokenization slice) is consistent
                        if hasattr(transformer_module, 'max_seq_length'):
                            if transformer_module.max_seq_length is None or transformer_module.max_seq_length > desired_max_tok_len:
                                logger.info(f"LexicalFeatureExtractor: Adjusting ST Transformer sub-module's max_seq_length from {transformer_module.max_seq_length} to {desired_max_tok_len}")
                                transformer_module.max_seq_length = desired_max_tok_len
                            else:
                                logger.info(f"LexicalFeatureExtractor: ST Transformer sub-module's max_seq_length ({transformer_module.max_seq_length}) is consistent with desired_max_tok_len ({desired_max_tok_len}).")
                    else:
                        logger.warning("LexicalFeatureExtractor: Could not access tokenizer or model_max_length on ST's first module to enforce tokenizer settings.")
                except Exception as e_configure_st:
                    logger.error(f"LexicalFeatureExtractor: Error configuring SentenceTransformer's tokenizer/max_length: {e_configure_st}", exc_info=True)

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
    
    def extract_etymology_features(self, all_word_ids: List[int], etymologies_df: Optional[pd.DataFrame], target_device: torch.device) -> Tuple[Optional[torch.Tensor], List[str]]:
        """
        Extract features from etymology data.
        Args:
            all_word_ids: List of all word IDs from lemmas_df.
            etymologies_df: Filtered DataFrame containing etymology data for relevant word_ids.
            target_device: The torch device to place the feature tensor on.
            
        Returns:
            A tuple containing:
                - Tensor of etymology features, shape (len(all_word_ids), feature_dim), or None.
                - List of feature names for these etymology features.
        """
        feature_dim = 10 # Let's define a fixed dim, e.g. 10, matching previous use for has_foreign etc.
        feature_names = [f"etym_feat_{i}" for i in range(feature_dim)] # Generate names

        if etymologies_df is None or etymologies_df.empty or 'word_id' not in etymologies_df.columns:
            logger.warning("Etymologies DataFrame is None, empty, or missing 'word_id' column. Returning None for etymology features.")
            # Return a zero tensor of the correct shape for all_word_ids if a consistent feature shape is needed for concatenation, even if no data.
            # However, if other features might also be None, it's better to return None and let the calling function handle it.
            # For now, to match previous logic that allowed concatenation: return a placeholder and empty names.
            # return torch.zeros((len(all_word_ids), feature_dim), device=target_device), [] # Option 1: placeholder
            return None, [] # Option 2: None, to be handled by caller

        num_total_words = len(all_word_ids)
        features = torch.zeros(num_total_words, feature_dim, device=target_device)
        word_id_to_index = {word_id: i for i, word_id in enumerate(all_word_ids)}

        # Create an etymology_data_map from the filtered_etymologies_df
        etymology_data_map_local = {}
        for _, row in etymologies_df.iterrows():
            word_id = row['word_id']
            if word_id not in etymology_data_map_local:
                etymology_data_map_local[word_id] = []
            # Assuming etymology structure in DataFrame rows needs to be converted to dict list per word_id
            # This part depends heavily on the actual structure of your etymologies_df
            # For example, if each row is one etymology link for a word:
            etym_entry = {key: row.get(key) for key in row.index if key != 'word_id'} # Simplistic conversion
            etymology_data_map_local[word_id].append(etym_entry)

        austronesian = {'tl', 'ceb', 'ilo', 'hil', 'war', 'pam', 'bik', 'pag', 'krj'}
        sino_tibetan = {'zh', 'yue', 'wuu', 'hak', 'nan'}
        romance = {'es', 'pt', 'fr', 'it', 'ro', 'la'}
        germanic = {'en', 'de', 'nl', 'af', 'sv', 'no', 'da'}
        
        processed_word_ids_count = 0
        for word_id, word_etyms in etymology_data_map_local.items():
            if word_id not in word_id_to_index:
                # This warning should be rare now due to pre-filtering in extract_all_features
                logger.warning(f"Word ID {word_id} from local etymology data map not in all_word_ids. Skipping.")
                continue
                
            idx = word_id_to_index[word_id]
            processed_word_ids_count +=1
            
            lang_counts = {'austronesian': 0, 'sino_tibetan': 0, 'romance': 0, 'germanic': 0, 'sanskrit': 0, 'arabic': 0, 'other': 0}
            
            for etym in word_etyms:
                # Assuming etym is a dict and might have 'language_codes' as a list or 'language_code' as a string
                raw_lang_codes = etym.get('language_codes', etym.get('language_code')) # Check both
                actual_lang_codes = []
                if isinstance(raw_lang_codes, str):
                    actual_lang_codes = [raw_lang_codes]
                elif isinstance(raw_lang_codes, list):
                    actual_lang_codes = raw_lang_codes
                
                if not actual_lang_codes:
                    continue
                    
                for lang in actual_lang_codes:
                    if pd.isna(lang): continue # Skip NaN lang codes
                    lang_lower = lang.lower()
                    if lang_lower in austronesian: lang_counts['austronesian'] += 1
                    elif lang_lower in sino_tibetan: lang_counts['sino_tibetan'] += 1
                    elif lang_lower in romance: lang_counts['romance'] += 1
                    elif lang_lower in germanic: lang_counts['germanic'] += 1
                    elif lang_lower == 'sa': lang_counts['sanskrit'] += 1
                    elif lang_lower == 'ar': lang_counts['arabic'] += 1
                    else: lang_counts['other'] += 1
            
            features[idx, 0] = float(len(word_etyms))
            features[idx, 1] = float(lang_counts['austronesian'])
            features[idx, 2] = float(lang_counts['sino_tibetan'])
            features[idx, 3] = float(lang_counts['romance'])
            features[idx, 4] = float(lang_counts['germanic'])
            features[idx, 5] = float(lang_counts['sanskrit'])
            features[idx, 6] = float(lang_counts['arabic'])
            features[idx, 7] = float(lang_counts['other'])
            nonzero_families = sum(1 for count in lang_counts.values() if count > 0)
            features[idx, 8] = float(nonzero_families > 1)
            has_foreign = sum(lang_counts[fam] for fam in ['sino_tibetan', 'romance', 'germanic', 'sanskrit', 'arabic', 'other']) > 0
            features[idx, 9] = float(has_foreign)
        
        logger.info(f"Processed etymology features for {processed_word_ids_count} word IDs out of {len(etymology_data_map_local)} in local map.")
        return features, feature_names
    
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
    
    def extract_transformer_embeddings(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        """
        Extract embeddings using SentenceTransformer model.
        
        Args:
            texts: List of strings to extract features from
            batch_size: Batch size for encoding
            
        Returns:
            Tensor of SentenceTransformer embeddings
        """
        if not self.use_transformer_embeddings or self.transformer_model is None or \
           not self.st_tokenizer or self.st_vocab_size is None:
            logger.warning("SentenceTransformer model, tokenizer, or vocab_size not available (from __init__). Cannot perform ST embedding.")
            fallback_dim = self.model_original_feat_dims.get('definition', self.model_original_feat_dims.get('word', 0))
            if fallback_dim == 0 and self.transformer_embedding_dim > 0: fallback_dim = self.transformer_embedding_dim
            if fallback_dim == 0: fallback_dim = 1
            effective_device_for_empty = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Returning empty tensor for ST embeddings (model/tokenizer not available) with dim {fallback_dim} on device {effective_device_for_empty}.")
            return torch.zeros(len(texts), fallback_dim if len(texts) > 0 else 0, device=effective_device_for_empty)

        primary_device_to_try = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Text Sanitization Loop ---
        sanitized_texts_for_batch = []
        problematic_texts_indices = []
        logger.info(f"Performing text sanitization and token validation for {len(texts)} texts...")
        for idx, text_item in enumerate(texts):
            current_text_to_process = str(text_item) if pd.notna(text_item) else "" # Ensure string, handle None/NaN
            try:
                # Tokenize without special tokens for the raw token ID check
                token_ids = self.st_tokenizer.encode(current_text_to_process, add_special_tokens=False) 
                
                if not token_ids and current_text_to_process: # Check if non-empty input text yields empty raw tokens
                    logger.warning(f"Text at index {idx} (original: '{str(text_item)[:50]}...') resulted in EMPTY raw token list. Replacing text with UNK.")
                    problematic_texts_indices.append(idx)
                    sanitized_texts_for_batch.append(self.st_tokenizer.unk_token if hasattr(self.st_tokenizer, 'unk_token') and self.st_tokenizer.unk_token else "[UNK]")
                    continue # Move to the next text_item

                valid_tokens = True
                for token_id in token_ids:
                    if not (0 <= token_id < self.st_vocab_size):
                        logger.warning(f"Text at index {idx} (original: '{str(text_item)[:50]}...') contains out-of-range token ID: {token_id} (raw token from text: '{current_text_to_process[:50]}...'). Vocab size: {self.st_vocab_size}. Replacing text with UNK.")
                        problematic_texts_indices.append(idx)
                        sanitized_texts_for_batch.append(self.st_tokenizer.unk_token if hasattr(self.st_tokenizer, 'unk_token') and self.st_tokenizer.unk_token else "[UNK]")
                        valid_tokens = False
                        break 
                if valid_tokens:
                    sanitized_texts_for_batch.append(current_text_to_process)
            except Exception as e_tokenize_sanitize:
                logger.error(f"Error tokenizing/sanitizing text at index {idx} ('{str(text_item)[:50]}...'): {e_tokenize_sanitize}. Replacing with UNK.", exc_info=True)
                problematic_texts_indices.append(idx)
                sanitized_texts_for_batch.append(self.st_tokenizer.unk_token if hasattr(self.st_tokenizer, 'unk_token') and self.st_tokenizer.unk_token else "[UNK]")
        
        if problematic_texts_indices:
            logger.warning(f"Sanitized {len(problematic_texts_indices)} texts by replacing them with UNK token due to out-of-range token IDs.")
        else:
            logger.info("All texts passed token validation or were already compliant.")
        # --- End Text Sanitization ---

        current_st_model_instance = self.transformer_model # Use the instance model

        try:
            if current_st_model_instance.device.type != primary_device_to_try:
                 logger.warning(f"LFE.transformer_model is on {current_st_model_instance.device} but primary_device_to_try is {primary_device_to_try}. Moving model.")
                 current_st_model_instance.to(primary_device_to_try)

            # Encode the sanitized texts
            embeddings = current_st_model_instance.encode(
                sanitized_texts_for_batch, # Use sanitized texts
                convert_to_tensor=True,
                show_progress_bar=len(sanitized_texts_for_batch) > 2*batch_size,
                batch_size=batch_size,
                device=primary_device_to_try 
            )
            return embeddings

        except RuntimeError as e_rt_primary_attempt:
            logger.error(f"RuntimeError on primary device ({primary_device_to_try}) using SANITIZED texts for {len(sanitized_texts_for_batch)} items: {e_rt_primary_attempt}. Attempting aggressive CPU fallback.", exc_info=True)
            # Log sample of *sanitized* texts that still caused issues
            sample_texts_on_error = sanitized_texts_for_batch[:min(3, len(sanitized_texts_for_batch))] 
            for i_txt, txt_err in enumerate(sample_texts_on_error):
                logger.warning(f"  Problematic Sanitized Sample Text {i_txt} (len {len(str(txt_err))}): '{str(txt_err)[:100]}...'")
            
            logger.info("Attempting AGGRESSIVE CPU FALLBACK: Loading a fresh ST model instance on CPU.")
            try:
                if not self.st_model_name_for_reload:
                    logger.error("Cannot attempt aggressive CPU fallback: self.st_model_name_for_reload is not set.")
                    raise ValueError("st_model_name_for_reload not available.")

                logger.info(f"Loading fresh ST model '{self.st_model_name_for_reload}' onto CPU.")
                cpu_st_model = SentenceTransformer(self.st_model_name_for_reload, device='cpu')
                logger.info(f"Fresh ST model loaded on CPU. Device: {cpu_st_model.device}")

                # No need for re-sanitization here if the first sanitization was thorough based on self.st_vocab_size
                # However, if the fresh model has a *different* vocab, this could be an issue.
                # For now, assume self.st_vocab_size is the reference for the named model.
                logger.info(f"Encoding {len(sanitized_texts_for_batch)} SANITIZED texts using the fresh CPU ST model...")
                cpu_embeddings = cpu_st_model.encode(
                    sanitized_texts_for_batch, # Use already sanitized texts
                    convert_to_tensor=True,
                    show_progress_bar=len(sanitized_texts_for_batch) > 2*batch_size,
                    batch_size=batch_size,
                    device='cpu' 
                )
                logger.info(f"Successfully encoded SANITIZED batch on fresh CPU model. Shape: {cpu_embeddings.shape}")
                
                if primary_device_to_try == 'cuda' and self.transformer_model is not None and self.transformer_model.device.type != 'cuda':
                    try: self.transformer_model.to(primary_device_to_try); logger.info(f"Restored self.transformer_model to {self.transformer_model.device}.")
                    except Exception as e_restore_gpu: logger.error(f"Failed to restore self.transformer_model to CUDA: {e_restore_gpu}")
                
                return cpu_embeddings.to(torch.device(primary_device_to_try))
            
            except Exception as e_aggressive_cpu_fallback: # Catch any error from aggressive CPU fallback (load or encode)
                logger.error(f"AGGRESSIVE CPU FALLBACK FAILED for {len(sanitized_texts_for_batch)} SANITIZED texts: {e_aggressive_cpu_fallback}", exc_info=True)
            
            # Common path to final fallback if any part of aggressive CPU fallback failed
            target_fallback_dim_agg_fail = self.transformer_embedding_dim if self.transformer_embedding_dim > 0 else 1
            logger.warning(f"FINAL FALLBACK (after aggressive CPU attempt on sanitized texts failed): Returning empty tensor. Shape: ({len(texts)}, {target_fallback_dim_agg_fail}) on device {primary_device_to_try}")
            return torch.zeros(len(texts), target_fallback_dim_agg_fail, device=torch.device(primary_device_to_try))

        except Exception as e_general_other: 
            logger.error(f"General (non-RuntimeError) error during ST encoding on {primary_device_to_try} for SANITIZED texts ({len(sanitized_texts_for_batch)} items): {e_general_other}", exc_info=True)
            target_fallback_dim_gen_err = self.transformer_embedding_dim if self.transformer_embedding_dim > 0 else 1
            logger.warning(f"FINAL FALLBACK (general error on sanitized texts): Returning empty tensor. Shape: ({len(texts)}, {target_fallback_dim_gen_err}) on device {primary_device_to_try}")
            return torch.zeros(len(texts), target_fallback_dim_gen_err, device=torch.device(primary_device_to_try))
    
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
                           graph_num_nodes_per_type: Dict[str, int],
                           node_to_original_id_maps: Optional[Dict[str, Dict[int, int]]] = None,
                           definitions_df: Optional[pd.DataFrame] = None,
                           etymologies_df: Optional[pd.DataFrame] = None,
                           pronunciations_df: Optional[pd.DataFrame] = None,
                           word_forms_df: Optional[pd.DataFrame] = None,
                           relevant_word_ids: Optional[Set[int]] = None,
                           relevant_definition_ids: Optional[Set[int]] = None,
                           relevant_etymology_ids: Optional[Set[int]] = None,
                           target_device: Optional[torch.device] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[int]]]:
        """
        Extract all features for the graph.

        Prioritizes using node_to_original_id_maps for ordering if available.
        Relevant_xxx_ids are secondary and might be used if maps are incomplete or for specific filtering.
        """
        final_features: Dict[str, torch.Tensor] = {}
        final_ordered_original_ids_map: Dict[str, List[int]] = {} # Stores original DB IDs in graph node order

        logger.info("Starting feature extraction for all node types...")
        
        if definitions_df is not None and not definitions_df.empty:
            logger.info(f"DEBUG: definitions_df columns: {definitions_df.columns.tolist()}")
            logger.info(f"DEBUG: definitions_df head:\n{definitions_df.head().to_string()}")
        else:
            logger.info("DEBUG: definitions_df is None or empty at the start of extract_all_features.")

        if etymologies_df is not None and not etymologies_df.empty:
            logger.info(f"DEBUG: etymologies_df columns: {etymologies_df.columns.tolist()}")
            logger.info(f"DEBUG: etymologies_df head:\n{etymologies_df.head().to_string()}")
        else:
            logger.info("DEBUG: etymologies_df is None or empty at the start of extract_all_features.")
            
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # NOTE: Removed the explicit ST model device check here to avoid linter issues with complex edits.
        # LFE.__init__ already attempts to move ST model to CUDA if available.
        # extract_transformer_embeddings also moves resulting tensors to target_device.

        # --- WORD FEATURES ---
        ntype_word = 'word'
        logger.info(f"Processing features for node type: {ntype_word}")
        num_word_nodes = graph_num_nodes_per_type.get(ntype_word, 0)
        word_node_to_original_id_map = node_to_original_id_maps.get(ntype_word, {}) if node_to_original_id_maps else {}
        
        ordered_word_original_ids = [] # Initialize
        word_texts = [] # Initialize word_texts here

        if num_word_nodes == 0:
            self.logger.info(f"No nodes of type '{ntype_word}' in graph. Skipping feature extraction.")
        else:
            ordered_word_original_ids = self._determine_ordered_ids_for_ntype(
                ntype_word, num_word_nodes, 
                lemmas_df['id'].tolist() if lemmas_df is not None and 'id' in lemmas_df else [], 
                word_node_to_original_id_map, logger
            )
            final_ordered_original_ids_map[ntype_word] = ordered_word_original_ids

            word_feature_parts = []
            if num_word_nodes > 0 and ordered_word_original_ids:
                # Align lemmas_df with the order of nodes in the graph
                # Create a temporary DataFrame with 'original_id' column for merging
                ordered_ids_df_word = pd.DataFrame({'original_id': ordered_word_original_ids, 'graph_order': range(len(ordered_word_original_ids))})
                # Merge with lemmas_df, then sort by graph_order to ensure alignment, then select relevant columns
                aligned_lemmas_for_features = pd.merge(lemmas_df, ordered_ids_df_word, left_on='id', right_on='original_id', how='inner')
                aligned_lemmas_for_features = aligned_lemmas_for_features.sort_values(by='graph_order').reset_index(drop=True)

                if not aligned_lemmas_for_features.empty:
                    word_texts = aligned_lemmas_for_features['lemma'].fillna('').tolist() # Ensure all are strings

                    # ---- START DIAGNOSTIC LOGGING FOR WORD ----
                    logger.info(f"LFE_PRE_ST_WORD: About to call ST for 'word'. Num texts: {len(word_texts)}. Sample texts: {word_texts[:min(3, len(word_texts))]}")
                    logger.info(f"LFE_PRE_ST_WORD: self.use_transformer_embeddings: {self.use_transformer_embeddings}") 
                    logger.info(f"LFE_PRE_ST_WORD: self.transformer_model is None: {self.transformer_model is None}")
                    if self.transformer_model:
                        logger.info(f"LFE_PRE_ST_WORD: self.transformer_model.device: {self.transformer_model.device if hasattr(self.transformer_model, 'device') else 'N/A'}")
                    logger.info(f"LFE_PRE_ST_WORD: aligned_lemmas_for_features shape: {aligned_lemmas_for_features.shape}")
                    # ---- END DIAGNOSTIC LOGGING FOR WORD ----

                    if self.use_transformer_embeddings and self.transformer_model:
                        logger.info(f"Extracting SentenceTransformer embeddings for {len(word_texts)} '{ntype_word}' texts.")
                        st_word_embeddings = self.extract_transformer_embeddings(word_texts)
                        if st_word_embeddings is not None and st_word_embeddings.nelement() > 0:
                            word_feature_parts.append(st_word_embeddings.to(target_device))
                            logger.info(f"SentenceTransformer '{ntype_word}' embeddings shape: {st_word_embeddings.shape}")
                        else:
                            logger.warning(f"SentenceTransformer embeddings for '{ntype_word}' are None or empty.")
                    
                    # Add other word feature types (char_ngrams, phonetic, etc.)
                    if self.use_char_ngrams:
                        char_ngram_features = self.extract_char_ngrams(word_texts)
                        word_feature_parts.append(char_ngram_features.to(target_device))
                    
                    # Note: Phonetic and Baybayin features can be added here if configured
                else: # aligned_lemmas_for_features is empty
                    logger.warning(f"No lemmas found in lemmas_df that match the ordered_word_original_ids for '{ntype_word}'. Transformer/char-ngram features will be skipped/empty.")

            # Assemble word features or create placeholder
            if word_feature_parts:
                # Ensure all parts are 2D before concatenation
                processed_word_feature_parts = []
                for i, part in enumerate(word_feature_parts):
                    if part.ndim == 1:
                        part = part.unsqueeze(1)
                    if part.shape[0] != num_word_nodes:
                        logger.warning(f"Word feature part {i} has {part.shape[0]} rows, expected {num_word_nodes}. This indicates misalignment. Attempting to use if non-empty.")
                        if part.shape[0] == 0: continue # Skip empty misaligned parts
                    processed_word_feature_parts.append(part)
                
                if processed_word_feature_parts:
                    final_features[ntype_word] = torch.cat(processed_word_feature_parts, dim=1).float()
                    logger.info(f"Final '{ntype_word}' features assembled. Shape: {final_features[ntype_word].shape}")
                else: # All parts were empty or misaligned
                    logger.warning(f"All word feature parts were empty or misaligned. Creating placeholder for '{ntype_word}'.")
                    target_dim_word = self.model_original_feat_dims.get(ntype_word, 1) # Default to 1 if not specified
                    final_features[ntype_word] = torch.zeros((num_word_nodes, target_dim_word), device=target_device).float()
            elif num_word_nodes > 0:
                logger.warning(f"No feature parts generated for '{ntype_word}'. Creating placeholder.")
                target_dim_word = self.model_original_feat_dims.get(ntype_word, 1)
                final_features[ntype_word] = torch.zeros((num_word_nodes, target_dim_word), device=target_device).float()
            else: # num_word_nodes is 0
                target_dim_word = self.model_original_feat_dims.get(ntype_word, 1)
                final_features[ntype_word] = torch.zeros((0, target_dim_word), device=target_device).float()
            logger.info(f"Final '{ntype_word}' features tensor shape: {final_features[ntype_word].shape if ntype_word in final_features else 'Not Created (0 nodes?)'}")


        # --- DEFINITION FEATURES ---
        ntype_def = 'definition'
        logger.info(f"Processing features for node type: {ntype_def}")
        num_def_nodes = graph_num_nodes_per_type.get(ntype_def, 0)
        def_node_to_original_id_map = node_to_original_id_maps.get(ntype_def, {}) if node_to_original_id_maps else {}

        all_definition_ids_from_df = []
        if definitions_df is not None and 'id' in definitions_df.columns and not definitions_df.empty:
            all_definition_ids_from_df = definitions_df['id'].unique().tolist()
        
        ordered_definition_original_ids = self._determine_ordered_ids_for_ntype(
            ntype_def, num_def_nodes, 
            all_definition_ids_from_df, 
            def_node_to_original_id_map, logger
        )
        final_ordered_original_ids_map[ntype_def] = ordered_definition_original_ids
        
        def_feature_parts = [] # This will primarily hold ST embeddings if successful
        if num_def_nodes > 0 and ordered_definition_original_ids and definitions_df is not None and not definitions_df.empty:
            ordered_ids_df_def = pd.DataFrame({'original_id': ordered_definition_original_ids, 'graph_order': range(len(ordered_definition_original_ids))})
            aligned_definitions_for_features = pd.merge(definitions_df, ordered_ids_df_def, left_on='id', right_on='original_id', how='inner')
            aligned_definitions_for_features = aligned_definitions_for_features.sort_values(by='graph_order').reset_index(drop=True)

            if not aligned_definitions_for_features.empty:
                def_text_col = None
                # Look for 'definition_text' first, then other fallbacks
                for col_candidate in ['definition_text', 'definition', 'gloss', 'content', 'text']: 
                    if col_candidate in aligned_definitions_for_features.columns:
                        def_text_col = col_candidate
                        break
                
                if def_text_col:
                    logger.info(f"Using column '{def_text_col}' from definitions_df for '{ntype_def}' texts.")
                    definition_texts = aligned_definitions_for_features[def_text_col].fillna('').astype(str).tolist()
                    
                    # ---- START DIAGNOSTIC LOGGING FOR DEFINITION ----
                    logger.info(f"LFE_PRE_ST_DEF: About to call ST for 'definition'. Num texts: {len(definition_texts)}. Sample texts: {[t[:50] + '...' if len(t) > 50 else t for t in definition_texts[:min(3, len(definition_texts))]]}")
                    logger.info(f"LFE_PRE_ST_DEF: self.use_transformer_embeddings: {self.use_transformer_embeddings}")
                    logger.info(f"LFE_PRE_ST_DEF: self.transformer_model is None: {self.transformer_model is None}")
                    if self.transformer_model:
                        logger.info(f"LFE_PRE_ST_DEF: self.transformer_model.device: {self.transformer_model.device if hasattr(self.transformer_model, 'device') else 'N/A'}")
                    logger.info(f"LFE_PRE_ST_DEF: aligned_definitions_for_features shape: {aligned_definitions_for_features.shape}")
                    # ---- END DIAGNOSTIC LOGGING FOR DEFINITION ----

                    if self.use_transformer_embeddings and self.transformer_model:
                        logger.info(f"Extracting SentenceTransformer embeddings for {len(definition_texts)} '{ntype_def}' texts.")
                        st_def_embeddings = self.extract_transformer_embeddings(definition_texts) # This method should handle .to(target_device)
                        if st_def_embeddings is not None and st_def_embeddings.nelement() > 0:
                            if st_def_embeddings.shape[0] == num_def_nodes:
                                def_feature_parts.append(st_def_embeddings) # Already on target_device from extract_transformer_embeddings
                                logger.info(f"SentenceTransformer '{ntype_def}' embeddings shape: {st_def_embeddings.shape}")
                        else:
                            logger.warning(f"SentenceTransformer embeddings for '{ntype_def}' are None or empty. Placeholder will likely be used if no other features.")
                else:
                    logger.warning(f"Could not find a suitable text column in definitions_df for '{ntype_def}'. ST embeddings will be skipped.")
            else:
                logger.warning(f"No definitions found in definitions_df matching ordered_definition_original_ids for '{ntype_def}'. Features will be placeholder.")

        # Assemble definition features or create placeholder
        if def_feature_parts: # Contains ST embeddings if successful and aligned
            final_features[ntype_def] = def_feature_parts[0].float() # Assuming ST is the sole primary feature
            logger.info(f"Final '{ntype_def}' features (from ST) assembled. Shape: {final_features[ntype_def].shape}")
        elif num_def_nodes > 0: # No usable ST features, create placeholder
            logger.warning(f"No usable SentenceTransformer features were generated for '{ntype_def}'. Creating placeholder.")
            target_dim_def = self.model_original_feat_dims.get(ntype_def)
            if target_dim_def is None:
                if self.use_transformer_embeddings and self.transformer_model and self.transformer_embedding_dim > 0:
                    target_dim_def = self.transformer_embedding_dim
                    logger.info(f"'{ntype_def}' placeholder dim set to ST model dim: {target_dim_def} (not specified in model_original_feat_dims).")
                else:
                    target_dim_def = 1 # Fallback dimension
                    logger.info(f"'{ntype_def}' placeholder dim defaulted to 1 (not in model_original_feat_dims, ST unavailable/unconfigured).")
            else:
                 logger.info(f"'{ntype_def}' placeholder dim from model_original_feat_dims: {target_dim_def}.")
            final_features[ntype_def] = torch.zeros((num_def_nodes, target_dim_def), device=target_device).float()
            logger.info(f"Placeholder for '{ntype_def}' created with shape: {final_features[ntype_def].shape}")
        else: # num_def_nodes is 0
            target_dim_def_fallback = 1
            if self.use_transformer_embeddings and self.transformer_model and self.transformer_embedding_dim > 0: 
                target_dim_def_fallback = self.transformer_embedding_dim
            target_dim_def = self.model_original_feat_dims.get(ntype_def, target_dim_def_fallback)
            final_features[ntype_def] = torch.zeros((0, target_dim_def), device=target_device).float()
        
        logger.info(f"Final '{ntype_def}' features tensor shape: {final_features[ntype_def].shape if ntype_def in final_features else 'Not Created (0 nodes?)'}")

        # --- ETYMOLOGY FEATURES ---
        ntype_etym = 'etymology'
        logger.info(f"Processing features for node type: {ntype_etym}")
        num_etym_nodes = graph_num_nodes_per_type.get(ntype_etym, 0)
        etym_node_to_original_id_map = node_to_original_id_maps.get(ntype_etym, {}) if node_to_original_id_maps else {}
        
        # Corrected: The third argument should be ids_from_primary_extraction
        # relevant_etymology_ids contains the set of original etymology IDs that are relevant.
        # This should be converted to a list for _determine_ordered_ids_for_ntype if it's the primary source.
        primary_etym_ids_list = list(relevant_etymology_ids) if relevant_etymology_ids is not None else []

        ordered_etym_original_ids = self._determine_ordered_ids_for_ntype(
            ntype_etym, 
            num_etym_nodes, 
            ids_from_primary_extraction=primary_etym_ids_list, # Corrected argument passing
            ntype_specific_node_to_original_id_map=etym_node_to_original_id_map, 
            logger=logger
        )
        final_ordered_original_ids_map[ntype_etym] = ordered_etym_original_ids
        
        etym_feature_parts = []
        if num_etym_nodes > 0 and ordered_etym_original_ids and etymologies_df is not None and not etymologies_df.empty:
            # Align etymologies_df with the order of nodes in the graph
            ordered_ids_df_etym = pd.DataFrame({'original_id': ordered_etym_original_ids, 'graph_order': range(len(ordered_etym_original_ids))})
            aligned_etymologies_for_features = pd.merge(etymologies_df, ordered_ids_df_etym, left_on='id', right_on='original_id', how='inner')
            aligned_etymologies_for_features = aligned_etymologies_for_features.sort_values(by='graph_order').reset_index(drop=True)

            if not aligned_etymologies_for_features.empty:
                # Determine the correct column for etymology text. Prioritize 'etymology_text'.
                etym_text_col = None
                for col_candidate in ['etymology_text', 'text', 'etymology', 'content', 'summary', 'description']:
                    if col_candidate in aligned_etymologies_for_features.columns:
                        etym_text_col = col_candidate
                        break
                
                if etym_text_col:
                    logger.info(f"Using column '{etym_text_col}' from etymologies_df for etymology texts.")
                    etymology_texts = aligned_etymologies_for_features[etym_text_col].fillna('').astype(str).tolist()
                    
                    # ---- START DIAGNOSTIC LOGGING FOR ETYMOLOGY ----
                    logger.info(f"LFE_PRE_ST_ETYM: About to call ST for 'etymology'. Num texts: {len(etymology_texts)}. Sample texts: {[t[:50] + '...' if len(t) > 50 else t for t in etymology_texts[:min(3, len(etymology_texts))]]}")
                    logger.info(f"LFE_PRE_ST_ETYM: self.use_transformer_embeddings: {self.use_transformer_embeddings}")
                    logger.info(f"LFE_PRE_ST_ETYM: self.transformer_model is None: {self.transformer_model is None}")
                    if self.transformer_model:
                        logger.info(f"LFE_PRE_ST_ETYM: self.transformer_model.device: {self.transformer_model.device if hasattr(self.transformer_model, 'device') else 'N/A'}")
                    logger.info(f"LFE_PRE_ST_ETYM: aligned_etymologies_for_features shape: {aligned_etymologies_for_features.shape}")
                    # ---- END DIAGNOSTIC LOGGING FOR ETYMOLOGY ----

                    if self.use_transformer_embeddings and self.transformer_model:
                        logger.info(f"Extracting SentenceTransformer embeddings for {len(etymology_texts)} '{ntype_etym}' texts.")
                        st_etym_embeddings = self.extract_transformer_embeddings(etymology_texts)
                        if st_etym_embeddings is not None and st_etym_embeddings.nelement() > 0:
                            # Ensure the ST embeddings have the correct number of rows (num_etym_nodes)
                            # This should be guaranteed if aligned_etymologies_for_features was correctly constructed
                            # and etymology_texts came from it.
                            if st_etym_embeddings.shape[0] == num_etym_nodes:
                                etym_feature_parts.append(st_etym_embeddings.to(target_device))
                                logger.info(f"SentenceTransformer '{ntype_etym}' embeddings shape: {st_etym_embeddings.shape}")
                        else:
                            logger.warning(f"SentenceTransformer embeddings for '{ntype_etym}' are None or empty.")
                    
                    # Optionally, add other feature types for etymologies (e.g., char n-grams from etymology text)
                    # if self.use_char_ngrams_for_etymologies: # Example of a new config
                    #     char_ngram_etym_features = self.extract_char_ngrams(etymology_texts)
                    #     etym_feature_parts.append(char_ngram_etym_features.to(target_device))
                else:
                    logger.warning(f"Could not find a suitable text column ('text', 'etymology', 'content') in etymologies_df for '{ntype_etym}'. Transformer embeddings will be skipped.")
            else:
                logger.warning(f"No etymologies found in etymologies_df that match the ordered_etym_original_ids for '{ntype_etym}'. Transformer features will be skipped/empty.")

        # Assemble etymology features or create placeholder
        if etym_feature_parts:
            processed_etym_feature_parts = []
            for i, part in enumerate(etym_feature_parts):
                if part.ndim == 1:
                    part = part.unsqueeze(1)
                if part.shape[0] != num_etym_nodes: # Strict check for alignment
                    logger.warning(f"Etymology feature part {i} has {part.shape[0]} rows, graph expects {num_etym_nodes}. Skipping this part due to misalignment.")
                    continue
                processed_etym_feature_parts.append(part)

            if processed_etym_feature_parts:
                final_features[ntype_etym] = torch.cat(processed_etym_feature_parts, dim=1).float()
                logger.info(f"Final '{ntype_etym}' features assembled. Shape: {final_features[ntype_etym].shape}")
            else:
                logger.warning(f"All etymology feature parts were empty or misaligned. Creating placeholder for '{ntype_etym}'.")
                target_dim_etym = self.model_original_feat_dims.get(ntype_etym, self.transformer_embedding_dim if self.use_transformer_embeddings and self.transformer_model else 1)
                final_features[ntype_etym] = torch.zeros((num_etym_nodes, target_dim_etym), device=target_device).float()
        elif num_etym_nodes > 0: # No feature parts, but nodes exist
            logger.warning(f"No feature parts generated for '{ntype_etym}'. Creating placeholder.")
            # Use transformer_embedding_dim as a sensible default if ST was intended.
            target_dim_etym = self.model_original_feat_dims.get(ntype_etym, self.transformer_embedding_dim if self.use_transformer_embeddings and self.transformer_model else 1)
            final_features[ntype_etym] = torch.zeros((num_etym_nodes, target_dim_etym), device=target_device).float()
            logger.info(f"Placeholder for '{ntype_etym}' created with target_dim: {target_dim_etym}, shape: {final_features[ntype_etym].shape}")
        else: # num_etym_nodes is 0
            target_dim_etym = self.model_original_feat_dims.get(ntype_etym, self.transformer_embedding_dim if self.use_transformer_embeddings and self.transformer_model else 1)
            final_features[ntype_etym] = torch.zeros((0, target_dim_etym), device=target_device).float()
        logger.info(f"Final '{ntype_etym}' features tensor shape: {final_features[ntype_etym].shape if ntype_etym in final_features else 'Not Created (0 nodes?)'}")

        logger.info(f"Final feature extraction complete. Returning features for types: {list(final_features.keys())}")
        for ntype, f_tensor in final_features.items():
            logger.info(f"  Type '{ntype}': Shape {f_tensor.shape}, Corresponding IDs in map: {len(final_ordered_original_ids_map.get(ntype, []))}")
            if len(final_ordered_original_ids_map.get(ntype, [])) != f_tensor.shape[0]:
                logger.error(f"  FATAL MISMATCH for '{ntype}': Final tensor shape {f_tensor.shape[0]} != ordered_original_ids_map length {len(final_ordered_original_ids_map.get(ntype, []))}")

        return final_features, final_ordered_original_ids_map
    
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

    def _determine_ordered_ids_for_ntype(self, ntype, num_nodes_in_graph, ids_from_primary_extraction, ntype_specific_node_to_original_id_map, logger):
        """
        Determines the ordered list of original database IDs for a given node type.
        Tries multiple strategies:
        1. ids_from_primary_extraction: If features were successfully extracted AND their count matches graph nodes.
        2. ntype_specific_node_to_original_id_map: From graph_data, if it maps all graph nodes to DB IDs.
        3. Fallback: Sequential graph indices (0 to N-1), which is usually incorrect for DB lookups.
        """
        final_ordered_ids = None
        # numpy import should be at the top of the file, but np.integer is used here.

        # Condition A: Use ids_from_primary_extraction (typically from feature_extraction function)
        condition_A_met = False
        if ids_from_primary_extraction is not None:
            if len(ids_from_primary_extraction) == num_nodes_in_graph:
                if all(isinstance(x, (int, np.integer)) for x in ids_from_primary_extraction):
                    condition_A_met = True
                    logger.info(f"LFE_ID_DEBUG ('{ntype}'): Condition A (primary IDs): Eligible. Length matches num_nodes ({num_nodes_in_graph}) and types valid.")
                else:
                    logger.warning(f"LFE_ID_DEBUG ('{ntype}'): Condition A (primary IDs): Not all IDs are int/np.integer. Length: {len(ids_from_primary_extraction)} vs Graph: {num_nodes_in_graph}.")
            else:
                logger.warning(f"LFE_ID_DEBUG ('{ntype}'): Condition A (primary IDs): Length mismatch. IDs: {len(ids_from_primary_extraction)}, Graph: {num_nodes_in_graph}.")
        else:
            logger.info(f"LFE_ID_DEBUG ('{ntype}'): Condition A (primary IDs): Not available.")

        # Condition B: Use ntype_specific_node_to_original_id_map from graph builder (graph_data)
        condition_B_met = False
        if not condition_A_met:
            can_use_graph_builder_map = False
            if (ntype_specific_node_to_original_id_map is not None and
                isinstance(ntype_specific_node_to_original_id_map, dict)):
                logger.info(f"LFE_ID_DEBUG ('{ntype}'): Condition B: Graph builder map is a dict. Map Length: {len(ntype_specific_node_to_original_id_map)}. Graph nodes: {num_nodes_in_graph}.")
                if num_nodes_in_graph == 0:
                    can_use_graph_builder_map = True
                    logger.info(f"LFE_ID_DEBUG ('{ntype}'): Condition B: Eligible (0 nodes in graph for this type).")
                elif num_nodes_in_graph > 0:
                    all_indices_present = all(i in ntype_specific_node_to_original_id_map for i in range(num_nodes_in_graph))
                    if all_indices_present:
                        logger.info(f"LFE_ID_DEBUG ('{ntype}'): Condition B: Graph builder map keys cover all {num_nodes_in_graph} graph indices.")
                        sample_values_valid = True
                        num_samples_to_check = min(num_nodes_in_graph, 5)
                        for i in range(num_samples_to_check):
                            val = ntype_specific_node_to_original_id_map[i]
                            if not isinstance(val, (int, np.integer)):
                                sample_values_valid = False
                                logger.warning(f"LFE_ID_MAP_REVISED ('{ntype}'): Graph builder map value for key {i} (DB ID) is not int/np.integer (type: {type(val)}). Map cannot be used.")
                                break
                        if sample_values_valid:
                            can_use_graph_builder_map = True
                            logger.info(f"LFE_ID_DEBUG ('{ntype}'): Condition B: Eligible. All graph indices covered by map keys, sample values are valid types.")
                    else:
                        logger.warning(f"LFE_ID_MAP_REVISED ('{ntype}'): Graph builder map does not contain all required graph indices [0...{num_nodes_in_graph-1}] as keys. Map length: {len(ntype_specific_node_to_original_id_map)}. Cannot use map.")
            elif ntype_specific_node_to_original_id_map is None:
                logger.info(f"LFE_ID_DEBUG ('{ntype}'): Condition B: Graph builder map not available.")
            else: # Not None and not a dict
                logger.warning(f"LFE_ID_DEBUG ('{ntype}'): Condition B: Graph builder map is not a dict (type: {type(ntype_specific_node_to_original_id_map)}). Cannot use map.")
            condition_B_met = can_use_graph_builder_map

        # Decision logic
        if condition_A_met:
            logger.info(f"LFE_ID_MAP_REVISED: Using 'ids_from_primary_extraction' for '{ntype}' (length {len(ids_from_primary_extraction)}).")
            final_ordered_ids = list(ids_from_primary_extraction)
        elif condition_B_met:
            logger.info(f"LFE_ID_MAP_REVISED: Using 'ntype_specific_node_to_original_id_map' (from graph_data) for '{ntype}'. Map length {len(ntype_specific_node_to_original_id_map)}, graph nodes {num_nodes_in_graph}.")
            candidate_ids = None
            try:
                candidate_ids = [ntype_specific_node_to_original_id_map[i] for i in range(num_nodes_in_graph)]
                if len(candidate_ids) != num_nodes_in_graph:
                    logger.error(f"LFE_ID_MAP_REVISED CRITICAL ('{ntype}'): Mismatch creating IDs from graph_builder_map. Expected {num_nodes_in_graph}, got {len(candidate_ids)}. Will fallback.")
                    candidate_ids = None
                elif not all(isinstance(x, (int, np.integer)) for x in candidate_ids):
                    logger.error(f"LFE_ID_MAP_REVISED CRITICAL ('{ntype}'): Not all IDs from graph_builder_map are int/np.integer. Will fallback.")
                    candidate_ids = None
            except KeyError as e:
                logger.error(f"LFE_ID_MAP_REVISED CRITICAL ('{ntype}'): KeyError accessing graph_builder_map for index {e}. Fallback will be triggered.")
                candidate_ids = None
            except Exception as e:
                logger.error(f"LFE_ID_MAP_REVISED CRITICAL ('{ntype}'): Unexpected error creating IDs from graph_builder_map: {str(e)}. Fallback will be triggered.")
                candidate_ids = None
            
            if candidate_ids is not None:
                final_ordered_ids = candidate_ids
            else:
                condition_B_met = False # Mark B as failed for the check below
                logger.warning(f"LFE_ID_MAP_REVISED ('{ntype}'): Failed to reliably use graph_builder_map, will attempt fallback to sequential IDs.")

        # Fallback C: If neither A nor B (successfully) provided IDs
        if final_ordered_ids is None:
            logger.error(f"LFE_ID_MAP_REVISED: CRITICAL FALLBACK! For '{ntype}', failed to get valid original DB IDs. "
                         f"Num_nodes: {num_nodes_in_graph}. Condition A met: {condition_A_met}. Condition B (attempted & succeeded): {condition_B_met and final_ordered_ids is not None}. "
                         f"Using sequential graph indices (0 to N-1) which is likely INCORRECT for DB lookups/linking.")
            final_ordered_ids = list(range(num_nodes_in_graph))

        # Final validation and logging
        if not (isinstance(final_ordered_ids, list) and all(isinstance(x, (int, np.integer)) for x in final_ordered_ids) and len(final_ordered_ids) == num_nodes_in_graph):
            logger.error(f"LFE_ID_MAP_REVISED: CRITICAL FAILURE! For '{ntype}', final_ordered_ids is invalid (type: {type(final_ordered_ids)}, length: {len(final_ordered_ids) if isinstance(final_ordered_ids, list) else 'N/A'}, content problem, or num_nodes mismatch {num_nodes_in_graph}). Re-falling back to sequential IDs as absolute failsafe.")
            final_ordered_ids = list(range(num_nodes_in_graph))
        
        logger.info(f"LFE_ID_DEBUG ('{ntype}'): Final ordered_ids_map sample (first 5 of {len(final_ordered_ids)}): {final_ordered_ids[:5]}")

        return final_ordered_ids 