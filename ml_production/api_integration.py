#!/usr/bin/env python3
"""
API Integration for Enhanced ML System
"""

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
import glob

logger = logging.getLogger(__name__)

class EnhancedSemanticAPI:
    def __init__(self):
        self.model_data = None
        self.load_latest_model()
    
    def load_latest_model(self):
        try:
            model_files = glob.glob('ml_models/enhanced_semantic_model_*.pkl')
            if not model_files:
                return False
            
            latest_model = max(model_files, key=os.path.getctime)
            with open(latest_model, 'rb') as f:
                self.model_data = pickle.load(f)
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def find_semantic_neighbors(self, target_word, top_k=10):
        if not self.model_data:
            return {'error': 'Model not loaded'}
        
        if target_word.lower() not in self.model_data['word_list']:
            return {'error': f'Word "{target_word}" not found'}
        
        idx = self.model_data['word_list'].index(target_word.lower())
        target_emb = self.model_data['embeddings'][idx].toarray().flatten()
        
        similarities = []
        for i, word in enumerate(self.model_data['word_list']):
            if word != target_word.lower():
                word_emb = self.model_data['embeddings'][i].toarray().flatten()
                sim = cosine_similarity([target_emb], [word_emb])[0, 0]
                similarities.append({'word': word, 'similarity': float(sim)})
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return {'target_word': target_word, 'neighbors': similarities[:top_k]}

enhanced_api = EnhancedSemanticAPI()
