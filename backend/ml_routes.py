"""
ML API routes for the Filipino Dictionary application.
This module provides ML-powered endpoints for semantic analysis, word similarity, and relationship prediction.
"""

from flask import Blueprint, request, jsonify, current_app, g
from sqlalchemy import text
from marshmallow import Schema, fields, validate, ValidationError
from datetime import datetime, timezone
import structlog
import logging
import time
import os
import sys
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import pickle
import json

# Set up logging first
logger = structlog.get_logger(__name__)

# Import our finalized production ML system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_production'))
try:
    # Import finalized semantic API
    from api_integration import enhanced_api
    ML_SYSTEM_AVAILABLE = True
    logger.info("✅ Production ML system loaded from ml_production/")
except ImportError as e:
    # Set up basic logger for import errors
    basic_logger = logging.getLogger(__name__)
    basic_logger.error(f"Failed to import production ML system: {e}")
    ML_SYSTEM_AVAILABLE = False

from .database import db
from .extensions import limiter
from .utils.ip import get_remote_address
from .metrics import API_REQUESTS, API_ERRORS, REQUEST_LATENCY

# Initialize blueprint
ml_bp = Blueprint("ml_api", __name__, url_prefix='/api/v2/ml')

# Global enhanced ML system instance
_enhanced_ml_system = None
_enhanced_ml_system_initialized = False

def get_enhanced_ml_system():
    """Get or initialize the enhanced ML system instance."""
    global _enhanced_ml_system, _enhanced_ml_system_initialized
    
    if not ML_SYSTEM_AVAILABLE:
        raise RuntimeError("Enhanced ML system is not available")
    
    if _enhanced_ml_system is None and not _enhanced_ml_system_initialized:
        try:
            logger.info("Initializing enhanced ML system...")
            _enhanced_ml_system = enhanced_api
            _enhanced_ml_system_initialized = True
            logger.info("Enhanced ML system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced ML system: {e}")
            _enhanced_ml_system_initialized = True  # Mark as attempted to avoid retrying
            raise
    
    if _enhanced_ml_system is None:
        raise RuntimeError("Enhanced ML system failed to initialize")
    
    return _enhanced_ml_system

# Schemas for request/response validation
class WordSimilarityQuerySchema(Schema):
    """Schema for word similarity queries."""
    word = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    limit = fields.Int(validate=validate.Range(min=1, max=50), load_default=10)
    threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0), load_default=0.1)
    include_scores = fields.Bool(load_default=True)
    languages = fields.List(fields.Str(), load_default=None)

class RelationshipPredictionSchema(Schema):
    """Schema for relationship prediction queries."""
    word1 = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    word2 = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    relation_types = fields.List(fields.Str(), load_default=None)
    threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0), load_default=0.5)

class SemanticAnalysisSchema(Schema):
    """Schema for semantic analysis queries."""
    words = fields.List(fields.Str(), required=True, validate=validate.Length(min=1, max=20))
    analysis_type = fields.Str(validate=validate.OneOf(['similarity', 'clustering', 'embeddings']), load_default='similarity')
    include_visualizations = fields.Bool(load_default=False)

class BatchSimilaritySchema(Schema):
    """Schema for batch similarity queries."""
    words = fields.List(fields.Str(), required=True, validate=validate.Length(min=2, max=100))
    mode = fields.Str(validate=validate.OneOf(['all_pairs', 'to_first']), load_default='all_pairs')
    threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0), load_default=0.1)

# Error handlers
@ml_bp.errorhandler(ValidationError)
def handle_validation_error(error):
    API_ERRORS.labels(endpoint=request.endpoint or '', error_type='validation_error').inc()
    return jsonify({"error": "Validation error", "details": error.messages}), 400

@ml_bp.errorhandler(RuntimeError)
def handle_runtime_error(error):
    API_ERRORS.labels(endpoint=request.endpoint or '', error_type='runtime_error').inc()
    logger.error(f"Runtime error in ML endpoint: {str(error)}")
    return jsonify({"error": "ML system error", "message": str(error)}), 503

# ML API Endpoints

@ml_bp.route('/status', methods=['GET'])
def ml_status():
    """Get enhanced ML system status and health information."""
    try:
        API_REQUESTS.labels(endpoint="ml_status", method="GET").inc()
        start_time = time.time()
        
        status_info = {
            "ml_system_available": ML_SYSTEM_AVAILABLE,
            "initialized": _enhanced_ml_system_initialized,
            "ready": _enhanced_ml_system is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_type": "enhanced_semantic_tfidf"
        }
        
        if ML_SYSTEM_AVAILABLE and _enhanced_ml_system is not None:
            try:
                ml_system = get_enhanced_ml_system()
                # Get enhanced system stats
                if hasattr(ml_system, 'model_data') and ml_system.model_data is not None:
                    status_info.update({
                        "model_loaded": True,
                        "vocab_size": len(ml_system.model_data.get('word_list', [])),
                        "feature_dimensions": ml_system.model_data.get('embeddings', {}).shape[1] if hasattr(ml_system.model_data.get('embeddings', {}), 'shape') else 0,
                        "model_timestamp": ml_system.model_data.get('timestamp', 'unknown'),
                        "performance_score": ml_system.model_data.get('evaluation_results', {}).get('overall_weighted_score', 0),
                        "model_type": ml_system.model_data.get('model_type', 'enhanced_multi_dimensional')
                    })
                else:
                    status_info.update({
                        "model_loaded": False,
                        "error": "Enhanced model not loaded"
                    })
            except Exception as e:
                status_info["error"] = str(e)
                status_info["ready"] = False
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="ml_status").observe(execution_time)
        
        status_code = 200 if status_info.get("ready", False) else 503
        return jsonify(status_info), status_code
        
    except Exception as e:
        logger.error(f"Error getting enhanced ML status: {e}")
        return jsonify({"error": "Failed to get ML status", "message": str(e)}), 500

@ml_bp.route('/similarity', methods=['POST'])
@limiter.limit("30 per minute", key_func=get_remote_address)
def word_similarity():
    """Find words similar to a given word using enhanced ML embeddings."""
    try:
        API_REQUESTS.labels(endpoint="word_similarity", method="POST").inc()
        start_time = time.time()
        
        # Validate request
        schema = WordSimilarityQuerySchema()
        try:
            params = schema.load(request.json or {})
        except ValidationError as e:
            return jsonify({"error": "Invalid request", "details": e.messages}), 400
        
        # Get enhanced ML system
        ml_system = get_enhanced_ml_system()
        
        word = params['word'].lower().strip()
        limit = params['limit']
        threshold = params['threshold']
        include_scores = params['include_scores']
        languages = params.get('languages')
        
        # Use enhanced API to find semantic neighbors
        result = ml_system.find_semantic_neighbors(word, top_k=limit, min_similarity=threshold)
        
        if 'error' in result:
            return jsonify(result), 404
        
        # Format results for API response
        similar_words = []
        for neighbor in result['neighbors']:
            # Language filtering if specified
            if languages:
                # Get word language from database
                word_lang_query = text("SELECT language_code FROM words WHERE LOWER(lemma) = :word LIMIT 1")
                db_result = db.session.execute(word_lang_query, {"word": neighbor['word']}).fetchone()
                if db_result and db_result.language_code not in languages:
                    continue
            
            result_item = {"word": neighbor['word']}
            if include_scores:
                result_item["similarity_score"] = neighbor['similarity']
                result_item["confidence"] = neighbor.get('confidence', 'medium')
                
            similar_words.append(result_item)
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="word_similarity").observe(execution_time)
        
        return jsonify({
            "query_word": word,
            "similar_words": similar_words,
            "total_found": len(similar_words),
            "total_available": result.get('total_found', len(similar_words)),
            "execution_time": execution_time,
            "model_type": "enhanced_semantic"
        })
        
    except Exception as e:
        API_ERRORS.labels(endpoint="word_similarity", error_type=type(e).__name__).inc()
        logger.error(f"Error in word similarity: {e}")
        return jsonify({"error": "Word similarity failed", "message": str(e)}), 500

@ml_bp.route('/relationships/predict', methods=['POST'])
@limiter.limit("20 per minute", key_func=get_remote_address)
def predict_relationship():
    """Predict the likelihood of relationships between two words."""
    try:
        API_REQUESTS.labels(endpoint="predict_relationship", method="POST").inc()
        start_time = time.time()
        
        # Validate request
        schema = RelationshipPredictionSchema()
        try:
            params = schema.load(request.json or {})
        except ValidationError as e:
            return jsonify({"error": "Invalid request", "details": e.messages}), 400
        
        # Get ML system
        ml_system = get_enhanced_ml_system()
        
        word1 = params['word1'].lower().strip()
        word2 = params['word2'].lower().strip()
        threshold = params['threshold']
        
        # Check if words exist in vocabulary
        if word1 not in ml_system.word_to_idx:
            return jsonify({"error": f"Word '{word1}' not found in vocabulary"}), 404
        if word2 not in ml_system.word_to_idx:
            return jsonify({"error": f"Word '{word2}' not found in vocabulary"}), 404
        
        # Get embeddings
        word1_idx = ml_system.word_to_idx[word1]
        word2_idx = ml_system.word_to_idx[word2]
        
        emb1 = ml_system.embeddings[word1_idx]
        emb2 = ml_system.embeddings[word2_idx]
        
        # Calculate similarity
        similarity = float(torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)))
        
        # Check existing relationships in database
        existing_relations_query = text("""
            SELECT r.relation_type, rt.name 
            FROM relations r
            JOIN words w1 ON r.word_id = w1.id
            JOIN words w2 ON r.related_word_id = w2.id
            LEFT JOIN relationship_types rt ON r.relation_type = rt.code
            WHERE (LOWER(w1.lemma) = :word1 AND LOWER(w2.lemma) = :word2)
               OR (LOWER(w1.lemma) = :word2 AND LOWER(w2.lemma) = :word1)
        """)
        
        existing_relations = db.session.execute(existing_relations_query, {
            "word1": word1, "word2": word2
        }).fetchall()
        
        # Predict relationship types based on similarity
        predictions = []
        if similarity > threshold:
            # Basic relationship type prediction based on similarity ranges
            if similarity > 0.8:
                predictions.append({"type": "synonym", "confidence": similarity})
            elif similarity > 0.6:
                predictions.append({"type": "related", "confidence": similarity})
            elif similarity > 0.4:
                predictions.append({"type": "semantic_field", "confidence": similarity})
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="predict_relationship").observe(execution_time)
        
        return jsonify({
            "word1": word1,
            "word2": word2,
            "similarity_score": similarity,
            "existing_relationships": [
                {"type": rel.relation_type, "name": rel.name} 
                for rel in existing_relations
            ],
            "predicted_relationships": predictions,
            "execution_time": execution_time
        })
        
    except Exception as e:
        API_ERRORS.labels(endpoint="predict_relationship", error_type=type(e).__name__).inc()
        logger.error(f"Error in relationship prediction: {e}")
        return jsonify({"error": "Relationship prediction failed", "message": str(e)}), 500

@ml_bp.route('/analysis/semantic', methods=['POST'])
@limiter.limit("15 per minute", key_func=get_remote_address)
def semantic_analysis():
    """Perform semantic analysis on a set of words."""
    try:
        API_REQUESTS.labels(endpoint="semantic_analysis", method="POST").inc()
        start_time = time.time()
        
        # Validate request
        schema = SemanticAnalysisSchema()
        try:
            params = schema.load(request.json or {})
        except ValidationError as e:
            return jsonify({"error": "Invalid request", "details": e.messages}), 400
        
        # Get ML system
        ml_system = get_enhanced_ml_system()
        
        words = [w.lower().strip() for w in params['words']]
        analysis_type = params['analysis_type']
        
        # Filter words that exist in vocabulary
        valid_words = []
        missing_words = []
        
        for word in words:
            if word in ml_system.word_to_idx:
                valid_words.append(word)
            else:
                missing_words.append(word)
        
        if not valid_words:
            return jsonify({
                "error": "No valid words found",
                "missing_words": missing_words
            }), 400
        
        # Get embeddings for valid words
        word_indices = [ml_system.word_to_idx[word] for word in valid_words]
        embeddings = ml_system.embeddings[word_indices]
        
        results = {"words": valid_words, "missing_words": missing_words}
        
        if analysis_type == 'similarity':
            # Calculate pairwise similarities
            similarity_matrix = torch.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
            ).cpu().numpy()
            
            results["similarity_matrix"] = similarity_matrix.tolist()
            
        elif analysis_type == 'clustering':
            # Simple clustering using similarity
            from sklearn.cluster import AgglomerativeClustering
            
            # Convert to distance matrix
            similarity_matrix = torch.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
            ).cpu().numpy()
            distance_matrix = 1 - similarity_matrix
            
            n_clusters = min(3, len(valid_words))
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            results["clusters"] = {
                "labels": cluster_labels.tolist(),
                "n_clusters": n_clusters,
                "cluster_words": {}
            }
            
            for i, word in enumerate(valid_words):
                cluster_id = int(cluster_labels[i])
                if cluster_id not in results["clusters"]["cluster_words"]:
                    results["clusters"]["cluster_words"][cluster_id] = []
                results["clusters"]["cluster_words"][cluster_id].append(word)
                
        elif analysis_type == 'embeddings':
            # Return raw embeddings (truncated for API response)
            results["embeddings"] = embeddings.cpu().numpy()[:, :50].tolist()  # First 50 dimensions
            results["embedding_dimension"] = embeddings.shape[1]
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="semantic_analysis").observe(execution_time)
        
        results["execution_time"] = execution_time
        return jsonify(results)
        
    except Exception as e:
        API_ERRORS.labels(endpoint="semantic_analysis", error_type=type(e).__name__).inc()
        logger.error(f"Error in semantic analysis: {e}")
        return jsonify({"error": "Semantic analysis failed", "message": str(e)}), 500

@ml_bp.route('/batch/similarity', methods=['POST'])
@limiter.limit("10 per minute", key_func=get_remote_address)
def batch_similarity():
    """Calculate similarities between multiple words efficiently."""
    try:
        API_REQUESTS.labels(endpoint="batch_similarity", method="POST").inc()
        start_time = time.time()
        
        # Validate request
        schema = BatchSimilaritySchema()
        try:
            params = schema.load(request.json or {})
        except ValidationError as e:
            return jsonify({"error": "Invalid request", "details": e.messages}), 400
        
        # Get ML system
        ml_system = get_enhanced_ml_system()
        
        words = [w.lower().strip() for w in params['words']]
        mode = params['mode']
        threshold = params['threshold']
        
        # Filter valid words
        valid_words = []
        word_indices = []
        
        for word in words:
            if word in ml_system.word_to_idx:
                valid_words.append(word)
                word_indices.append(ml_system.word_to_idx[word])
        
        if len(valid_words) < 2:
            return jsonify({"error": "Need at least 2 valid words"}), 400
        
        # Get embeddings
        embeddings = ml_system.embeddings[word_indices]
        
        results = {"words": valid_words}
        
        if mode == 'all_pairs':
            # Calculate all pairwise similarities
            similarity_matrix = torch.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
            ).cpu().numpy()
            
            pairs = []
            for i in range(len(valid_words)):
                for j in range(i + 1, len(valid_words)):
                    similarity = float(similarity_matrix[i, j])
                    if similarity >= threshold:
                        pairs.append({
                            "word1": valid_words[i],
                            "word2": valid_words[j],
                            "similarity": similarity
                        })
            
            # Sort by similarity descending
            pairs.sort(key=lambda x: x['similarity'], reverse=True)
            results["similar_pairs"] = pairs
            
        elif mode == 'to_first':
            # Calculate similarities to the first word
            first_embedding = embeddings[0].unsqueeze(0)
            similarities = torch.cosine_similarity(first_embedding, embeddings[1:])
            
            similar_words = []
            for i, similarity in enumerate(similarities):
                similarity_score = float(similarity)
                if similarity_score >= threshold:
                    similar_words.append({
                        "word": valid_words[i + 1],
                        "similarity": similarity_score
                    })
            
            # Sort by similarity descending
            similar_words.sort(key=lambda x: x['similarity'], reverse=True)
            results["reference_word"] = valid_words[0]
            results["similar_words"] = similar_words
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="batch_similarity").observe(execution_time)
        
        results["execution_time"] = execution_time
        return jsonify(results)
        
    except Exception as e:
        API_ERRORS.labels(endpoint="batch_similarity", error_type=type(e).__name__).inc()
        logger.error(f"Error in batch similarity: {e}")
        return jsonify({"error": "Batch similarity failed", "message": str(e)}), 500

@ml_bp.route('/vocabulary', methods=['GET'])
def get_vocabulary_info():
    """Get information about the ML system vocabulary."""
    try:
        API_REQUESTS.labels(endpoint="vocabulary_info", method="GET").inc()
        start_time = time.time()
        
        if not ML_SYSTEM_AVAILABLE or _enhanced_ml_system is None:
            return jsonify({"error": "ML system not available"}), 503
        
        ml_system = get_enhanced_ml_system()
        
        # Get vocabulary statistics
        vocab_size = len(ml_system.word_to_idx) if hasattr(ml_system, 'word_to_idx') else 0
        
        # Sample some words
        sample_words = []
        if hasattr(ml_system, 'idx_to_word') and vocab_size > 0:
            import random
            sample_indices = random.sample(range(min(vocab_size, 100)), min(20, vocab_size))
            sample_words = [ml_system.idx_to_word[idx] for idx in sample_indices]
        
        # Get language distribution
        language_dist = {}
        if vocab_size > 0:
            lang_query = text("""
                SELECT language_code, COUNT(*) as count
                FROM words 
                WHERE LOWER(lemma) IN :words
                GROUP BY language_code
                ORDER BY count DESC
                LIMIT 10
            """)
            
            # Use sample words to estimate language distribution
            if sample_words:
                results = db.session.execute(lang_query, {"words": tuple(sample_words)}).fetchall()
                language_dist = {row.language_code: row.count for row in results}
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="vocabulary_info").observe(execution_time)
        
        return jsonify({
            "vocabulary_size": vocab_size,
            "sample_words": sample_words,
            "language_distribution": language_dist,
            "execution_time": execution_time
        })
        
    except Exception as e:
        API_ERRORS.labels(endpoint="vocabulary_info", error_type=type(e).__name__).inc()
        logger.error(f"Error getting vocabulary info: {e}")
        return jsonify({"error": "Failed to get vocabulary info", "message": str(e)}), 500

# Note: Flask's before_app_first_request is deprecated
# ML system will be initialized lazily when first accessed 