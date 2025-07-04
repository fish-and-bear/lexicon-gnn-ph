#!/usr/bin/env python3
"""
Semantic Relationship Demo
Interactive demonstration of enhanced semantic understanding
"""

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import psycopg2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_enhanced_model():
    """Load the latest enhanced model."""
    import glob
    import os
    
    model_files = glob.glob('ml_models/enhanced_semantic_model_*.pkl')
    if not model_files:
        logger.error("No enhanced model found! Run enhanced_semantic_trainer.py first.")
        return None
    
    latest_model = max(model_files, key=os.path.getctime)
    logger.info(f"Loading model: {latest_model}")
    
    with open(latest_model, 'rb') as f:
        return pickle.load(f)

def connect_db():
    """Connect to database."""
    with open('ml/db_config.json', 'r') as f:
        config = json.load(f)['database']
    return psycopg2.connect(**{k: v for k, v in config.items() if k != 'ssl_mode'})

def get_word_embedding(word, model_data):
    """Get embedding for a word."""
    if word not in model_data['word_list']:
        return None
    
    idx = model_data['word_list'].index(word)
    return model_data['embeddings'][idx].toarray().flatten()

def find_semantic_neighbors(target_word, model_data, top_k=10):
    """Find semantic neighbors for a word."""
    target_embedding = get_word_embedding(target_word, model_data)
    if target_embedding is None:
        return []
    
    similarities = []
    for i, word in enumerate(model_data['word_list']):
        if word != target_word:
            word_embedding = model_data['embeddings'][i].toarray().flatten()
            sim = cosine_similarity([target_embedding], [word_embedding])[0, 0]
            similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def analyze_word_relationships(word, model_data):
    """Analyze relationships for a specific word."""
    logger.info(f"\n🔍 ANALYZING WORD: '{word}'")
    
    # Get known relationships from database
    conn = connect_db()
    query = """
        SELECT w2.lemma, r.relation_type, w2.language_code
        FROM relations r
        JOIN words w1 ON r.from_word_id = w1.id
        JOIN words w2 ON r.to_word_id = w2.id
        WHERE w1.lemma = %s
        ORDER BY r.relation_type;
    """
    
    cursor = conn.cursor()
    cursor.execute(query, (word,))
    known_relationships = cursor.fetchall()
    conn.close()
    
    # Show known relationships
    if known_relationships:
        logger.info("📚 KNOWN RELATIONSHIPS:")
        for related_word, rel_type, lang in known_relationships:
            logger.info(f"  {rel_type}: {related_word} ({lang})")
    else:
        logger.info("📚 No known relationships in database")
    
    # Find semantic neighbors
    neighbors = find_semantic_neighbors(word, model_data)
    if neighbors:
        logger.info("\n🤖 AI DISCOVERED RELATIONSHIPS:")
        for neighbor_word, similarity in neighbors:
            logger.info(f"  {similarity:.3f}: {neighbor_word}")
    else:
        logger.info("\n🤖 Word not found in model vocabulary")

def linguistic_test_cases(model_data):
    """Run specific linguistic test cases."""
    logger.info("\n🧪 LINGUISTIC TEST CASES")
    
    test_cases = [
        {
            'name': 'Filipino Morphological Variants',
            'pairs': [
                ('lakad', 'maglakad'),
                ('ganda', 'maganda'),
                ('sulat', 'magsulat'),
                ('kain', 'kumain')
            ]
        },
        {
            'name': 'Spanish Loanwords',
            'pairs': [
                ('mesa', 'table'),
                ('libro', 'book'),
                ('casa', 'house'),
                ('tiempo', 'time')
            ]
        },
        {
            'name': 'English-Filipino Translation',
            'pairs': [
                ('love', 'pag-ibig'),
                ('water', 'tubig'),
                ('food', 'pagkain'),
                ('house', 'bahay')
            ]
        },
        {
            'name': 'Synonyms and Variants',
            'pairs': [
                ('bahay', 'tahanan'),
                ('ganda', 'kagandahan'),
                ('mabait', 'mabuting-loob')
            ]
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n--- {test_case['name']} ---")
        
        for word1, word2 in test_case['pairs']:
            emb1 = get_word_embedding(word1, model_data)
            emb2 = get_word_embedding(word2, model_data)
            
            if emb1 is not None and emb2 is not None:
                similarity = cosine_similarity([emb1], [emb2])[0, 0]
                status = "✅" if similarity > 0.15 else "❌"
                logger.info(f"  {word1} ↔ {word2}: {similarity:.3f} {status}")
            else:
                missing = []
                if emb1 is None: missing.append(word1)
                if emb2 is None: missing.append(word2)
                logger.info(f"  {word1} ↔ {word2}: MISSING ({', '.join(missing)})")

def interactive_exploration(model_data):
    """Interactive word exploration."""
    logger.info("\n🎯 INTERACTIVE EXPLORATION")
    logger.info("Enter words to explore their semantic relationships (or 'quit' to exit)")
    
    while True:
        try:
            word = input("\nEnter word: ").strip().lower()
            if word == 'quit':
                break
            
            if word:
                analyze_word_relationships(word, model_data)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")

def main():
    """Main demo function."""
    logger.info("🚀 Enhanced Semantic Relationship Demo")
    
    # Load model
    model_data = load_enhanced_model()
    if not model_data:
        return
    
    # Show model info
    logger.info(f"📊 Model loaded:")
    logger.info(f"  - Vocabulary size: {len(model_data['word_list'])}")
    logger.info(f"  - Feature dimensions: {model_data['embeddings'].shape[1]}")
    logger.info(f"  - Training timestamp: {model_data['timestamp']}")
    
    # Show evaluation results
    if 'evaluation_results' in model_data:
        results = model_data['evaluation_results']
        if 'overall_weighted_score' in results:
            score = results['overall_weighted_score']
            logger.info(f"  - Overall performance: {score:.3f}")
    
    # Run test cases
    linguistic_test_cases(model_data)
    
    # Interactive exploration
    interactive_exploration(model_data)
    
    logger.info("👋 Demo completed!")

if __name__ == '__main__':
    main() 