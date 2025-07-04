#!/usr/bin/env python3
"""
Semantically-Aware Enhanced Trainer
Uses actual semantic relationships, not just spelling similarity
"""

import pandas as pd
import numpy as np
import json
import psycopg2
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from datetime import datetime
import pickle
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_db():
    with open('ml/db_config.json', 'r') as f:
        config = json.load(f)['database']
    return psycopg2.connect(**{k: v for k, v in config.items() if k != 'ssl_mode'})

def load_semantic_training_data():
    """Load training data using TRUE semantic relationships."""
    logger.info("Loading semantically-aware training data...")
    
    conn = connect_db()
    
    # Enhanced query that includes root word information for semantic validation
    query = """
        SELECT 
            w1.lemma as word1, 
            w2.lemma as word2, 
            r.relation_type,
            w1.language_code as lang1, 
            w2.language_code as lang2,
            w1_root.lemma as word1_root,
            w2_root.lemma as word2_root,
            -- Semantic validation: same root = true morphological relationship
            CASE 
                WHEN w1.root_word_id = w2.root_word_id AND w1.root_word_id IS NOT NULL THEN 'morphological'
                WHEN r.relation_type IN ('synonym', 'translation') THEN 'semantic'
                ELSE 'other'
            END as semantic_type
        FROM relations r
        JOIN words w1 ON r.from_word_id = w1.id
        JOIN words w2 ON r.to_word_id = w2.id
        LEFT JOIN words w1_root ON w1.root_word_id = w1_root.id
        LEFT JOIN words w2_root ON w2.root_word_id = w2_root.id
        WHERE r.relation_type IN ('synonym', 'translation', 'etymology', 'variant', 'related')
        AND w1.lemma IS NOT NULL AND w2.lemma IS NOT NULL
        AND LENGTH(w1.lemma) > 1 AND LENGTH(w2.lemma) > 1
        LIMIT 30000;
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} semantically-validated pairs")
    logger.info(f"Semantic types: {df['semantic_type'].value_counts().to_dict()}")
    logger.info(f"Relation types: {df['relation_type'].value_counts().to_dict()}")
    
    return df

def create_semantic_features(df):
    """Create features that emphasize semantic over orthographic similarity."""
    logger.info("Creating semantically-aware features...")
    
    all_words = list(set(df['word1'].tolist() + df['word2'].tolist()))
    logger.info(f"Semantic vocabulary: {len(all_words)} words")
    
    # 1. Semantic context features (using relation and root information)
    semantic_contexts = []
    word_to_context = {}
    
    for _, row in df.iterrows():
        # Create rich semantic context
        context1 = f"{row['word1']} {row.get('word1_root', '')} {row['lang1']} {row['semantic_type']}"
        context2 = f"{row['word2']} {row.get('word2_root', '')} {row['lang2']} {row['semantic_type']}"
        
        word_to_context[row['word1']] = context1
        word_to_context[row['word2']] = context2
    
    for word in all_words:
        context = word_to_context.get(word, f"{word} unknown unknown other")
        semantic_contexts.append(context)
    
    # Semantic vectorizer (emphasizes meaning over spelling)
    semantic_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=4000,
        lowercase=True,
        token_pattern=r'[a-zA-Z-]+'
    )
    semantic_features = semantic_vectorizer.fit_transform(semantic_contexts)
    
    # 2. Reduced morphological features (de-emphasize spelling similarity)
    morph_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),  # Longer n-grams, less spelling similarity
        max_features=2000,   # Fewer morphological features
        lowercase=True,
        sublinear_tf=True
    )
    morph_features = morph_vectorizer.fit_transform(all_words)
    
    # 3. Language family features
    lang_contexts = []
    for word in all_words:
        context = word_to_context.get(word, "")
        lang = context.split()[2] if len(context.split()) > 2 else 'unknown'
        lang_contexts.append(f"{word} {lang}")
    
    lang_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=1000,
        lowercase=True
    )
    lang_features = lang_vectorizer.fit_transform(lang_contexts)
    
    # Normalize features
    semantic_features = normalize(semantic_features, norm='l2')
    morph_features = normalize(morph_features, norm='l2')
    lang_features = normalize(lang_features, norm='l2')
    
    # SEMANTIC-FIRST combination (70% semantic, 20% morphological, 10% language)
    from scipy.sparse import hstack
    combined_features = hstack([
        semantic_features * 0.70,    # Emphasize semantic relationships
        morph_features * 0.20,       # De-emphasize spelling similarity  
        lang_features * 0.10         # Language awareness
    ])
    
    logger.info(f"Semantic-first features: {combined_features.shape}")
    logger.info(f"  - Semantic (70%): {semantic_features.shape}")
    logger.info(f"  - Morphological (20%): {morph_features.shape}")
    logger.info(f"  - Language (10%): {lang_features.shape}")
    
    return combined_features, {
        'semantic': semantic_vectorizer,
        'morph': morph_vectorizer,
        'lang': lang_vectorizer
    }, all_words

def evaluate_semantic_quality(df, embeddings, all_words):
    """Evaluate quality focusing on true semantic relationships."""
    logger.info("Evaluating semantic relationship quality...")
    
    # Test semantic vs orthographic relationships
    semantic_pairs = df[df['semantic_type'] == 'semantic']
    morphological_pairs = df[df['semantic_type'] == 'morphological'] 
    other_pairs = df[df['semantic_type'] == 'other']
    
    results = {}
    
    for pair_type, subset in [('semantic', semantic_pairs), ('morphological', morphological_pairs), ('other', other_pairs)]:
        if subset.empty:
            continue
            
        similarities = []
        examples = []
        
        for _, row in subset.iterrows():
            if row['word1'] in all_words and row['word2'] in all_words:
                idx1 = all_words.index(row['word1'])
                idx2 = all_words.index(row['word2'])
                
                emb1 = embeddings[idx1].toarray().flatten()
                emb2 = embeddings[idx2].toarray().flatten()
                
                sim = cosine_similarity([emb1], [emb2])[0, 0]
                similarities.append(sim)
                examples.append((row['word1'], row['word2'], sim, row['relation_type']))
        
        if similarities:
            avg_sim = np.mean(similarities)
            examples.sort(key=lambda x: x[2], reverse=True)
            
            results[pair_type] = {
                'avg_similarity': avg_sim,
                'count': len(similarities),
                'best_examples': examples[:3],
                'worst_examples': examples[-3:]
            }
            
            logger.info(f"{pair_type.upper()}: {avg_sim:.3f} avg similarity ({len(similarities)} pairs)")
            if examples:
                best = examples[0]
                logger.info(f"  Best: {best[0]} ↔ {best[1]} = {best[2]:.3f} ({best[3]})")
    
    return results

def save_semantic_model(embeddings, vectorizers, all_words, results):
    """Save the semantically-aware model."""
    os.makedirs('ml_models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_data = {
        'embeddings': embeddings,
        'vectorizers': vectorizers,
        'word_list': all_words,
        'evaluation_results': results,
        'timestamp': timestamp,
        'model_type': 'semantic_first',
        'approach': 'semantic_over_orthographic'
    }
    
    model_path = f'ml_models/semantic_aware_model_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Semantic model saved to: {model_path}")
    return model_path

def main():
    """Main semantic training pipeline."""
    logger.info("🧠 Starting Semantically-Aware Enhanced Trainer...")
    logger.info("🎯 Focus: True semantic relationships over spelling similarity")
    
    try:
        df = load_semantic_training_data()
        if df.empty:
            logger.error("No training data found!")
            return
        
        embeddings, vectorizers, all_words = create_semantic_features(df)
        
        results = evaluate_semantic_quality(df, embeddings, all_words)
        
        model_path = save_semantic_model(embeddings, vectorizers, all_words, results)
        
        logger.info("✅ Semantic training completed!")
        logger.info(f"\n🧠 SEMANTIC-FIRST SUMMARY:")
        logger.info(f"- Training pairs: {len(df):,}")
        logger.info(f"- Vocabulary: {len(all_words):,} words")
        logger.info(f"- Features: {embeddings.shape[1]:,} (70% semantic)")
        logger.info(f"- Approach: Semantic relationships over spelling similarity")
        logger.info(f"- Model: {model_path}")
        
    except Exception as e:
        logger.error(f"Semantic training failed: {e}", exc_info=True)

if __name__ == '__main__':
    main()
