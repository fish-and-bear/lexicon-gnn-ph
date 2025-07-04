#!/usr/bin/env python3
"""
Linguistic Evaluation of FilRelex ML Model
A comprehensive evaluation from a linguist's perspective
"""

import pandas as pd
import numpy as np
import json
import psycopg2
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_db():
    with open('ml/db_config.json', 'r') as f:
        config = json.load(f)['database']
    return psycopg2.connect(**{k: v for k, v in config.items() if k != 'ssl_mode'})

def load_test_data():
    """Load comprehensive test data for linguistic evaluation."""
    conn = connect_db()
    
    # Get diverse relationship types
    query = """
        SELECT w1.lemma as word1, w2.lemma as word2, r.relation_type,
               w1.language_code as lang1, w2.language_code as lang2
        FROM relations r
        JOIN words w1 ON r.from_word_id = w1.id
        JOIN words w2 ON r.to_word_id = w2.id
        WHERE r.relation_type IN ('synonym', 'translation', 'etymology', 'variant', 'related')
        AND w1.lemma IS NOT NULL AND w2.lemma IS NOT NULL
        AND LENGTH(w1.lemma) > 1 AND LENGTH(w2.lemma) > 1
        ORDER BY r.relation_type, RANDOM()
        LIMIT 2000;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} test pairs across {df['relation_type'].nunique()} relation types")
    return df

def create_embeddings(df):
    """Create embeddings for all words in the test set."""
    all_words = list(set(df['word1'].tolist() + df['word2'].tolist()))
    
    # Enhanced TF-IDF with better parameters for linguistic analysis
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),  # Capture longer morphological patterns
        max_features=2000,   # More features for better discrimination
        lowercase=True,
        sublinear_tf=True    # Better handling of frequent patterns
    )
    
    embeddings = vectorizer.fit_transform(all_words)
    
    # Create word-to-embedding mapping
    word_embeddings = {}
    for i, word in enumerate(all_words):
        word_embeddings[word] = embeddings[i].toarray().flatten()
    
    return word_embeddings, vectorizer, all_words

def analyze_morphological_patterns(df, word_embeddings):
    """Analyze how well the model captures morphological relationships."""
    logger.info("=== MORPHOLOGICAL ANALYSIS ===")
    
    # Find potential morphological pairs (words with shared roots)
    morphological_pairs = []
    
    for _, row in df.iterrows():
        word1, word2 = row['word1'], row['word2']
        
        # Simple morphological heuristics for Filipino
        if row['lang1'] == 'tl' and row['lang2'] == 'tl':
            # Check for common Filipino prefixes/suffixes
            prefixes = ['mag', 'nag', 'pag', 'ka', 'ma', 'um']
            suffixes = ['an', 'in', 'han']
            
            # Find shared roots
            for prefix in prefixes:
                if word1.startswith(prefix) and len(word1) > len(prefix) + 2:
                    root1 = word1[len(prefix):]
                    if root1 in word2 or word2 in root1:
                        morphological_pairs.append((word1, word2, f"prefix_{prefix}"))
    
    # Evaluate morphological similarities
    morph_similarities = []
    for word1, word2, pattern in morphological_pairs[:50]:  # Limit for analysis
        if word1 in word_embeddings and word2 in word_embeddings:
            sim = cosine_similarity(
                word_embeddings[word1].reshape(1, -1),
                word_embeddings[word2].reshape(1, -1)
            )[0, 0]
            morph_similarities.append(sim)
    
    if morph_similarities:
        avg_morph_sim = np.mean(morph_similarities)
        logger.info(f"Morphological pairs found: {len(morphological_pairs)}")
        logger.info(f"Average morphological similarity: {avg_morph_sim:.3f}")
        return avg_morph_sim
    else:
        logger.info("No morphological patterns detected")
        return 0.0

def analyze_cross_linguistic_relationships(df, word_embeddings):
    """Analyze translation and borrowing relationships."""
    logger.info("=== CROSS-LINGUISTIC ANALYSIS ===")
    
    translation_results = {}
    
    # Analyze by language pair
    for (lang1, lang2), group in df[df['relation_type'] == 'translation'].groupby(['lang1', 'lang2']):
        similarities = []
        
        for _, row in group.iterrows():
            if row['word1'] in word_embeddings and row['word2'] in word_embeddings:
                sim = cosine_similarity(
                    word_embeddings[row['word1']].reshape(1, -1),
                    word_embeddings[row['word2']].reshape(1, -1)
                )[0, 0]
                similarities.append(sim)
        
        if similarities:
            avg_sim = np.mean(similarities)
            translation_results[f"{lang1}-{lang2}"] = {
                'avg_similarity': avg_sim,
                'count': len(similarities)
            }
            logger.info(f"{lang1} → {lang2}: {avg_sim:.3f} (n={len(similarities)})")
    
    return translation_results

def analyze_semantic_relationships(df, word_embeddings):
    """Analyze different types of semantic relationships."""
    logger.info("=== SEMANTIC RELATIONSHIP ANALYSIS ===")
    
    relationship_analysis = {}
    
    for rel_type in ['synonym', 'etymology', 'variant', 'related']:
        subset = df[df['relation_type'] == rel_type]
        if subset.empty:
            continue
            
        similarities = []
        examples = []
        
        for _, row in subset.iterrows():
            if row['word1'] in word_embeddings and row['word2'] in word_embeddings:
                sim = cosine_similarity(
                    word_embeddings[row['word1']].reshape(1, -1),
                    word_embeddings[row['word2']].reshape(1, -1)
                )[0, 0]
                similarities.append(sim)
                examples.append((row['word1'], row['word2'], sim))
        
        if similarities:
            avg_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            
            # Find best and worst examples
            examples.sort(key=lambda x: x[2], reverse=True)
            best_examples = examples[:3]
            worst_examples = examples[-3:]
            
            relationship_analysis[rel_type] = {
                'avg_similarity': avg_sim,
                'std_similarity': std_sim,
                'count': len(similarities),
                'best_examples': best_examples,
                'worst_examples': worst_examples
            }
            
            logger.info(f"{rel_type.upper()}: {avg_sim:.3f} ± {std_sim:.3f} (n={len(similarities)})")
            logger.info(f"  Best: {best_examples[0][0]} ↔ {best_examples[0][1]} ({best_examples[0][2]:.3f})")
            logger.info(f"  Worst: {worst_examples[0][0]} ↔ {worst_examples[0][1]} ({worst_examples[0][2]:.3f})")
    
    return relationship_analysis

def analyze_character_patterns(word_embeddings, vectorizer):
    """Analyze what character patterns the model has learned."""
    logger.info("=== CHARACTER PATTERN ANALYSIS ===")
    
    # Get feature names (character n-grams)
    feature_names = vectorizer.get_feature_names_out()
    
    # Find most common patterns
    logger.info("Most common character patterns learned:")
    for i, pattern in enumerate(feature_names[:20]):
        logger.info(f"  {pattern}")
    
    # Look for linguistic patterns
    filipino_patterns = [p for p in feature_names if any(x in p for x in ['ng', 'an', 'in', 'ka', 'ma', 'um'])]
    spanish_patterns = [p for p in feature_names if any(x in p for x in ['cion', 'dad', 'ero', 'ado'])]
    
    logger.info(f"Filipino-like patterns detected: {len(filipino_patterns)}")
    logger.info(f"Spanish-like patterns detected: {len(spanish_patterns)}")
    
    if filipino_patterns:
        logger.info(f"Example Filipino patterns: {filipino_patterns[:5]}")
    if spanish_patterns:
        logger.info(f"Example Spanish patterns: {spanish_patterns[:5]}")

def linguistic_quality_assessment(relationship_analysis):
    """Provide overall linguistic quality assessment."""
    logger.info("=== LINGUISTIC QUALITY ASSESSMENT ===")
    
    # Expected similarity rankings (higher = more similar)
    expected_ranking = ['synonym', 'variant', 'etymology', 'related']
    
    # Get actual similarities
    actual_similarities = {}
    for rel_type in expected_ranking:
        if rel_type in relationship_analysis:
            actual_similarities[rel_type] = relationship_analysis[rel_type]['avg_similarity']
    
    # Check if ranking matches expectations
    ranking_correct = True
    for i in range(len(expected_ranking) - 1):
        current = expected_ranking[i]
        next_rel = expected_ranking[i + 1]
        
        if current in actual_similarities and next_rel in actual_similarities:
            if actual_similarities[current] < actual_similarities[next_rel]:
                ranking_correct = False
                logger.warning(f"Ranking issue: {current} ({actual_similarities[current]:.3f}) < {next_rel} ({actual_similarities[next_rel]:.3f})")
    
    if ranking_correct:
        logger.info("✅ Relationship similarity ranking matches linguistic expectations")
    else:
        logger.warning("⚠️ Relationship similarity ranking has issues")
    
    # Overall assessment
    synonym_quality = actual_similarities.get('synonym', 0)
    if synonym_quality > 0.15:
        logger.info("✅ Synonym detection: GOOD")
    elif synonym_quality > 0.10:
        logger.info("⚠️ Synonym detection: MODERATE")
    else:
        logger.info("❌ Synonym detection: POOR")

def main():
    """Run comprehensive linguistic evaluation."""
    logger.info("🔬 Starting Comprehensive Linguistic Evaluation...")
    
    # Load test data
    df = load_test_data()
    
    # Create embeddings
    word_embeddings, vectorizer, all_words = create_embeddings(df)
    
    # Run linguistic analyses
    morph_score = analyze_morphological_patterns(df, word_embeddings)
    translation_results = analyze_cross_linguistic_relationships(df, word_embeddings)
    relationship_analysis = analyze_semantic_relationships(df, word_embeddings)
    analyze_character_patterns(word_embeddings, vectorizer)
    linguistic_quality_assessment(relationship_analysis)
    
    logger.info("🏁 Linguistic evaluation complete!")

if __name__ == '__main__':
    main() 