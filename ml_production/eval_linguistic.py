import pandas as pd
import numpy as np
import json
import psycopg2
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_db():
    with open('ml/db_config.json', 'r') as f:
        config = json.load(f)['database']
    return psycopg2.connect(**{k: v for k, v in config.items() if k != 'ssl_mode'})

def main():
    logger.info("🔬 LINGUISTIC EVALUATION - Professional Analysis")
    
    # Load diverse test data
    conn = connect_db()
    query = """
        SELECT w1.lemma as word1, w2.lemma as word2, r.relation_type,
               w1.language_code as lang1, w2.language_code as lang2
        FROM relations r
        JOIN words w1 ON r.from_word_id = w1.id
        JOIN words w2 ON r.to_word_id = w2.id
        WHERE r.relation_type IN ('synonym', 'translation', 'etymology', 'variant')
        AND w1.lemma IS NOT NULL AND w2.lemma IS NOT NULL
        LIMIT 1500;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} test pairs")
    
    # Create embeddings
    all_words = list(set(df['word1'].tolist() + df['word2'].tolist()))
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=1500)
    embeddings = vectorizer.fit_transform(all_words)
    
    # === LINGUISTIC ANALYSIS ===
    logger.info("\n=== SEMANTIC RELATIONSHIP ANALYSIS ===")
    
    results = {}
    for rel_type in ['synonym', 'translation', 'etymology', 'variant']:
        subset = df[df['relation_type'] == rel_type]
        similarities = []
        examples = []
        
        for _, row in subset.iterrows():
            if row['word1'] in all_words and row['word2'] in all_words:
                idx1 = all_words.index(row['word1'])
                idx2 = all_words.index(row['word2'])
                sim = cosine_similarity(embeddings[idx1], embeddings[idx2])[0, 0]
                similarities.append(sim)
                examples.append((row['word1'], row['word2'], sim))
        
        if similarities:
            avg_sim = np.mean(similarities)
            examples.sort(key=lambda x: x[2], reverse=True)
            
            results[rel_type] = avg_sim
            logger.info(f"{rel_type.upper()}: {avg_sim:.3f} (n={len(similarities)})")
            logger.info(f"  Best: {examples[0][0]} ↔ {examples[0][1]} ({examples[0][2]:.3f})")
            logger.info(f"  Worst: {examples[-1][0]} ↔ {examples[-1][1]} ({examples[-1][2]:.3f})")
    
    # === CROSS-LINGUISTIC ANALYSIS ===
    logger.info("\n=== CROSS-LINGUISTIC ANALYSIS ===")
    
    translations = df[df['relation_type'] == 'translation']
    for (lang1, lang2), group in translations.groupby(['lang1', 'lang2']):
        sims = []
        for _, row in group.iterrows():
            if row['word1'] in all_words and row['word2'] in all_words:
                idx1 = all_words.index(row['word1'])
                idx2 = all_words.index(row['word2'])
                sim = cosine_similarity(embeddings[idx1], embeddings[idx2])[0, 0]
                sims.append(sim)
        
        if sims:
            logger.info(f"{lang1} → {lang2}: {np.mean(sims):.3f} (n={len(sims)})")
    
    # === MORPHOLOGICAL ANALYSIS ===
    logger.info("\n=== MORPHOLOGICAL PATTERN ANALYSIS ===")
    
    feature_names = vectorizer.get_feature_names_out()
    filipino_patterns = [p for p in feature_names if any(x in p for x in ['ng', 'an', 'in', 'ka', 'ma'])]
    spanish_patterns = [p for p in feature_names if any(x in p for x in ['cion', 'dad', 'ero'])]
    
    logger.info(f"Filipino morphological patterns detected: {len(filipino_patterns)}")
    logger.info(f"Spanish morphological patterns detected: {len(spanish_patterns)}")
    
    # === LINGUISTIC QUALITY ASSESSMENT ===
    logger.info("\n=== LINGUISTIC QUALITY ASSESSMENT ===")
    
    # Check if synonyms > variants > etymology (expected hierarchy)
    if 'synonym' in results and 'variant' in results:
        if results['synonym'] > results['variant']:
            logger.info("✅ Synonym > Variant relationship: CORRECT")
        else:
            logger.info("❌ Synonym < Variant relationship: INCORRECT")
    
    if 'synonym' in results:
        if results['synonym'] > 0.15:
            logger.info("✅ Synonym detection quality: GOOD")
        elif results['synonym'] > 0.10:
            logger.info("⚠️ Synonym detection quality: MODERATE") 
        else:
            logger.info("❌ Synonym detection quality: POOR")
    
    # Overall assessment
    avg_performance = np.mean(list(results.values())) if results else 0
    logger.info(f"\n🏆 OVERALL PERFORMANCE: {avg_performance:.3f}")
    
    if avg_performance > 0.15:
        logger.info("🎉 Model shows GOOD linguistic understanding")
    elif avg_performance > 0.10:
        logger.info("⚠️ Model shows MODERATE linguistic understanding")
    else:
        logger.info("❌ Model shows POOR linguistic understanding")

if __name__ == '__main__':
    main() 