#!/usr/bin/env python3
"""
Qualitative Linguistic Evaluation Script for FilRelex

This script loads a trained production model and performs a qualitative
analysis by "interviewing" the model. For a given set of probe words,
it finds the most similar words in the embedding space and compares them
to the ground truth relationships from the database.
"""

import torch
import pandas as pd
import numpy as np
import json
import psycopg2
import logging
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
MODEL_PATH = 'ml/final_production_model_20250702_112329.pt'
DB_CONFIG_PATH = 'ml/db_config.json'
PROBE_WORDS = [
    {'lemma': 'bahay', 'lang': 'tl'},
    {'lemma': 'puso', 'lang': 'tl'},
    {'lemma': 'araw', 'lang': 'tl'},
    {'lemma': 'love', 'lang': 'en'},
    {'lemma': 'house', 'lang': 'en'},
    {'lemma': 'sol', 'lang': 'es'}
]
TOP_K = 10  # Number of similar words to retrieve

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# --- Database Connection ---
def connect_to_database():
    """Establishes a connection to the PostgreSQL database."""
    with open(DB_CONFIG_PATH, 'r') as f:
        config = json.load(f)['database']
    # Adapt to different key names for SSL mode
    db_config = {k: v for k, v in config.items() if k != 'ssl_mode'}
    if 'ssl_mode' in config and 'sslmode' not in db_config:
        db_config['sslmode'] = config['ssl_mode']
    return psycopg2.connect(**db_config)

# --- Main Evaluation Logic ---
def get_ground_truth_relations(conn, word_id):
    """Fetches all known relations for a given word ID from the database."""
    query = """
        SELECT w.lemma, w.language_code, r.relation_type
        FROM relations r
        JOIN words w ON r.to_word_id = w.id
        WHERE r.from_word_id = %s
    """
    return pd.read_sql(query, conn, params=(word_id,))

def run_linguistic_evaluation():
    """Main function to run the evaluation process."""
    logger.info(f"🔬 Starting Linguistic Evaluation with model: {MODEL_PATH}")
    
    # 1. Load Model and Data
    try:
        data = torch.load(MODEL_PATH)
        embeddings = data['embeddings']
        word_to_idx = data['word_to_idx']
        idx_to_word = {i: w for w, i in word_to_idx.items()}
    except FileNotFoundError:
        logger.critical(f"Model file not found at {MODEL_PATH}. Please ensure the path is correct.")
        return
    except KeyError:
        logger.critical(f"Model file {MODEL_PATH} is missing required data (embeddings, word_to_idx).")
        return

    # 2. Load word metadata from DB for easier lookup
    conn = connect_to_database()
    try:
        word_ids_str = ','.join(map(str, word_to_idx.keys()))
        if not word_ids_str:
            logger.critical("Model vocabulary is empty. Aborting.")
            return
            
        words_df = pd.read_sql(f"SELECT id, lemma, language_code FROM words WHERE id IN ({word_ids_str})", conn)
        word_meta = {row['id']: (row['lemma'], row['language_code']) for _, row in words_df.iterrows()}
    finally:
        conn.close()

    # 3. Perform Qualitative Probing
    for probe in PROBE_WORDS:
        probe_lemma, probe_lang = probe['lemma'], probe['lang']
        
        # Find the word ID for the probe word
        probe_word_results = words_df[(words_df['lemma'] == probe_lemma) & (words_df['language_code'] == probe_lang)]
        if probe_word_results.empty:
            logger.warning(f"Probe word '{probe_lemma}' ({probe_lang}) not found in the model's vocabulary. Skipping.")
            continue
        probe_word_id = probe_word_results['id'].values[0]
        
        if probe_word_id not in word_to_idx:
            logger.warning(f"Probe word '{probe_lemma}' ({probe_lang}) ID {probe_word_id} not in word_to_idx map. This may indicate a data mismatch. Skipping.")
            continue
            
        probe_idx = word_to_idx[probe_word_id]
        probe_embedding = embeddings[probe_idx].reshape(1, -1)
        
        # Calculate similarity with all other words
        similarities = cosine_similarity(probe_embedding, embeddings)[0]
        
        # Get top K most similar words (excluding the word itself)
        top_indices = np.argsort(similarities)[-TOP_K-1:-1][::-1]
        
        print("\n" + "="*80)
        logger.info(f"LINGUISTIC PROBE: '{probe_lemma}' ({probe_lang})")
        print("="*80)
        
        # Display model's top predictions
        print(f"\n🧠 Model's Top {TOP_K} Similar Words:")
        print("-------------------------------------")
        for i in top_indices:
            sim_word_id = idx_to_word[i]
            sim_word_lemma, sim_word_lang = word_meta.get(sim_word_id, ("<UNKNOWN>", "<UNKNOWN>"))
            print(f"  - {sim_word_lemma:<20} ({sim_word_lang})  (Similarity: {similarities[i]:.3f})")

        # Get and display ground truth from the database
        conn = connect_to_database()
        try:
            ground_truth_df = get_ground_truth_relations(conn, int(probe_word_id))
        finally:
            conn.close()

        print(f"\n📚 Ground Truth Relations from Database:")
        print("---------------------------------------")
        if not ground_truth_df.empty:
            for _, row in ground_truth_df.iterrows():
                print(f"  - {row['relation_type'].upper():<12} -> {row['lemma']} ({row['language_code']})")
        else:
            print("  - No relations found in the database for this word.")

        print("\n" + "="*80)

if __name__ == '__main__':
    run_linguistic_evaluation() 