import psycopg2
import json

# Connect to database
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='fil_dict_db',
        user='postgres',
        password='postgres'
    )
    cur = conn.cursor()
    
    # Query for the word "es"
    cur.execute("""
        SELECT w.id, w.lemma, w.normalized_lemma, w.language_code, w.root_word_id, 
               w.source_info::text, 
               e.etymology_text
        FROM words w
        LEFT JOIN etymologies e ON w.id = e.word_id
        WHERE w.lemma = %s
    """, ('es',))
    
    rows = cur.fetchall()
    
    # Convert to list of dicts for better readability
    columns = ['id', 'lemma', 'normalized_lemma', 'language_code', 
               'root_word_id', 'source_info', 'etymology_text']
    result = []
    
    for row in rows:
        result.append(dict(zip(columns, row)))
    
    print("Word 'es' in database:")
    print(json.dumps(result, indent=2, default=str))
    
    # Check words with [ Ing ] and [ Esp ] etymologies
    print("\nSample of words with [ Ing ] or [ Esp ] etymology format:")
    cur.execute("""
        SELECT w.id, w.lemma, w.root_word_id, e.etymology_text
        FROM words w
        JOIN etymologies e ON w.id = e.word_id
        WHERE e.etymology_text LIKE '[ Ing ]%%' OR e.etymology_text LIKE '[ Esp ]%%'
        LIMIT 15
    """)
    ing_esp_rows = cur.fetchall()
    ing_esp_columns = ['id', 'lemma', 'root_word_id', 'etymology_text']
    ing_esp_result = []
    
    for row in ing_esp_rows:
        ing_esp_result.append(dict(zip(ing_esp_columns, row)))
    
    print(json.dumps(ing_esp_result, indent=2, default=str))
    
    # Check how many words with [ Esp ] have a root_word_id that is NULL
    cur.execute("""
        SELECT COUNT(*)
        FROM words w
        JOIN etymologies e ON w.id = e.word_id
        WHERE e.etymology_text LIKE '[ Esp ]%%' AND w.root_word_id IS NULL
    """)
    count_esp_roots = cur.fetchone()[0]
    
    print(f"\nCount of words with '[ Esp ]' etymology that are root words (root_word_id IS NULL): {count_esp_roots}")
    
    # Check words with foreign etymologies that AREN'T treated as root words
    cur.execute("""
        SELECT w.id, w.lemma, w.root_word_id, e.etymology_text
        FROM words w
        JOIN etymologies e ON w.id = e.word_id
        WHERE (e.etymology_text LIKE '[ Ing ]%%' OR e.etymology_text LIKE '[ Esp ]%%')
            AND w.root_word_id IS NOT NULL
        LIMIT 10
    """)
    non_root_rows = cur.fetchall()
    
    if non_root_rows:
        print("\nWords with foreign etymologies that are NOT treated as root words:")
        non_root_result = []
        for row in non_root_rows:
            non_root_result.append(dict(zip(ing_esp_columns, row)))
        print(json.dumps(non_root_result, indent=2, default=str))
    else:
        print("\nAll words with [ Ing ] and [ Esp ] etymologies are treated as root words")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if conn:
        conn.close() 