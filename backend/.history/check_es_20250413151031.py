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
    
    # Also check other words where es appears as etymology term
    cur.execute("""
        SELECT w.id, w.lemma, w.normalized_lemma, w.language_code, 
               e.etymology_text
        FROM words w
        JOIN etymologies e ON w.id = e.word_id
        WHERE e.etymology_text ILIKE '%%es%%'
        LIMIT 5
    """)
    
    rows = cur.fetchall()
    columns = ['id', 'lemma', 'normalized_lemma', 'language_code', 'etymology_text']
    result = []
    
    for row in rows:
        result.append(dict(zip(columns, row)))
    
    print("\nWords with 'es' in etymology:")
    print(json.dumps(result, indent=2, default=str))
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if conn:
        conn.close() 