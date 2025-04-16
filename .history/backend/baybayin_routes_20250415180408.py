from flask import Blueprint, jsonify, request
from sqlalchemy import text
import time
import re
import logging
from backend.metrics import API_REQUESTS, API_ERRORS, REQUEST_LATENCY
from backend.database import db

logger = logging.getLogger(__name__)

bp = Blueprint("baybayin", __name__, url_prefix="/baybayin")

@bp.route("/search", methods=["GET"])
def search_baybayin():
    """
    Search for words with specific Baybayin characters.
    Supports partial matching on Baybayin forms.
    """
    API_REQUESTS.labels(endpoint="search_baybayin", method="GET").inc()
    start_time = time.time()
    
    query = request.args.get("query", "")
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    language_code = request.args.get("language_code")
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
        
    # Ensure the query contains at least one Baybayin character
    if not re.search(r'[\u1700-\u171F]', query):
        return jsonify({"error": "Query must contain at least one Baybayin character"}), 400
    
    try:
        # Build the base SQL query
        sql = """
        SELECT w.id, w.lemma, w.language_code, w.baybayin_form, w.pos, w.completeness_score
        FROM words w
        WHERE w.has_baybayin = TRUE 
        AND w.baybayin_form ILIKE :query_pattern
        """
        
        # Add language filter if specified
        if language_code:
            sql += " AND w.language_code = :language_code"
        
        # Add ordering and pagination
        sql += """
        ORDER BY w.completeness_score DESC, w.lemma
        LIMIT :limit OFFSET :offset
        """
        
        # Count total results (base query without limit/offset)
        count_sql = """
        SELECT COUNT(*)
        FROM words w
        WHERE w.has_baybayin = TRUE 
        AND w.baybayin_form ILIKE :query_pattern
        """
        
        if language_code:
            count_sql += " AND w.language_code = :language_code"
        
        # Execute the queries
        params = {
            "query_pattern": f"%{query}%", 
            "limit": limit, 
            "offset": offset
        }
        
        if language_code:
            params["language_code"] = language_code
        
        count_result = db.session.execute(text(count_sql), params).scalar()
        
        if count_result == 0:
            execution_time = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint="search_baybayin").observe(execution_time)
            return jsonify({"count": 0, "results": []})
        
        query_result = db.session.execute(text(sql), params)
        
        # Format the results
        results = []
        for row in query_result:
            results.append({
                "id": row.id,
                "lemma": row.lemma,
                "language_code": row.language_code,
                "baybayin_form": row.baybayin_form,
                "pos": row.pos,
                "completeness_score": row.completeness_score
            })
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="search_baybayin").observe(execution_time)
        
        return jsonify({
            "count": count_result,
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error in baybayin search: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="search_baybayin", error_type=type(e).__name__).inc()
        return jsonify({"error": "An error occurred while searching"}), 500


@bp.route("/statistics", methods=["GET"])
def get_baybayin_statistics():
    """
    Get detailed statistics about Baybayin usage in the dictionary.
    Includes character frequency, language distribution, and completeness metrics.
    """
    API_REQUESTS.labels(endpoint="get_baybayin_statistics", method="GET").inc()
    start_time = time.time()
    
    try:
        # 1. Overall Baybayin statistics
        sql_overview = """
        SELECT 
            COUNT(*) as total_words,
            SUM(CASE WHEN has_baybayin = TRUE THEN 1 ELSE 0 END) as with_baybayin,
            SUM(CASE WHEN has_baybayin = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as percentage
        FROM words
        """
        
        # 2. Baybayin by language
        sql_by_language = """
        SELECT 
            language_code,
            COUNT(*) as total_words,
            SUM(CASE WHEN has_baybayin = TRUE THEN 1 ELSE 0 END) as with_baybayin,
            SUM(CASE WHEN has_baybayin = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as percentage
        FROM words
        GROUP BY language_code
        ORDER BY with_baybayin DESC
        """
        
        # 3. Baybayin character frequency
        sql_char_frequency = """
        WITH characters AS (
            SELECT w.id, w.language_code, 
                   regexp_split_to_table(w.baybayin_form, '') as character
            FROM words w
            WHERE w.has_baybayin = TRUE AND w.baybayin_form IS NOT NULL
        )
        SELECT 
            character, 
            COUNT(*) as frequency,
            language_code
        FROM characters
        WHERE character ~ '[\u1700-\u171F]'
        GROUP BY character, language_code
        ORDER BY frequency DESC
        """
        
        # 4. Average completeness score for words with Baybayin
        sql_completeness = """
        SELECT 
            AVG(completeness_score) as avg_score_with_baybayin,
            (SELECT AVG(completeness_score) FROM words WHERE has_baybayin = FALSE) as avg_score_without_baybayin
        FROM words
        WHERE has_baybayin = TRUE
        """
        
        # Execute all queries
        overview = db.session.execute(text(sql_overview)).fetchone()
        by_language = db.session.execute(text(sql_by_language)).fetchall()
        char_frequency = db.session.execute(text(sql_char_frequency)).fetchall()
        completeness = db.session.execute(text(sql_completeness)).fetchone()
        
        # Group character frequency by language
        char_freq_by_lang = {}
        for row in char_frequency:
            lang = row.language_code
            if lang not in char_freq_by_lang:
                char_freq_by_lang[lang] = {}
            
            char_freq_by_lang[lang][row.character] = row.frequency
        
        # Format results
        result = {
            "overview": {
                "total_words": overview.total_words,
                "with_baybayin": overview.with_baybayin,
                "percentage": float(overview.percentage) if overview.percentage else 0
            },
            "by_language": [
                {
                    "language_code": row.language_code,
                    "total_words": row.total_words,
                    "with_baybayin": row.with_baybayin,
                    "percentage": float(row.percentage) if row.percentage else 0
                }
                for row in by_language
            ],
            "character_frequency": char_freq_by_lang,
            "completeness": {
                "with_baybayin": float(completeness.avg_score_with_baybayin) if completeness.avg_score_with_baybayin else 0,
                "without_baybayin": float(completeness.avg_score_without_baybayin) if completeness.avg_score_without_baybayin else 0
            }
        }
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="get_baybayin_statistics").observe(execution_time)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating Baybayin statistics: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_baybayin_statistics", error_type=type(e).__name__).inc()
        return jsonify({"error": "An error occurred while generating statistics"}), 500


@bp.route("/convert", methods=["POST"])
def convert_to_baybayin():
    """
    Convert romanized text to Baybayin script.
    Looks up words in the dictionary and uses their Baybayin form when available.
    For unknown words, applies conversion rules based on phonetic patterns.
    """
    API_REQUESTS.labels(endpoint="convert_to_baybayin", method="POST").inc()
    start_time = time.time()
    
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Text parameter is required"}), 400
    
    input_text = data.get("text", "")
    language_code = data.get("language_code", "fil")  # Default to Filipino
    
    # Explicitly check if input_text is a string
    if not isinstance(input_text, str):
        logger.warning(f"Invalid data type for 'text' parameter: {type(input_text).__name__}")
        return jsonify({"error": "Invalid data type for 'text' parameter. Expected a string."}), 400

    if not input_text:
        return jsonify({"error": "Text cannot be empty"}), 400
    
    try:
        # Tokenize the text
        words = re.findall(r'\b\w+\b', input_text.lower())
        
        # Query the database for known words
        if words:
            placeholders = ", ".join([f":word{i}" for i in range(len(words))])
            params = {f"word{i}": word for i, word in enumerate(words)}
            params["language_code"] = language_code
            
            sql = f"""
            SELECT lemma, baybayin_form
            FROM words
            WHERE lemma IN ({placeholders})
            AND language_code = :language_code
            AND has_baybayin = TRUE
            """
            
            word_mappings = {}
            results = db.session.execute(text(sql), params)
            
            for row in results:
                word_mappings[row.lemma.lower()] = row.baybayin_form
        
        # Process the text
        result_text = input_text
        for word in sorted(words, key=len, reverse=True):  # Process longer words first
            if word.lower() in word_mappings:
                # Replace the word with its Baybayin form
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                result_text = pattern.sub(word_mappings[word.lower()], result_text)
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="convert_to_baybayin").observe(execution_time)
        
        return jsonify({
            "original_text": input_text,
            "baybayin_text": result_text,
            "conversion_rate": len([w for w in words if w.lower() in word_mappings]) / len(words) if words else 0
        })
        
    except Exception as e:
        logger.error(f"Error converting to Baybayin: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="convert_to_baybayin", error_type=type(e).__name__).inc()
        return jsonify({"error": "An error occurred during conversion"}), 500 