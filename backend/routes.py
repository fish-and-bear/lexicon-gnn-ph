from flask import Blueprint, jsonify, request, current_app, abort
from sqlalchemy.orm import joinedload, subqueryload
from sqlalchemy import or_
from models import (
    Word, Definition, Meaning, Source, Language, Etymology,
    EtymologyComponent, Form, HeadTemplate, Derivative,
    Example, Hypernym, Hyponym, Meronym, Holonym, AssociatedWord,
    AlternateForm, Inflection
)
from database import db_session
from functools import wraps
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('api', __name__)

# Rate limiting decorator
def rate_limit(limit=100, per=60):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            now = time.time()
            key = f"{request.remote_addr}:{f.__name__}"
            calls = getattr(current_app, 'rate_limit_calls', {})
            if key not in calls:
                calls[key] = [(now, 1)]
            else:
                calls[key] = [(t, c) for t, c in calls[key] if now - t < per]
                if len(calls[key]) >= limit:
                    return jsonify({"error": "Rate limit exceeded"}), 429
                calls[key].append((now, len(calls[key]) + 1))
            setattr(current_app, 'rate_limit_calls', calls)
            return f(*args, **kwargs)
        return wrapped
    return decorator

# Error handling
@bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


def get_word_details(word_entry):
    return {
        "meta": {
            "version": "1.0",
            "word": word_entry.word,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "data": {
            "word": word_entry.word,
            "pronunciation": {
                "text": word_entry.pronunciation,
                "ipa": ", ".join([p for p in word_entry.audio_pronunciation if p.startswith('/')]),
                "audio": [p for p in word_entry.audio_pronunciation if p and not p.startswith('/')]
            },
            "etymology": {
                "kaikki": word_entry.kaikki_etymology,
                "components": [
                    {"component": comp.component, "order": comp.order}
                    for etym in word_entry.etymologies
                    for comp in etym.components
                ],
                "text": [etym.etymology_text for etym in word_entry.etymologies]
            },
            "definitions": [
                {
                    "partOfSpeech": d.part_of_speech,
                    "meanings": [
                        {"definition": m.meaning, "source": m.source.source_name}
                        for m in d.meanings
                    ],
                    "usageNotes": d.usage_notes or [],
                    "examples": [e.example for e in d.examples],
                    "tags": d.tags or []
                }
                for d in word_entry.definitions
            ],
            "relationships": {
                "rootWord": word_entry.root_word,
                "derivatives": [d.derivative for d in word_entry.derivatives],
                "synonyms": [w.word for w in word_entry.synonyms],
                "antonyms": [w.word for w in word_entry.antonyms],
                "associatedWords": [aw.associated_word for aw in word_entry.associated_words],
                "relatedTerms": [w.word for w in word_entry.related_terms],
                "hypernyms": [h.hypernym for h in word_entry.hypernyms],
                "hyponyms": [h.hyponym for h in word_entry.hyponyms],
                "meronyms": [m.meronym for m in word_entry.meronyms],
                "holonyms": [h.holonym for h in word_entry.holonyms]
            },
            "forms": [
                {"form": f.form, "tags": f.tags}
                for f in word_entry.forms
            ],
            "languages": [lang.code for lang in word_entry.languages],
            "tags": word_entry.tags or [],
            "headTemplates": [
                {
                    "name": ht.template_name,
                    "args": ht.args,
                    "expansion": ht.expansion
                }
                for ht in word_entry.head_templates
            ],
            "inflections": [
                {
                    "name": infl.name,
                    "args": infl.args
                }
                for infl in word_entry.inflections
            ],
            "alternateForms": [af.alternate_form for af in word_entry.alternate_forms]
        }
    }

def get_word_network_data(word_entry):
    return {
        "word": word_entry.word,
        "pronunciation": word_entry.pronunciation,
        "languages": [lang.code for lang in word_entry.languages],
        "definitions": [{"part_of_speech": d.part_of_speech, "meaning": m.meaning} 
                        for d in word_entry.definitions 
                        for m in d.meanings[:1]],  # Only include the first meaning for brevity
        "related_words": [
            *[aw.associated_word for aw in word_entry.associated_words],
            *[w.word for w in word_entry.synonyms],
            *[w.word for w in word_entry.antonyms],
            *[d.derivative for d in word_entry.derivatives]
        ]
    }

def get_related_words(word, depth=2, breadth=10):
    visited = set()
    queue = [(word, 0)]
    network = {}

    while queue and len(network) < 100:  # Limit total network size
        current_word, current_depth = queue.pop(0)
        
        if current_word in visited or current_depth > depth:
            continue
        
        visited.add(current_word)
        
        word_entry = Word.query.options(
            joinedload(Word.pronunciation),
            joinedload(Word.languages),
            joinedload(Word.definitions).joinedload(Definition.meanings),
            joinedload(Word.associated_words),
            joinedload(Word.synonyms),
            joinedload(Word.antonyms),
            joinedload(Word.derivatives)
        ).filter_by(word=current_word).first()
        
        if word_entry:
            network[current_word] = get_word_network_data(word_entry)
            
            if current_depth < depth:
                related_words = (
                    [aw.associated_word for aw in word_entry.associated_words] +
                    [w.word for w in word_entry.synonyms] +
                    [w.word for w in word_entry.antonyms] +
                    [d.derivative for d in word_entry.derivatives]
                )[:breadth]
                queue.extend((w, current_depth + 1) for w in related_words 
                             if w not in visited)

    return network

@bp.route('/api/v1/words', methods=['GET'])
@rate_limit()
def get_words():
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 20)), 100)
    search = request.args.get('search', '')
    
    query = Word.query
    
    if search:
        query = query.filter(or_(
            Word.word.ilike(f"%{search}%"),
            Word.definitions.any(Definition.meanings.any(Meaning.meaning.ilike(f"%{search}%")))
        ))
    
    total = query.count()
    words = query.order_by(Word.word).offset((page - 1) * per_page).limit(per_page).all()
    
    return jsonify({
        "words": [{"word": w.word, "id": w.id} for w in words],
        "page": page,
        "per_page": per_page,
        "total": total
    })


@bp.route('/api/v1/words/<word>', methods=['GET'])
@rate_limit()
def get_word(word):
    try:
        # First, fetch the main word entry
        word_entry = Word.query.filter_by(word=word.lower()).first()
        
        if word_entry is None:
            logger.info(f"Word not found: {word}")
            return jsonify({"error": "Word not found"}), 404

        # Then, fetch related data in separate queries
        word_entry = Word.query.options(
            joinedload(Word.languages),
            subqueryload(Word.definitions).joinedload(Definition.meanings).joinedload(Meaning.source),
            subqueryload(Word.definitions).joinedload(Definition.examples),
            subqueryload(Word.forms),
            subqueryload(Word.head_templates),
            subqueryload(Word.derivatives),
            subqueryload(Word.examples),
            subqueryload(Word.hypernyms),
            subqueryload(Word.hyponyms),
            subqueryload(Word.meronyms),
            subqueryload(Word.holonyms),
            subqueryload(Word.associated_words),
            subqueryload(Word.synonyms),
            subqueryload(Word.antonyms),
            subqueryload(Word.related_terms),
            subqueryload(Word.etymologies).joinedload(Etymology.components),
            subqueryload(Word.alternate_forms),
            subqueryload(Word.inflections)
        ).filter_by(id=word_entry.id).first()

        return jsonify(get_word_details(word_entry))
    except Exception as e:
        logger.error(f"Error in get_word: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@bp.route('/api/v1/check_word/<word>', methods=['GET'])
def check_word(word):
    try:
        word_entry = Word.query.filter_by(word=word.lower()).first()
        if word_entry:
            return jsonify({"exists": True, "word": word_entry.word}), 200
        else:
            return jsonify({"exists": False}), 404
    except Exception as e:
        logging.error(f"Error in check_word: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500
    

@bp.route('/api/v1/word_network/<word>', methods=['GET'])
@rate_limit()
def get_word_network(word):
    depth = min(int(request.args.get('depth', 2)), 5)
    breadth = min(int(request.args.get('breadth', 10)), 20)
    
    network = get_related_words(word.lower(), depth, breadth)
    
    if not network:
        return jsonify({"error": "Word not found"}), 404

    return jsonify(network)

@bp.route('/api/v1/etymology/<word>', methods=['GET'])
@rate_limit()
def get_etymology(word):
    word_entry = Word.query.options(
        joinedload(Word.etymologies).joinedload(Etymology.components)
    ).filter_by(word=word.lower()).first_or_404()

    etymologies = [{
        "etymology_text": etym.etymology_text,
        "components": [
            {"component": comp.component, "order": comp.order}
            for comp in etym.components
        ]
    } for etym in word_entry.etymologies]

    return jsonify({
        "word": word_entry.word,
        "etymologies": etymologies,
        "kaikki_etymology": word_entry.kaikki_etymology
    })

@bp.route('/api/v1/bulk_words', methods=['POST'])
@rate_limit(limit=10, per=60)  # Stricter rate limit for bulk operations
def bulk_get_words():
    words = request.json.get('words', [])
    if not words or not isinstance(words, list):
        return jsonify({"error": "Invalid input"}), 400
    
    word_entries = Word.query.filter(Word.word.in_(words)).all()
    return jsonify({
        "words": [get_word_details(w) for w in word_entries]
    })

@bp.route('/favicon.ico')
def favicon():
    return '', 204

@bp.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@bp.errorhandler(500)
def internal_error(error):
    logger.error('Server Error: %s', (error), exc_info=True)
    return jsonify({"error": "An unexpected error occurred"}), 500

@bp.teardown_request
def remove_session(exception=None):
    db_session.remove()