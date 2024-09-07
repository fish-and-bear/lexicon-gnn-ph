from flask import Blueprint, jsonify, request, current_app
from sqlalchemy.orm import joinedload, selectinload, load_only
from sqlalchemy import or_, func
from models import Word, Definition, Meaning, Etymology, EtymologyComponent
from database import db_session
import logging
from datetime import datetime
from unidecode import unidecode
from functools import lru_cache
import re
from caching import multi_level_cache
from urllib.parse import unquote
from fuzzywuzzy import fuzz, process

bp = Blueprint("api", __name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1000)
def normalize_word(word):
    return unidecode(str(word).lower()) if word else None

@lru_cache(maxsize=1000)
def parse_etymology(etymology):
    if not etymology:
        return []
    cleaned_etymology = etymology.strip("[]")
    parts = re.split(r"\s*\+\s*|\s*-\s*", cleaned_etymology)
    return [part.strip() for part in parts if part.strip()]

def filter_valid_meanings(meanings):
    return [
        {
            "definition": m.meaning,
            "source": m.source.source_name if m.source else None,
        }
        for m in meanings
        if m.meaning and m.meaning.strip() and m.meaning.strip() != "0"
    ]

def filter_valid_definitions(definitions):
    return [
        {
            "partOfSpeech": d.part_of_speech,
            "meanings": filter_valid_meanings(d.meanings),
            "usageNotes": d.usage_notes or [],
            "examples": [e.example for e in d.examples],
            "tags": d.tags or [],
        }
        for d in definitions
        if filter_valid_meanings(d.meanings)
    ]

def get_word_details(word_entry):
    etymologies = [
        {
            "text": etym.etymology_text,
            "components": [
                {"component": comp.component, "order": comp.order}
                for comp in etym.components
            ],
        }
        for etym in word_entry.etymologies
    ]

    parsed_etymology = set()
    for etym in etymologies:
        parsed_etymology.update(parse_etymology(etym["text"]))
        parsed_etymology.update(comp["component"] for comp in etym["components"])

    return {
        "meta": {
            "version": "1.0",
            "word": word_entry.word,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
        "data": {
            "word": word_entry.word,
            "pronunciation": {
                "text": word_entry.pronunciation,
                "ipa": ", ".join(
                    [
                        p
                        for p in word_entry.audio_pronunciation or []
                        if p.startswith("/")
                    ]
                ),
                "audio": [
                    p
                    for p in word_entry.audio_pronunciation or []
                    if p and not p.startswith("/")
                ],
            },
            "etymology": {
                "kaikki": word_entry.kaikki_etymology,
                "components": [
                    {"component": comp.component, "order": comp.order}
                    for etym in word_entry.etymologies
                    for comp in etym.components
                ],
                "text": [etym.etymology_text for etym in word_entry.etymologies],
                "parsed": list(parsed_etymology),
            },
            "definitions": filter_valid_definitions(word_entry.definitions),
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
                "holonyms": [h.holonym for h in word_entry.holonyms],
            },
            "forms": [{"form": f.form, "tags": f.tags} for f in word_entry.forms],
            "languages": [lang.code for lang in word_entry.languages],
            "tags": word_entry.tags or [],
            "headTemplates": [
                {"name": ht.template_name, "args": ht.args, "expansion": ht.expansion}
                for ht in word_entry.head_templates
            ],
            "inflections": [
                {"name": infl.name, "args": infl.args}
                for infl in word_entry.inflections
            ],
            "alternateForms": [af.alternate_form for af in word_entry.alternate_forms],
        },
    }

@multi_level_cache
def get_word_network_data(word_entry):
    if not word_entry:
        logger.warning(f"No word entry found for {word_entry}")
        return None

    first_definition = next((m.meaning for d in word_entry.definitions for m in d.meanings 
                             if m.meaning and m.meaning.strip() and m.meaning.strip() != "0"), 
                            "No definition available")

    network_data = {
        "word": word_entry.word,
        "definition": first_definition,
        "derivatives": [d.derivative for d in word_entry.derivatives],
        "root_word": word_entry.root_word,
        "synonyms": [w.word for w in word_entry.synonyms],
        "antonyms": [w.word for w in word_entry.antonyms],
        "associated_words": [aw.associated_word for aw in word_entry.associated_words],
        "etymology": parse_etymology(
            word_entry.etymologies[0].etymology_text if word_entry.etymologies else ""
        ),
    }
    
    # Add reverse relationships
    network_data.update({
        "derived_from": [w.word for w in Word.query.with_entities(Word.word).filter(Word.derivatives.any(derivative=word_entry.word)).all()],
    })
    
    return network_data

@multi_level_cache
def get_related_words(word, depth=2, breadth=10):
    visited = set()
    queue = [(word, 0)]
    network = {}
    max_network_size = current_app.config.get("MAX_NETWORK_SIZE", 100)

    while queue and len(network) < max_network_size:
        current_word, current_depth = queue.pop(0)

        if current_word in visited or current_depth > depth:
            continue

        visited.add(current_word)
        normalized_word = normalize_word(current_word)

        word_entry = Word.query.options(
            load_only('word', 'id')
        ).filter(func.lower(func.unaccent(Word.word)) == normalized_word).first()

        if word_entry:
            network_data = get_word_network_data(word_entry)
            if network_data:
                network[current_word] = network_data

                if current_depth < depth:
                    related_words = set()
                    for relation in [
                        "derivatives", "synonyms", "antonyms", "associated_words",
                        "etymology", "derived_from"
                    ]:
                        related_words.update(network[current_word].get(relation, []))
                    if network[current_word].get("root_word"):
                        related_words.add(network[current_word]["root_word"])

                    new_words = list(related_words - visited)[:breadth]
                    queue.extend((w, current_depth + 1) for w in new_words if w)

    # Ensure all nodes are connected to the main word
    connected_nodes = set([word])
    to_remove = set()
    for node, data in network.items():
        is_connected = False
        for relation in data.values():
            if isinstance(relation, list):
                if any(r in connected_nodes for r in relation):
                    is_connected = True
                    connected_nodes.add(node)
                    break
            elif isinstance(relation, str) and relation in connected_nodes:
                is_connected = True
                connected_nodes.add(node)
                break
        if not is_connected:
            to_remove.add(node)

    for node in to_remove:
        del network[node]

    logger.info(f"Network for {word}: {network}")
    return network

@bp.route("/api/v1/words", methods=["GET"])
def get_words():
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(int(request.args.get("per_page", 20)), 100)
    search = request.args.get("search", "")
    fuzzy = request.args.get("fuzzy", "false").lower() == "true"

    query = Word.query.options(load_only('word', 'id'))

    if search:
        normalized_search = normalize_word(search)
        if fuzzy:
            all_words = [w.word for w in query.all()]
            fuzzy_matches = process.extract(normalized_search, all_words, limit=per_page * 2, scorer=fuzz.ratio)
            matched_words = [match[0] for match in fuzzy_matches if match[1] >= 80]
            query = query.filter(Word.word.in_(matched_words))
        else:
            query = query.filter(func.lower(func.unaccent(Word.word)).like(f"{normalized_search}%"))

    total = query.count()
    words = query.order_by(Word.word).offset((page - 1) * per_page).limit(per_page).all()

    return jsonify({
        "words": [{"word": w.word, "id": w.id} for w in words],
        "page": page,
        "per_page": per_page,
        "total": total,
    })

@bp.route("/api/v1/words/<word>", methods=["GET"])
@multi_level_cache
def get_word(word):
    try:
        word_entry = get_word_with_relations(word)
        if word_entry is None:
            logger.info(f"Word not found: {word}")
            return jsonify({"error": "Word not found"}), 404

        return jsonify(get_word_details(word_entry))
    except Exception as e:
        logger.error(f"Error in get_word: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

def get_word_with_relations(word):
    normalized_word = normalize_word(word)
    
    word_entry = Word.query.options(
        selectinload(Word.definitions)
        .selectinload(Definition.meanings)
        .selectinload(Meaning.source),
        selectinload(Word.etymologies)
        .selectinload(Etymology.components)
    ).filter(
        func.lower(func.unaccent(Word.word)) == normalized_word
    ).first()
    
    if not word_entry:
        logger.warning(f"Word not found: {normalized_word}")

    return word_entry

@bp.route("/api/v1/check_word/<word>", methods=["GET"])
@multi_level_cache
def check_word(word):
    try:
        normalized_word = normalize_word(word)
        word_entry = Word.query.filter(
            func.lower(func.unaccent(Word.word)) == normalized_word
        ).first()
        return jsonify(
            {
                "exists": bool(word_entry),
                "word": word_entry.word if word_entry else None,
            }
        )
    except Exception as e:
        logger.error(f"Error in check_word: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@bp.route("/api/v1/word_network/<path:word>", methods=["GET"])
@multi_level_cache
def get_word_network(word):
    try:
        depth = min(int(request.args.get("depth", 2)), 5)
        breadth = min(int(request.args.get("breadth", 10)), 20)

        if not word:
            return jsonify({"error": "Word not provided"}), 400

        decoded_word = unquote(word).strip()
        normalized_word = normalize_word(decoded_word)
        logger.info(f"Fetching word network for: {normalized_word}, depth: {depth}, breadth: {breadth}")
        network = get_related_words(normalized_word, depth, breadth)

        if not network:
            logger.warning(f"Word network not found for: {normalized_word}")
            return jsonify({"error": "Word not found"}), 404

        logger.info(f"Successfully fetched word network for: {normalized_word}")
        return jsonify(network)
    except Exception as e:
        logger.error(f"Error in get_word_network: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@bp.route("/api/v1/etymology/<word>", methods=["GET"])
@multi_level_cache
def get_etymology(word):
    try:
        normalized_word = normalize_word(word)
        word_entry = (
            Word.query.options(
                joinedload(Word.etymologies).joinedload(Etymology.components)
            )
            .filter(func.lower(func.unaccent(Word.word)) == normalized_word)
            .first_or_404()
        )

        etymologies = [
            {
                "etymology_text": etym.etymology_text,
                "components": [
                    {"component": comp.component, "order": comp.order}
                    for comp in etym.components
                ],
            }
            for etym in word_entry.etymologies
        ]

        parsed_etymology = set()
        for etym in etymologies:
            parsed_etymology.update(parse_etymology(etym["etymology_text"]))
            parsed_etymology.update(comp["component"] for comp in etym["components"])

        return jsonify(
            {
                "word": word_entry.word,
                "etymologies": etymologies,
                "kaikki_etymology": word_entry.kaikki_etymology,
                "parsed_etymology": list(parsed_etymology),
            }
        )
    except Exception as e:
        logger.error(f"Error in get_etymology: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500


@bp.route("/api/v1/bulk_words", methods=["POST"])
def bulk_get_words():
    try:
        words = request.json.get("words", [])
        if not words or not isinstance(words, list):
            return jsonify({"error": "Invalid input"}), 400

        normalized_words = [normalize_word(w) for w in words]
        word_entries = Word.query.filter(
            func.lower(func.unaccent(Word.word)).in_(normalized_words)
        ).options(
            load_only('word', 'id')
        ).all()

        word_details = [get_word_details(w) for w in word_entries]

        return jsonify({"words": word_details})
    except Exception as e:
        logger.error(f"Error in bulk_get_words: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500


@bp.route("/favicon.ico")
def favicon():
    return "", 204


@bp.teardown_request
def remove_session(exception=None):
    db_session.remove()
