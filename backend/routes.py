from flask import Blueprint, jsonify, request, current_app, abort
from sqlalchemy.orm import joinedload, subqueryload, contains_eager
from sqlalchemy import or_, func
from models import (
    Word,
    Definition,
    Meaning,
    Source,
    Language,
    Etymology,
    EtymologyComponent,
    Form,
    HeadTemplate,
    Derivative,
    Example,
    Hypernym,
    Hyponym,
    Meronym,
    Holonym,
    AssociatedWord,
    AlternateForm,
    Inflection,
)
from database import db_session
from functools import wraps
import time
import logging
from datetime import datetime
from unidecode import unidecode
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)

# Error handling within blueprint
@bp.app_errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404


@bp.app_errorhandler(500)
def internal_error(error):
    logger.error("Server Error: %s", error)
    return jsonify({"error": "Internal server error"}), 500


def normalize_word(word):
    if word is None:
        return None
    return unidecode(str(word).lower())

def parse_etymology(etymology):
    cleaned_etymology = etymology.strip("[]")
    parts = re.split(r"\s*\+\s*|\s*-\s*", cleaned_etymology)
    filtered_parts = [part.strip() for part in parts if part.strip()]
    return filtered_parts


def filter_valid_meanings(meanings):
    valid_meanings = []
    for m in meanings:
        if m.meaning and m.meaning.strip() and m.meaning.strip() != "0":
            valid_meanings.append(
                {
                    "definition": m.meaning,
                    "source": m.source.source_name if m.source else None,
                }
            )
    return valid_meanings


def filter_valid_definitions(definitions):
    valid_definitions = []
    for d in definitions:
        valid_meanings = filter_valid_meanings(d.meanings)
        if valid_meanings:
            valid_definitions.append(
                {
                    "partOfSpeech": d.part_of_speech,
                    "meanings": valid_meanings,
                    "usageNotes": d.usage_notes or [],
                    "examples": [e.example for e in d.examples],
                    "tags": d.tags or [],
                }
            )
    return valid_definitions


def filter_empty_values(data):
    return {k: v for k, v in data.items() if v not in (None, [], "", {}, "0")}


def get_word_details(word_entry):
    def filter_existing_words(words):
        return [
            word
            for word in words
            if Word.query.filter(
                func.lower(func.unaccent(Word.word)) == normalize_word(word)
            ).first()
        ]

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
                "ipa": (
                    ", ".join(
                        [p for p in word_entry.audio_pronunciation if p.startswith("/")]
                    )
                    if word_entry.audio_pronunciation
                    else ""
                ),
                "audio": (
                    [
                        p
                        for p in word_entry.audio_pronunciation
                        if p and not p.startswith("/")
                    ]
                    if word_entry.audio_pronunciation
                    else []
                ),
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
                "derivatives": filter_existing_words(
                    [d.derivative for d in word_entry.derivatives]
                ),
                "synonyms": filter_existing_words(
                    [w.word for w in word_entry.synonyms]
                ),
                "antonyms": filter_existing_words(
                    [w.word for w in word_entry.antonyms]
                ),
                "associatedWords": filter_existing_words(
                    [aw.associated_word for aw in word_entry.associated_words]
                ),
                "relatedTerms": filter_existing_words(
                    [w.word for w in word_entry.related_terms]
                ),
                "hypernyms": filter_existing_words(
                    [h.hypernym for h in word_entry.hypernyms]
                ),
                "hyponyms": filter_existing_words(
                    [h.hyponym for h in word_entry.hyponyms]
                ),
                "meronyms": filter_existing_words(
                    [m.meronym for m in word_entry.meronyms]
                ),
                "holonyms": filter_existing_words(
                    [h.holonym for h in word_entry.holonyms]
                ),
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


def get_word_network_data(word_entry):
    return {
        "word": word_entry.word,
        "pronunciation": word_entry.pronunciation,
        "languages": [lang.code for lang in word_entry.languages],
        "definitions": [
            {"part_of_speech": d.part_of_speech, "meaning": m.meaning}
            for d in word_entry.definitions
            for m in d.meanings[:1]
        ],
        "derivatives": [d.derivative for d in word_entry.derivatives],
        "root_word": word_entry.root_word,
        "synonyms": [w.word for w in word_entry.synonyms],
        "antonyms": [w.word for w in word_entry.antonyms],
        "associated_words": [aw.associated_word for aw in word_entry.associated_words],
        "related_terms": [w.word for w in word_entry.related_terms],
        "hypernyms": [h.hypernym for h in word_entry.hypernyms],
        "hyponyms": [h.hyponym for h in word_entry.hyponyms],
        "meronyms": [m.meronym for m in word_entry.meronyms],
        "holonyms": [h.holonym for h in word_entry.holonyms],
        "etymology": parse_etymology(
            word_entry.etymologies[0].etymology_text if word_entry.etymologies else ""
        ),
    }

def get_related_words(word, depth=2, breadth=10):
    visited = set()
    queue = [(word, 0)]
    network = {}

    while queue and len(network) < 100:  # Limit total network size
        current_word, current_depth = queue.pop(0)
        
        if current_word is None or current_word in visited or current_depth > depth:
            continue
        
        visited.add(current_word)
        
        normalized_word = normalize_word(current_word)
        if normalized_word is None:
            continue

        word_entry = Word.query.options(
            contains_eager(Word.definitions).contains_eager(Definition.meanings),
            joinedload(Word.languages),
            joinedload(Word.associated_words),
            joinedload(Word.synonyms),
            joinedload(Word.antonyms),
            joinedload(Word.derivatives),
            joinedload(Word.forms),
            joinedload(Word.etymologies),
            joinedload(Word.related_terms),
            joinedload(Word.hypernyms),
            joinedload(Word.hyponyms),
            joinedload(Word.meronyms),
            joinedload(Word.holonyms)
        ).outerjoin(Word.definitions).outerjoin(Definition.meanings).filter(
            func.lower(func.unaccent(Word.word)) == normalized_word
        ).first()
        
        if word_entry:
            network[current_word] = get_word_network_data(word_entry)
            
            if current_depth < depth:
                related_words = set()
                related_words.update(network[current_word].get("derivatives", []))
                related_words.update(network[current_word].get("synonyms", []))
                related_words.update(network[current_word].get("antonyms", []))
                related_words.update(network[current_word].get("associated_words", []))
                related_words.update(network[current_word].get("related_terms", []))
                related_words.update(network[current_word].get("hypernyms", []))
                related_words.update(network[current_word].get("hyponyms", []))
                related_words.update(network[current_word].get("meronyms", []))
                related_words.update(network[current_word].get("holonyms", []))
                related_words.update(network[current_word].get("etymology", []))
                if network[current_word].get("root_word"):
                    related_words.add(network[current_word]["root_word"])
                
                new_words = list(related_words - visited)[:breadth]
                queue.extend((w, current_depth + 1) for w in new_words if w is not None)

    return network

@bp.route("/api/v1/words", methods=["GET"])
def get_words():
    page = int(request.args.get("page", 1))
    per_page = min(int(request.args.get("per_page", 20)), 100)
    search = request.args.get("search", "")

    query = Word.query

    if search:
        normalized_search = normalize_word(search)
        query = query.filter(
            or_(
                func.lower(func.unaccent(Word.word)).ilike(f"%{normalized_search}%"),
                Word.definitions.any(
                    Definition.meanings.any(
                        func.lower(func.unaccent(Meaning.meaning)).ilike(
                            f"%{normalized_search}%"
                        )
                    )
                ),
            )
        )

    total = query.count()
    words = (
        query.order_by(Word.word).offset((page - 1) * per_page).limit(per_page).all()
    )

    return jsonify(
        {
            "words": [{"word": w.word, "id": w.id} for w in words],
            "page": page,
            "per_page": per_page,
            "total": total,
        }
    )


@bp.route("/api/v1/words/<word>", methods=["GET"])
def get_word(word):
    try:
        normalized_word = normalize_word(word)
        word_entry = Word.query.filter(
            func.lower(func.unaccent(Word.word)) == normalized_word
        ).first()

        if word_entry is None:
            logger.info(f"Word not found: {word}")
            return jsonify({"error": "Word not found"}), 404

        word_entry = (
            Word.query.options(
                joinedload(Word.languages),
                subqueryload(Word.definitions)
                .joinedload(Definition.meanings)
                .joinedload(Meaning.source),
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
                subqueryload(Word.inflections),
            )
            .filter_by(id=word_entry.id)
            .first()
        )

        return jsonify(get_word_details(word_entry))
    except Exception as e:
        logger.error(f"Error in get_word: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500


@bp.route("/api/v1/check_word/<word>", methods=["GET"])
def check_word(word):
    try:
        normalized_word = normalize_word(word)
        word_entry = Word.query.filter(
            func.lower(func.unaccent(Word.word)) == normalized_word
        ).first()
        if word_entry:
            return jsonify({"exists": True, "word": word_entry.word}), 200
        else:
            return jsonify({"exists": False}), 404
    except Exception as e:
        logging.error(f"Error in check_word: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500


@bp.route('/api/v1/word_network/<word>', methods=['GET'])
def get_word_network(word):
    depth = min(int(request.args.get('depth', 2)), 5)
    breadth = min(int(request.args.get('breadth', 10)), 20)
    
    if word is None:
        return jsonify({"error": "Word not provided"}), 400

    network = get_related_words(word, depth, breadth)
    
    if not network:
        return jsonify({"error": "Word not found"}), 404

    return jsonify(network)

@bp.route("/api/v1/etymology/<word>", methods=["GET"])
def get_etymology(word):
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


@bp.route("/api/v1/bulk_words", methods=["POST"])
def bulk_get_words():
    words = request.json.get("words", [])
    if not words or not isinstance(words, list):
        return jsonify({"error": "Invalid input"}), 400

    normalized_words = [normalize_word(w) for w in words]
    word_entries = Word.query.filter(
        func.lower(func.unaccent(Word.word)).in_(normalized_words)
    ).all()
    return jsonify({"words": [get_word_details(w) for w in word_entries]})


@bp.route("/favicon.ico")
def favicon():
    return "", 204


@bp.teardown_request
def remove_session(exception=None):
    db_session.remove()
