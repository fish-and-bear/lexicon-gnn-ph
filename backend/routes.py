from flask import Blueprint, jsonify
from models import Word, Definition, Meaning, Source, AssociatedWord
from database import db_session
import json
import re

bp = Blueprint('api', __name__)

# List of language codes to ignore (case-sensitive)
IGNORE_CODES = ('Esp', 'Ing', 'Tag', 'Hil', 'Seb', 'War', 'Kap', 'Bik')

def remove_language_codes(etymology):
    # Create a regex pattern to match and remove the language codes (case-sensitive)
    pattern = r'\b(?:' + '|'.join(re.escape(code) for code in IGNORE_CODES) + r')\b\s*'
    # Remove the language codes from the string
    cleaned_etymology = re.sub(pattern, '', etymology).strip()
    return cleaned_etymology

def parse_etymology(etymology):
    # Remove brackets
    cleaned_etymology = etymology.strip('[]')
    # Remove any language codes
    cleaned_etymology = remove_language_codes(cleaned_etymology)
    # Split by plus signs or dashes, considering possible spaces around plus signs
    parts = re.split(r'\s*\+\s*|\s*-\s*', cleaned_etymology)
    # Filter out any empty parts or parts that are now just whitespace
    filtered_parts = [part.strip() for part in parts if part.strip()]
    return filtered_parts

def get_word_info(word):
    word_entry = Word.query.filter_by(word=word.lower()).first()
    if not word_entry:
        return None

    # Parse and filter etymology parts
    etymology_parts = parse_etymology(word_entry.etymology)
    root_words = etymology_parts if etymology_parts else []

    # Handle derivatives
    try:
        derivatives = json.loads(word_entry.derivatives) if word_entry.derivatives else {}
    except json.JSONDecodeError:
        derivatives = {"": word_entry.derivatives} if word_entry.derivatives else {}

    # Ensure derivatives is a dictionary and filter out empty entries
    if not isinstance(derivatives, dict):
        derivatives = {"": derivatives}
    derivatives = {k: v for k, v in derivatives.items() if k and v}

    # Filter associated words to remove empty strings or whitespace
    associated_words = [aw.associated_word for aw in word_entry.associated_words if aw.associated_word.strip()]

    # Collect all definitions with their meanings and sources
    definitions = []
    for definition in word_entry.definitions:
        def_entry = {
            "part_of_speech": definition.part_of_speech,
            "meanings": [m.meaning for m in definition.meanings],
            "sources": [s.source for s in definition.sources]
        }
        definitions.append(def_entry)

    return {
        "word": word_entry.word,
        "pronunciation": word_entry.pronunciation,
        "etymology": word_entry.etymology,  # Return the original etymology string
        "language_codes": word_entry.language_codes,  # Include language codes in the response if needed
        "definitions": definitions,  # Include all definitions with meanings and sources
        "derivatives": derivatives,
        "associated_words": associated_words,
        "root_words": root_words,
        "root_word": word_entry.root_word if word_entry.root_word.strip() else None  # Filter out empty root words
    }

@bp.route('/api/words/<word>', methods=['GET'])
def get_word_network(word):
    main_word_info = get_word_info(word)
    if not main_word_info:
        return jsonify({"error": "Word not found"}), 404

    network = {main_word_info["word"]: main_word_info}
    words_to_process = set(
        list(main_word_info["derivatives"].keys()) + 
        list(main_word_info["derivatives"].values()) + 
        main_word_info["associated_words"] + 
        main_word_info["root_words"] + 
        ([main_word_info["root_word"]] if main_word_info["root_word"] else [])
    )

    while words_to_process:
        current_word = words_to_process.pop()
        if current_word and current_word not in network:
            word_info = get_word_info(current_word)
            if word_info and word_info['word'].strip():
                network[current_word] = word_info
                words_to_process.update(
                    set(list(word_info["derivatives"].keys()) + 
                        list(word_info["derivatives"].values()) + 
                        word_info["associated_words"] + 
                        word_info["root_words"] + 
                        ([word_info["root_word"]] if word_info["root_word"] else [])) - 
                    set(network.keys())
                )

    return jsonify(network)

@bp.teardown_request
def remove_session(exception=None):
    db_session.remove()
