#!/usr/bin/env python3
"""
kaikki_processor.py

Processes dictionary entries from Kaikki.org JSONL files.
"""

import json
import logging
import os
import re
import traceback
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json
from tqdm import tqdm

# Absolute imports from dictionary_manager
from backend.dictionary_manager.db_helpers import (
    DEFAULT_LANGUAGE_CODE,  # Assuming DEFAULT_LANGUAGE_CODE is here
    Json,
    RelationshipType,  # Assuming RelationshipType enum is exposed here or in enums directly
    add_linguistic_note,
    get_standardized_pos_id, # Added correct function
    get_uncategorized_pos_id, # Needed as fallback
    get_or_create_word_id,
    insert_definition,
    insert_definition_category,
    insert_definition_example,
    insert_definition_link,
    insert_etymology,
    insert_pronunciation,
    insert_relation,
    insert_word_form,
    insert_word_template,
    with_transaction,
)
from backend.dictionary_manager.text_helpers import (
    NON_WORD_STRINGS,
    VALID_BAYBAYIN_REGEX,
    BaybayinRomanizer,
    clean_html,
    extract_script_info, # Add missing import
    get_non_word_note_type,
    standardize_source_identifier,
    # process_kaikki_lemma, # Moved to text_helpers, should be imported if needed, but seems unused now
)
# Import enums directly (if RelationshipType isn't in db_helpers)
# from backend.dictionary_manager.enums import RelationshipType

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Relationship Processing Logic (Moved from dictionary_manager.py)
# -------------------------------------------------------------------
@with_transaction(commit=True)
def process_relationships(cur, word_id, data, sources):
    """Process relationship data for a word."""
    if not data or not word_id:
        return

    try:
        # Extract relationships from definitions
        definitions = []
        if "definitions" in data and isinstance(data["definitions"], list):
            definitions = [
                d.get("text") if isinstance(d, dict) else d for d in data["definitions"]
            ]
        elif "definition" in data and data["definition"]:
            definitions = [data["definition"]]

        # Process each definition for relationships
        for definition in definitions:
            if not definition or not isinstance(definition, str):
                continue

            # Look for synonyms in definitions
            syn_patterns = [
                r"ka(singkahulugan|tulad) ng\s+(\w+)",  # Tagalog
                r"syn(onym)?[\.:]?\s+(\w+)",  # English
                r"same\s+as\s+(\w+)",  # English
                r"another\s+term\s+for\s+(\w+)",  # English
                r"another\s+word\s+for\s+(\w+)",  # English
                r"also\s+called\s+(\w+)",  # English
                r"also\s+known\s+as\s+(\w+)",  # English
            ]

            for pattern in syn_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    synonym = (
                        match[1]
                        if isinstance(match, tuple) and len(match) > 1
                        else match
                    )
                    if synonym and isinstance(synonym, str):
                        # Get or create the synonym word
                        lang_code = data.get("language_code", "tl")
                        syn_id = get_or_create_word_id(
                            cur, synonym.strip(), language_code=lang_code
                        )

                        # Insert the synonym relationship
                        insert_relation(
                            cur, word_id, syn_id, "synonym", sources=sources
                        )

                        # For bidirectional synonyms
                        insert_relation(
                            cur, syn_id, word_id, "synonym", sources=sources
                        )

            # Look for antonyms in definitions
            ant_patterns = [
                r"(kasalungat|kabaligtaran) ng\s+(\w+)",  # Tagalog
                r"ant(onym)?[\.:]?\s+(\w+)",  # English
                r"opposite\s+of\s+(\w+)",  # English
                r"contrary\s+to\s+(\w+)",  # English
            ]

            for pattern in ant_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    antonym = (
                        match[1]
                        if isinstance(match, tuple) and len(match) > 1
                        else match
                    )
                    if antonym and isinstance(antonym, str):
                        # Get or create the antonym word
                        lang_code = data.get("language_code", "tl")
                        ant_id = get_or_create_word_id(
                            cur, antonym.strip(), language_code=lang_code
                        )

                        # Insert the antonym relationship
                        insert_relation(
                            cur, word_id, ant_id, "antonym", sources=sources
                        )

                        # For bidirectional antonyms
                        insert_relation(
                            cur, ant_id, word_id, "antonym", sources=sources
                        )

            # Look for hypernyms (broader terms) in definitions
            hyper_patterns = [
                r"uri ng\s+(\w+)",  # Tagalog
                r"type of\s+(\w+)",  # English
                r"kind of\s+(\w+)",  # English
                r"form of\s+(\w+)",  # English
                r"variety of\s+(\w+)",  # English
                r"species of\s+(\w+)",  # English
                r"member of\s+the\s+(\w+)\s+family",  # English
            ]

            for pattern in hyper_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    hypernym = match
                    if hypernym and isinstance(hypernym, str):
                        # Get or create the hypernym word
                        lang_code = data.get("language_code", "tl")
                        hyper_id = get_or_create_word_id(
                            cur, hypernym.strip(), language_code=lang_code
                        )

                        # Insert the hypernym relationship
                        insert_relation(
                            cur, word_id, hyper_id, "hyponym_of", sources=sources
                        )

                        # Add the inverse relationship
                        insert_relation(
                            cur, hyper_id, word_id, "hypernym_of", sources=sources
                        )

            # Look for variations in definitions
            var_patterns = [
                r"(iba\'t ibang|ibang) (anyo|baybay|pagsulat|bigkas) ng\s+(\w+)",  # Tagalog: different form/spelling/pronunciation of
                r"(alternatibo|alternativ|kahalili) ng\s+(\w+)",  # Tagalog: alternative of
                r"(variant|variation|alt(ernative)?) (form|spelling) of\s+(\w+)",  # English
                r"alternative (to|for)\s+(\w+)",  # English
                r"also (written|spelled) as\s+(\w+)",  # English
                r"(var\.|variant)\s+(\w+)",  # English abbreviated
                r"(regional|dialectal) form of\s+(\w+)",  # English regional variant
                r"(slang|informal) for\s+(\w+)",  # English slang variant
                r"commonly (misspelled|written) as\s+(\w+)",  # English common misspelling
                r"(baryant|lokal na anyo) ng\s+(\w+)",  # Tagalog regional variant
            ]

            for pattern in var_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    # Different patterns have target word in different positions
                    variant = None
                    if len(match) == 3 and isinstance(
                        match, tuple
                    ):  # For patterns with 3 capture groups
                        variant = match[2]
                    elif len(match) == 2 and isinstance(
                        match, tuple
                    ):  # For patterns with 2 capture groups
                        variant = match[1]
                    elif isinstance(match, str):  # For patterns with 1 capture group
                        variant = match

                    if variant and isinstance(variant, str):
                        # Get or create the variant word
                        lang_code = data.get("language_code", "tl")
                        var_id = get_or_create_word_id(
                            cur, variant.strip(), language_code=lang_code
                        )

                        # Insert the variant relationship
                        insert_relation(
                            cur, word_id, var_id, "variant", sources=sources
                        )

                        # For bidirectional variant relationship
                        insert_relation(
                            cur, var_id, word_id, "variant", sources=sources
                        )

        # Process derivative information
        derivative = data.get("derivative", "")
        if derivative and isinstance(derivative, str):
            # This indicates the word is derived from another root
            mula_sa_patterns = [
                r"mula sa\s+(.+?)(?:\s+na|\s*$)",  # Tagalog
                r"derived from\s+(?:the\s+)?(\w+)",  # English
                r"comes from\s+(?:the\s+)?(\w+)",  # English
                r"root word(?:\s+is)?\s+(\w+)",  # English
            ]

            for pattern in mula_sa_patterns:
                root_match = re.search(pattern, derivative, re.IGNORECASE)
                if root_match:
                    root_word = root_match.group(1).strip()
                    if root_word:
                        # Get or create the root word
                        lang_code = data.get("language_code", "tl")
                        root_id = get_or_create_word_id(
                            cur, root_word, language_code=lang_code
                        )

                        # Insert the derived_from relationship
                        insert_relation(
                            cur, word_id, root_id, "derived_from", sources=sources
                        )

                        # Add the inverse relationship
                        insert_relation(
                            cur, root_id, word_id, "root_of", sources=sources
                        )

        # Process etymology information for potential language relationships
        etymology = data.get("etymology", "")
        if etymology and isinstance(etymology, str):
            # Try to extract language information from etymology
            lang_patterns = {
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:Spanish|Esp)[\.:]?\s+(\w+)": "es",  # Spanish
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:English|Eng)[\.:]?\s+(\w+)": "en",  # English
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:Chinese|Ch|Tsino)[\.:]?\s+(\w+)": "zh",  # Chinese
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:Japanese|Jap)[\.:]?\s+(\w+)": "ja",  # Japanese
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:Sanskrit|San)[\.:]?\s+(\w+)": "sa",  # Sanskrit
            }

            for pattern, lang_code in lang_patterns.items():
                lang_matches = re.findall(pattern, etymology, re.IGNORECASE)
                for lang_word in lang_matches:
                    if lang_word and isinstance(lang_word, str):
                        # Get or create the foreign word
                        foreign_id = get_or_create_word_id(
                            cur, lang_word.strip(), language_code=lang_code
                        )

                        # Insert the etymology relationship
                        insert_relation(
                            cur, word_id, foreign_id, "borrowed_from", sources=sources
                        )

        # Process alternate forms and variations
        # Check if there's variations data in metadata
        variations = data.get("variations", [])
        if variations and isinstance(variations, list):
            for variant in variations:
                if isinstance(variant, str) and variant.strip():
                    # Add this explicit variation
                    var_id = get_or_create_word_id(
                        cur,
                        variant.strip(),
                        language_code=data.get("language_code", "tl"),
                    )
                    insert_relation(cur, word_id, var_id, "variant", sources=sources)
                    insert_relation(cur, var_id, word_id, "variant", sources=sources)
                elif isinstance(variant, dict) and "form" in variant:
                    var_form = variant.get("form", "").strip()
                    if var_form:
                        var_type = variant.get("type", "variant")
                        var_id = get_or_create_word_id(
                            cur, var_form, language_code=data.get("language_code", "tl")
                        )

                        # Use specific relationship type if provided, otherwise default to "variant"
                        rel_type = (
                            var_type
                            if var_type
                            in [
                                "abbreviation",
                                "misspelling",
                                "regional",
                                "alternate",
                                "dialectal",
                            ]
                            else "variant"
                        )
                        insert_relation(cur, word_id, var_id, rel_type, sources=sources)
                        insert_relation(cur, var_id, word_id, rel_type, sources=sources)

        # Look for variations by checking spelling differences
        # This will detect common spelling variations in Filipino like f/p, e/i, o/u substitutions
        word = data.get("word", "")
        if word and isinstance(word, str) and len(word) > 3:
            # Common letter substitutions in Filipino
            substitutions = [
                ("f", "p"),
                ("p", "f"),  # Filipino/Pilipino
                ("e", "i"),
                ("i", "e"),  # like in leeg/liig (neck)
                ("o", "u"),
                ("u", "o"),  # like in puso/poso (heart)
                ("k", "c"),
                ("c", "k"),  # like in karera/carera (race)
                ("w", "u"),
                ("u", "w"),  # like in uwi/uwi (go home)
                ("j", "h"),
                ("h", "j"),  # like in jahit/hahit
                ("s", "z"),
                ("z", "s"),  # like in kasoy/kazoy
                ("ts", "ch"),
                ("ch", "ts"),  # like in tsaa/chaa (tea)
            ]

            # Generate possible variations
            potential_variations = []
            for i, char in enumerate(word):
                for orig, repl in substitutions:
                    if char.lower() == orig:
                        var = word[:i] + repl + word[i + 1 :]
                        potential_variations.append(var)
                    elif (
                        char.lower() == orig[0]
                        and i < len(word) - 1
                        and word[i + 1].lower() == orig[1]
                    ):
                        var = word[:i] + repl + word[i + 2 :]
                        potential_variations.append(var)

            # Check if these variations actually exist in the database
            for var in potential_variations:
                # Skip if the variation is the same as the original word
                if var.lower() == word.lower():
                    continue

                cur.execute(
                    """
                    SELECT id FROM words 
                    WHERE normalized_lemma = %s AND language_code = %s
                """,
                    (normalize_lemma(var), data.get("language_code", "tl")),
                )

                result = cur.fetchone()
                if result:
                    # Found a real variation in the database
                    var_id = result[0]
                    insert_relation(
                        cur, word_id, var_id, "spelling_variant", sources=sources
                    )
                    insert_relation(
                        cur, var_id, word_id, "spelling_variant", sources=sources
                    )

    except Exception as e:
        logger.error(f"Error processing relationships for word_id {word_id}: {str(e)}")
        if hasattr(e, "__traceback__"):
            import traceback

            tb_str = "".join(traceback.format_tb(e.__traceback__))
            logger.error(f"Traceback: {tb_str}")


@with_transaction(commit=True)
def process_direct_relations(cur, word_id, entry, lang_code, source):
    """Process direct relationships specified in the entry."""
    relationship_mappings = {
        "synonyms": ("synonym", True),  # bidirectional
        "antonyms": ("antonym", True),  # bidirectional
        "derived": ("derived_from", False),  # not bidirectional, direction is important
        "related": ("related", True),  # bidirectional
    }

    for rel_key, (rel_type, bidirectional) in relationship_mappings.items():
        if rel_key in entry and isinstance(entry[rel_key], list):
            for rel_item in entry[rel_key]:
                # Initialize metadata
                metadata = {}

                # Handle both string words and dictionary objects with word property
                if isinstance(rel_item, dict) and "word" in rel_item:
                    rel_word = rel_item["word"]

                    # Extract metadata fields if available
                    if "strength" in rel_item:
                        metadata["strength"] = rel_item["strength"]

                    if "tags" in rel_item and rel_item["tags"]:
                        metadata["tags"] = rel_item["tags"]

                    if "english" in rel_item and rel_item["english"]:
                        metadata["english"] = rel_item["english"]

                    # Extract any other useful fields
                    for field in ["sense", "extra", "notes"]:
                        if field in rel_item and rel_item[field]:
                            metadata[field] = rel_item[field]

                elif isinstance(rel_item, str):
                    rel_word = rel_item
                else:
                    continue

                # Skip empty strings
                if not rel_word or not isinstance(rel_word, str):
                    continue

                # For derived words, the entry word is derived from the related word
                if rel_type == "derived_from":
                    from_id = word_id
                    to_id = get_or_create_word_id(
                        cur, rel_word, language_code=lang_code
                    )
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(
                            cur,
                            from_id,
                            to_id,
                            rel_type,
                            sources=source,
                            metadata=metadata,
                        )
                    else:
                        insert_relation(cur, from_id, to_id, rel_type, sources=source)
                else:
                    to_id = get_or_create_word_id(
                        cur, rel_word, language_code=lang_code
                    )
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(
                            cur,
                            word_id,
                            to_id,
                            rel_type,
                            sources=source,
                            metadata=metadata,
                        )
                    else:
                        insert_relation(cur, word_id, to_id, rel_type, sources=source)

                    # Add bidirectional relationship if needed
                    if bidirectional:
                        # For bidirectional relationships, we might want to copy the metadata
                        if metadata:
                            insert_relation(
                                cur,
                                to_id,
                                word_id,
                                rel_type,
                                sources=source,
                                metadata=metadata,
                            )
                        else:
                            insert_relation(
                                cur, to_id, word_id, rel_type, sources=source
                            )


@with_transaction(commit=True)
def process_relations(
    cur,
    from_word_id: int,
    relations_dict: Dict[str, List[str]],
    lang_code: str,
    source: str,
):
    """
    Process relationship data from dictionary.
    Convert each raw Kaikki relation key using normalize_relation_type,
    then create appropriate relations in the database.
    """
    for raw_key, related_list in relations_dict.items():
        if not related_list:
            continue
        # Normalize relation key
        relation_type, bidirectional, inverse_type = normalize_relation_type(raw_key)

        for rel_item in related_list:
            metadata = {}

            # Handle both string values and dictionary objects
            if isinstance(rel_item, dict) and "word" in rel_item:
                rel_word_lemma = rel_item["word"]

                # Extract metadata fields if available
                if "strength" in rel_item:
                    metadata["strength"] = rel_item["strength"]

                if "tags" in rel_item and rel_item["tags"]:
                    metadata["tags"] = rel_item["tags"]

                if "english" in rel_item and rel_item["english"]:
                    metadata["english"] = rel_item["english"]

                # Extract any other useful fields
                for field in ["sense", "extra", "notes", "context"]:
                    if field in rel_item and rel_item[field]:
                        metadata[field] = rel_item[field]
            elif isinstance(rel_item, str):
                rel_word_lemma = rel_item
            else:
                continue

            to_word_id = get_or_create_word_id(
                cur, rel_word_lemma, language_code=lang_code
            )

            # Only include metadata if it's not empty
            if metadata:
                insert_relation(
                    cur,
                    from_word_id,
                    to_word_id,
                    relation_type,
                    sources=source,
                    metadata=metadata,
                )
            else:
                insert_relation(
                    cur, from_word_id, to_word_id, relation_type, sources=source
                )

            if bidirectional and inverse_type:
                # For bidirectional relationships, we might want to copy the metadata
                if metadata:
                    insert_relation(
                        cur,
                        to_word_id,
                        from_word_id,
                        inverse_type,
                        sources=source,
                        metadata=metadata,
                    )
                else:
                    insert_relation(
                        cur, to_word_id, from_word_id, inverse_type, sources=source
                    )


@with_transaction(commit=True)
def extract_sense_relations(cur, word_id, sense, lang_code, source):
    """Extract and process relationship data from a word sense."""
    for rel_type in ["synonyms", "antonyms", "derived", "related"]:
        if rel_type in sense and isinstance(sense[rel_type], list):
            relation_items = sense[rel_type]
            relationship_type = (
                "synonym"
                if rel_type == "synonyms"
                else (
                    "antonym"
                    if rel_type == "antonyms"
                    else "derived_from" if rel_type == "derived" else "related"
                )
            )
            bidirectional = (
                rel_type != "derived"
            )  # derived relationships are not bidirectional

            for item in relation_items:
                # Initialize metadata
                metadata = {}

                # Handle both string words and dictionary objects with word property
                if isinstance(item, dict) and "word" in item:
                    rel_word = item["word"]

                    # Extract metadata fields if available
                    if "strength" in item:
                        metadata["strength"] = item["strength"]

                    if "tags" in item and item["tags"]:
                        metadata["tags"] = item["tags"]

                    if "english" in item and item["english"]:
                        metadata["english"] = item["english"]

                    # Extract sense-specific context if available
                    if "sense" in item and item["sense"]:
                        metadata["sense"] = item["sense"]

                    # Extract any other useful fields
                    for field in ["extra", "notes", "context"]:
                        if field in item and item[field]:
                            metadata[field] = item[field]
                elif isinstance(item, str):
                    rel_word = item
                else:
                    continue

                # Skip empty strings
                if not rel_word or not isinstance(rel_word, str):
                    continue

                # For derived words, the entry word is derived from the related word
                if relationship_type == "derived_from":
                    from_id = word_id
                    to_id = get_or_create_word_id(
                        cur, rel_word, language_code=lang_code
                    )
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(
                            cur,
                            from_id,
                            to_id,
                            relationship_type,
                            sources=source,
                            metadata=metadata,
                        )
                    else:
                        insert_relation(
                            cur, from_id, to_id, relationship_type, sources=source
                        )
                else:
                    to_id = get_or_create_word_id(
                        cur, rel_word, language_code=lang_code
                    )
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(
                            cur,
                            word_id,
                            to_id,
                            relationship_type,
                            sources=source,
                            metadata=metadata,
                        )
                    else:
                        insert_relation(
                            cur, word_id, to_id, relationship_type, sources=source
                        )

                    # Add bidirectional relationship if needed
                    if bidirectional:
                        # For bidirectional relationships, we might want to copy the metadata
                        if metadata:
                            insert_relation(
                                cur,
                                to_id,
                                word_id,
                                relationship_type,
                                sources=source,
                                metadata=metadata,
                            )
                        else:
                            insert_relation(
                                cur, to_id, word_id, relationship_type, sources=source
                            )

# -------------------------------------------------------------------
# Main Kaikki Processing Function (Moved from dictionary_manager.py)
# -------------------------------------------------------------------

@with_transaction(
    commit=True
)  # Changed to commit=True to follow process_tagalog_words pattern
def process_kaikki_jsonl(cur, filename: str):
    """Process Kaikki.org dictionary entries with optimized transaction handling."""
    # Configurable parameters for performance tuning
    CHUNK_SIZE = 50  # Process in chunks of 50 entries
    MAX_RETRIES = 3  # Retry failed chunks up to 3 times
    PROGRESS_REPORT_INTERVAL = 100  # Report progress every 100 entries
    
    # Initialize statistics
    entry_stats = {
        "definitions": 0,
        "relations": 0,
        "etymologies": 0,
        "pronunciations": 0,
        "forms": 0,
        "categories": 0,
        "templates": 0,
        "links": 0,
        "scripts": 0,
        "ety_relations": 0,
        "sense_relations": 0,
        "form_relations": 0,
        "examples": 0,
    }

    # Cache common lookups to reduce database queries
    word_cache = {}  # Format: "{lemma}|{lang_code}" -> word_id
    pos_code_cache = {}  # POS code -> standardized_pos_id

    # Standardize the source identifier from the filename
    source_identifier = standardize_source_identifier(os.path.basename(filename))
    raw_source_identifier = os.path.basename(filename)
    logger.info(
        f"Processing Kaikki dictionary: {filename} with source ID: '{source_identifier}'"
    )

    # Count total lines in file
    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f_count:
            total_lines = sum(1 for _ in f_count)
        logger.info(f"Found {total_lines} entries to process in {filename}")
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return {
            "total_entries": 0,
            "processed_entries": 0,
            "error_entries": 1,
            "skipped_entries": 0,
        }
    except Exception as count_error:
        logger.error(f"Error counting lines in {filename}: {count_error}")
        total_lines = -1  # Indicate unknown total

    # Initialize romanizer
    romanizer = None
    try:
        if "BaybayinRomanizer" in globals() and callable(
            globals()["BaybayinRomanizer"]
        ):
            romanizer = BaybayinRomanizer()
            logger.info("Initialized BaybayinRomanizer for script romanization")
        else:
            logger.warning("BaybayinRomanizer class not found. Cannot initialize.")
    except Exception as rom_err:
        logger.warning(f"Could not initialize BaybayinRomanizer: {rom_err}")

    # --- Process entries in chunks for better performance ---
    def process_entry_chunk(entries_chunk, chunk_idx, retry_count=0):
        """Process a chunk of entries to reduce overhead."""
        processed_ok = 0
        processed_with_errors = 0
        failed_entries = 0
        error_details = {}
        chunk_stats = {k: 0 for k in entry_stats.keys()}
        
        try:
            # Start a transaction for the entire chunk
            cur.execute("BEGIN;")
            
            for entry in entries_chunk:
                result, error = process_single_entry(entry)
                
                if result:
                    # Count success or partial success
                    if error:
                        processed_with_errors += 1
                        error_key = error[:100] if error else "Unknown error"
                        error_details[error_key] = error_details.get(error_key, 0) + 1
                    else:
                        processed_ok += 1
                else:
                    # Count complete failure
                    failed_entries += 1
                    error_key = error[:100] if error else "Unknown critical failure"
                    error_details[error_key] = error_details.get(error_key, 0) + 1
                    
                # Accumulate stats from successful processing
                for stat_key, stat_value in result.get('stats', {}).items():
                    if stat_key in chunk_stats:
                        chunk_stats[stat_key] += stat_value
                        
            # If we reach here without exceptions, commit the chunk
            cur.execute("COMMIT;")
            logger.debug(f"Successfully committed chunk {chunk_idx} with {len(entries_chunk)} entries")
            
            return {
                "processed_ok": processed_ok,
                "processed_with_errors": processed_with_errors,
                "failed_entries": failed_entries,
                "error_details": error_details,
                "stats": chunk_stats
            }
            
        except psycopg2.Error as db_err:
            # Database error handling
            cur.execute("ROLLBACK;")
            
            if retry_count < MAX_RETRIES:
                logger.warning(f"Database error in chunk {chunk_idx}, attempt {retry_count+1}/{MAX_RETRIES}: {db_err}")
                logger.warning(f"Retrying chunk {chunk_idx}...")
                # Recursive retry with incremented retry counter
                return process_entry_chunk(entries_chunk, chunk_idx, retry_count + 1)
            else:
                logger.error(f"Failed to process chunk {chunk_idx} after {MAX_RETRIES} attempts: {db_err}")
                # Return integer counts for failed entries
                failed_entry_count = len(entries_chunk)
                return {
                    "processed_ok": 0,
                    "processed_with_errors": 0,
                    "failed_entries": failed_entry_count,
                    "error_details": {
                        f"Database Error ({type(db_err).__name__})": failed_entry_count
                    },
                    "stats": {k: 0 for k in entry_stats.keys()},
                }
        except Exception as e:
            # Other exception handling
            cur.execute("ROLLBACK;")
            logger.error(f"Unexpected error in chunk {chunk_idx}: {e}", exc_info=True)
            
            if retry_count < MAX_RETRIES:
                logger.warning(f"Retrying chunk {chunk_idx} after error...")
                return process_entry_chunk(entries_chunk, chunk_idx, retry_count + 1)
            else:
                logger.error(f"Failed to process chunk {chunk_idx} after {MAX_RETRIES} attempts: {e}")
                # Return integer counts for failed entries
                failed_entry_count = len(entries_chunk)
                return {
                    "processed_ok": 0,
                    "processed_with_errors": 0,
                    "failed_entries": failed_entry_count,
                    "error_details": {
                        f"Unexpected Error ({type(e).__name__})": failed_entry_count
                    },
                    "stats": {k: 0 for k in entry_stats.keys()},
                }

    # --- Single entry processor ---
    def process_single_entry(entry):
        """Process a single dictionary entry with optimized data handling."""
        if "word" not in entry or not entry["word"]:
            return {"word_id": None, "stats": {}, "success": False}, "No word field"

        word = entry["word"]
        # Apply special Kaikki lemma processing to preserve parenthesized variants and handle numbers
        # processed_word = process_kaikki_lemma(word)
        processed_word = word
        word_id = None
        error_messages = []
        local_stats = {k: 0 for k in entry_stats.keys()}

        try:
            # Get language code and validate
            pos = entry.get("pos", "unc")
            language_code = entry.get("lang_code", DEFAULT_LANGUAGE_CODE)
            if not language_code or len(language_code) > 10:
                language_code = DEFAULT_LANGUAGE_CODE

            # Extract Baybayin and other script info
            baybayin_form, baybayin_romanized = extract_script_info(
                entry, "Baybayin", "Baybayin"
            )
            badlit_form, badlit_romanized = extract_script_info(
                entry, "Badlit", "Badlit"
            )

            # --- Baybayin Validation ---
            validated_baybayin_form = None
            has_baybayin = False
            # --- BEGIN IMPROVED BAYBAYIN EXTRACTION ---
            original_baybayin_for_log = baybayin_form  # Store original for logging if needed
            if isinstance(baybayin_form, str) and baybayin_form:
                # Attempt to extract pure Baybayin script from potentially mixed strings
                # Find all sequences of Baybayin characters (U+1700-U+171F) and punctuation (U+1735)
                # Use a slightly more specific pattern to capture contiguous blocks
                baybayin_matches = re.findall(r"([\u1700-\u171F\u1735]+)", baybayin_form)
                if baybayin_matches:
                    # Join all found sequences, separated by space, as the intended script
                    extracted_script = " ".join(baybayin_matches)
                    # Check if this extracted script is different from the original potentially mixed string
                    if extracted_script != original_baybayin_for_log:  # Compare with original
                        # Log the original and the extracted version
                        logger.info(f"Extracted Baybayin '{extracted_script}' from mixed string '{original_baybayin_for_log}' for word '{word}'.")
                    baybayin_form = extracted_script  # Replace the original with the extracted pure script
                # If no Baybayin characters were found, baybayin_form remains as it was (possibly invalid)
            # --- END IMPROVED BAYBAYIN EXTRACTION ---
            if baybayin_form:
                # Check if the form contains only valid characters
                # Stripping the problematic dotted circle U+25CC proactively
                cleaned_baybayin_form = baybayin_form.replace("\u25cc", "").strip()
                if cleaned_baybayin_form and VALID_BAYBAYIN_REGEX.match(cleaned_baybayin_form):
                    validated_baybayin_form = cleaned_baybayin_form
                    has_baybayin = True
                else:
                    # Log if the original form was non-empty but became invalid/empty after cleaning
                    if baybayin_form:
                        logger.warning(f"Invalid or problematic Baybayin form found for '{word}': '{baybayin_form}'. Ignoring Baybayin for this entry.")
                        error_messages.append(f"Invalid Baybayin form '{baybayin_form}' ignored")

            # Generate romanization if needed, using the validated form
            romanized_form = None
            if (
                validated_baybayin_form
                and not baybayin_romanized
                and romanizer
                and romanizer.validate_text(validated_baybayin_form)
            ):
                try:
                    romanized_form = romanizer.romanize(validated_baybayin_form)
                except Exception as rom_err:
                    # Log the romanization error instead of passing silently
                    logger.warning(
                        f"Could not romanize Baybayin '{validated_baybayin_form}' for word '{word}': {rom_err}"
                    )
                    error_messages.append(f"Romanization error: {rom_err}")

            if not romanized_form and baybayin_romanized:
                romanized_form = baybayin_romanized
            elif not romanized_form and badlit_romanized:
                romanized_form = badlit_romanized

            if (
                has_baybayin or badlit_form
            ):  # Count scripts based on validated/present forms
                local_stats["scripts"] += 1

            # Check cache for existing word
            cache_key = f"{processed_word.lower()}|{language_code}"
            if cache_key in word_cache:
                word_id = word_cache[cache_key]
            else:
                # Prepare word attributes - MOVED OUTSIDE THE CONDITIONAL BLOCK
                is_proper_noun = entry.get("proper", False) or pos in [
                    "prop",
                    "proper noun",
                    "name",
                ]
                is_abbreviation = pos in ["abbrev", "abbreviation"]
                is_initialism = pos in ["init", "initialism", "acronym"]

                # Process tags
                tags_list = entry.get("tags", [])
                if isinstance(tags_list, list):
                    if any(t in ["abbreviation", "abbrev"] for t in tags_list):
                        is_abbreviation = True
                    if any(t in ["initialism", "acronym"] for t in tags_list):
                        is_initialism = True

                # Format tag string and prepare metadata
                word_tags_str = ",".join(tags_list) if tags_list else None
                word_metadata = {"source_file": raw_source_identifier}

                # Handle hyphenation
                hyphenation = entry.get("hyphenation")
                hyphenation_json = (
                    Json(hyphenation)
                    if hyphenation and isinstance(hyphenation, list)
                    else None
                )

                # Create or get word ID - use processed_word instead of original word
                # Create or get word ID - use processed_word instead of original word
                word_id = get_or_create_word_id(
                    cur,
                    lemma=processed_word,  # Use the specially processed word
                    language_code=language_code,
                    source_identifier=source_identifier,
                    preserve_numbers=True,  # Add this parameter to prevent further number processing
                    has_baybayin=has_baybayin,
                    baybayin_form=validated_baybayin_form,
                    romanized_form=romanized_form,
                    badlit_form=badlit_form,
                    hyphenation=hyphenation,
                    is_proper_noun=is_proper_noun,
                    is_abbreviation=is_abbreviation,
                    is_initialism=is_initialism,
                    tags=word_tags_str,
                    word_metadata=word_metadata,
                )

                # Add to cache for potential reuse
                if word_id:
                    word_cache[cache_key] = word_id

            if not word_id:
                return None, f"Failed to get/create word ID for '{word}'"

            # Process related data in properly grouped operations

            # 1. Process pronunciations
            if "sounds" in entry and isinstance(entry["sounds"], list):
                for sound in entry.get("sounds", []):
                    if not isinstance(sound, dict):
                        continue

                    # Extract pronunciation info
                    pron_type, pron_value = None, None
                    if "ipa" in sound and sound["ipa"]:
                        pron_type, pron_value = "ipa", sound["ipa"]
                    elif "enpr" in sound and sound["enpr"]:
                        pron_type, pron_value = "enpr", sound["enpr"]
                    elif "rhymes" in sound and sound["rhymes"]:
                        pron_type, pron_value = "rhyme", sound["rhymes"]
                    elif "audio" in sound and sound["audio"]:
                        pron_type, pron_value = "audio", sound["audio"]
                    else:
                        continue

                    if pron_type and pron_value:
                        pron_data = sound.copy()
                        pron_data["type"], pron_data["value"] = pron_type, pron_value

                        try:
                            pron_id = insert_pronunciation(
                                cur, word_id, pron_data, source_identifier
                            )
                            if pron_id:
                                local_stats["pronunciations"] += 1
                        except Exception as e:
                            error_messages.append(
                                f"Pronunciation error ({pron_type}): {str(e)}"
                            )

            # 2. Process word forms and relationships together
            if "forms" in entry and isinstance(entry["forms"], list):
                # First collect canonical forms identified in the data
                canonical_forms = set()
                for form_check in entry["forms"]:
                    if (
                        isinstance(form_check, dict)
                        and "form" in form_check
                        and form_check.get("form")
                        and "tags" in form_check
                        and "canonical" in form_check.get("tags", [])
                    ):
                        cf_text = form_check["form"].strip()
                        if cf_text:
                            canonical_forms.add(cf_text)

                # Process each form entry in the list
                for form_data_idx, form_data in enumerate(entry["forms"]):
                    if not isinstance(form_data, dict) or not form_data.get("form"):
                        logger.debug(
                            f"Skipping invalid form data item at index {form_data_idx} for word '{word}'."
                        )
                        continue

                    form_text = form_data["form"].strip()

                    # --- MODIFICATION START: Filter non-word forms and capture full metadata ---
                    if not form_text:
                        logger.debug(
                            f"Skipping empty form text at index {form_data_idx} for word '{word}'."
                        )
                        continue  # Skip empty forms

                    # Filter out forms that look like template names or placeholders
                    if (
                        form_text.startswith("tl-")
                        or ":" in form_text
                        or form_text == "no-table-tags"
                    ):
                        logger.debug(
                            f"Skipping template/placeholder form '{form_text}' at index {form_data_idx} for word '{word}'."
                        )
                        continue

                    # Prepare the metadata dictionary - include everything from form_data except 'form'
                    form_metadata = {k: v for k, v in form_data.items() if k != "form"}

                    # Explicitly ensure 'tags' exists in metadata, defaulting to empty list if missing
                    if "tags" not in form_metadata:
                        form_metadata["tags"] = []
                    elif not isinstance(form_metadata["tags"], list):
                        # Attempt to convert non-list tags to list
                        logger.warning(
                            f"Form tags for '{form_text}' (word '{word}') are not a list ({type(form_metadata['tags'])}). Attempting conversion."
                        )
                        form_metadata["tags"] = (
                            [str(form_metadata["tags"])]
                            if form_metadata["tags"]
                            else []
                        )

                    # Determine if this specific form is marked as canonical
                    is_canonical = form_text in canonical_forms
                    # Add is_canonical status to the metadata for storage
                    form_metadata["is_canonical"] = is_canonical

                    # Store the form using a dedicated function (implementation needed)
                    try:
                        # Assumes insert_word_form accepts a 'metadata' argument (JSONB)
                        # def insert_word_form(cur, word_id, form, metadata=None, source_identifier=None):
                        #    # ... implementation using WordFormModel ...
                        #    return form_db_id or None

                        if "insert_word_form" in globals() and callable(
                            globals()["insert_word_form"]
                        ):
                            form_id = insert_word_form(
                                cur,
                                word_id,
                                form_text,
                                metadata=form_metadata,  # Pass the whole metadata dict
                                source_identifier=source_identifier,
                            )
                            if form_id:
                                local_stats["forms"] += 1
                            else:
                                logger.warning(
                                    f"insert_word_form returned non-ID for form '{form_text}' (Word ID: {word_id}). Metadata: {form_metadata}"
                                )
                                error_messages.append(
                                    f"Failed to store form '{form_text}'"
                                )
                        else:
                            if "insert_word_form_missing_logged" not in locals():
                                logger.error(
                                    f"Function 'insert_word_form' not defined. Cannot store forms for word ID {word_id}."
                                )
                                error_messages.append(
                                    f"Missing function: insert_word_form"
                                )
                                insert_word_form_missing_logged = True

                    except Exception as form_e:
                        error_msg = f"Error storing form '{form_text}' for word ID {word_id}: {str(form_e)}. Metadata: {form_metadata}"
                        logger.error(error_msg, exc_info=True)
                        error_messages.append(error_msg)
                    # --- MODIFICATION END ---

            # --- MODIFICATION START: Process and store inflection templates ---
            if "inflection_templates" in entry and isinstance(
                entry["inflection_templates"], list
            ):
                for template_data_idx, template_data in enumerate(
                    entry["inflection_templates"]
                ):
                    if not isinstance(template_data, dict):
                        logger.debug(
                            f"Skipping non-dict item in inflection_templates at index {template_data_idx} for word '{word}'."
                        )
                        continue

                    template_name = template_data.get("name")
                    template_args = template_data.get(
                        "args"
                    )  # Expected to be a dictionary

                    if not template_name or not isinstance(template_name, str):
                        logger.debug(
                            f"Skipping inflection template item at index {template_data_idx} due to missing/invalid name for word '{word}'."
                        )
                        continue  # Skip if template name is missing or invalid

                    template_name = template_name.strip()
                    if not template_name:
                        continue  # Skip empty names after stripping

                    # Ensure args is a dictionary, default to empty if not or if missing
                    if not isinstance(template_args, dict):
                        # Log if args was present but not a dict
                        if "args" in template_data:
                            logger.warning(
                                f"Inflection template '{template_name}' for word '{word}' has non-dict 'args' ({type(template_args)}). Storing empty args."
                            )
                        template_args = {}  # Default to empty dict

                    # Store the template info using a dedicated function (implementation needed)
                    try:
                        # Assumes insert_word_template accepts 'args' as a dict to be stored as JSONB
                        # def insert_word_template(cur, word_id, template_name, args=None, source_identifier=None):
                        #    # ... implementation using WordTemplateModel ...
                        #    return template_db_id or None

                        if "insert_word_template" in globals() and callable(
                            globals()["insert_word_template"]
                        ):
                            template_id = insert_word_template(
                                cur,
                                word_id,
                                template_name,
                                args=template_args,  # Pass the dictionary directly
                                source_identifier=source_identifier,
                            )
                            if template_id:
                                local_stats[
                                    "templates"
                                ] += 1  # Use existing 'templates' counter
                            else:
                                logger.warning(
                                    f"insert_word_template returned non-ID for template '{template_name}' (Word ID: {word_id}). Args: {template_args}"
                                )
                                error_messages.append(
                                    f"Failed to store template '{template_name}'"
                                )

                        else:
                            # Log error only once if function is missing
                            if "insert_word_template_missing_logged" not in locals():
                                logger.error(
                                    f"Function 'insert_word_template' not defined. Cannot store inflection templates for word ID {word_id}."
                                )
                                error_messages.append(
                                    f"Missing function: insert_word_template"
                                )
                                insert_word_template_missing_logged = True

                    except Exception as template_e:
                        # Log error during template insertion
                        error_msg = f"Error storing inflection template '{template_name}' for word ID {word_id}: {str(template_e)}. Args: {template_args}"
                        logger.error(error_msg, exc_info=True)
                        error_messages.append(error_msg)
            # --- MODIFICATION END ---

            # 3. Process etymology
            if "etymology_text" in entry and entry["etymology_text"]:
                try:
                    # Process etymology text
                    etymology_text = entry["etymology_text"]
                    etymology_templates = entry.get("etymology_templates", [])
                    language_codes = None

                    # Extract language codes from templates
                    if etymology_templates and isinstance(etymology_templates, list):
                        languages = set()
                        for template in etymology_templates:
                            if not isinstance(template, dict) or "name" not in template:
                                continue

                            args = template.get("args", {})
                            for key, value in args.items():
                                if (
                                    isinstance(value, str)
                                    and 1 < len(value) <= 3
                                    and value.isalpha()
                                ):
                                    languages.add(value.lower())
                                elif (
                                    isinstance(key, str)
                                    and 1 < len(key) <= 3
                                    and key.isalpha()
                                ):
                                    languages.add(key.lower())

                        if languages:
                            language_codes = ",".join(sorted(list(languages)))

                    # Insert etymology
                    cleaned_text = clean_html(etymology_text.strip())
                    ety_id = insert_etymology(
                        cur,
                        word_id=word_id,
                        etymology_text=cleaned_text,
                        source_identifier=source_identifier,
                        language_codes=language_codes,
                    )

                    if ety_id:
                        local_stats["etymologies"] += 1

                    # Process etymology relationships
                    if etymology_templates and isinstance(etymology_templates, list):
                        for template in etymology_templates:
                            if not isinstance(template, dict) or "name" not in template:
                                continue

                            template_name = template["name"].lower()
                            args = template.get("args", {})

                            # --- Improvement 1: Handle blend etymology ---
                            if template_name == "blend":
                                # Extract components (usually args 2 and 3, sometimes more)
                                components = []
                                for i in range(2, 6):  # Check args 2, 3, 4, 5
                                    comp = args.get(str(i))
                                    if comp and isinstance(comp, str):
                                        components.append(comp)

                                if (
                                    not components
                                ):  # Fallback to positional args if numbered ones fail
                                    if isinstance(args, dict) and all(
                                        isinstance(k, str) and k.isdigit()
                                        for k in args.keys()
                                    ):
                                        positional_args = [
                                            v
                                            for k, v in sorted(args.items())
                                            if k != "1"
                                        ]  # Skip language arg "1" if present
                                        components = [
                                            arg
                                            for arg in positional_args
                                            if isinstance(arg, str)
                                        ]

                                for component_word in components:
                                    component_word_clean = clean_html(component_word)
                                    # --- BEGIN TARGETED CHANGE (Store Metadata for Non-Words) ---
                                    if component_word_clean.lower() in NON_WORD_STRINGS:
                                        try:
                                            note_type = "Descriptor"  # Default for blend components
                                            # Basic check if it looks like a language code
                                            if (
                                                len(component_word_clean) == 2
                                                or len(component_word_clean) == 3
                                            ) and component_word_clean.lower().isalpha():
                                                note_type = "Language Code"
                                            # Add other specific type checks if needed for blend components (e.g., Usage, Place Name)
                                            elif component_word_clean.lower() in [
                                                "obsolete",
                                                "archaic",
                                                "dialectal",
                                                "uncommon",
                                                "inclusive",
                                                "regional",
                                                "formal",
                                                "informal",
                                                "slang",
                                                "colloquial",
                                                "figurative",
                                                "literal",
                                                "standard",
                                                "nonstandard",
                                                "proscribed",
                                                "dated",
                                                "rare",
                                                "poetic",
                                                "historical",
                                            ]:
                                                note_type = "Usage"
                                            elif component_word_clean in [
                                                "Batangas",
                                                "Marinduque",
                                                "Rizal",
                                            ]:
                                                note_type = "Place Name"

                                            cur.execute(
                                                """
                                                UPDATE words
                                                SET word_metadata = jsonb_set(
                                                    COALESCE(word_metadata, '{}'::jsonb),
                                                    '{linguistic_notes}',
                                                    COALESCE(word_metadata->'linguistic_notes', '[]'::jsonb) ||
                                                    jsonb_build_object('type', %s, 'value', %s, 'source', %s)::jsonb
                                                )
                                                WHERE id = %s
                                            """,
                                                (
                                                    note_type,
                                                    component_word_clean,
                                                    f"etymology_{template_name}",
                                                    word_id,
                                                ),
                                            )
                                            logger.debug(
                                                f"Stored non-word component '{component_word_clean}' (type: {note_type}) from blend template for '{word}' into metadata."
                                            )
                                        except Exception as meta_e:
                                            error_msg = f"Error storing non-word metadata for component '{component_word_clean}' in blend template for word '{word}': {meta_e}"
                                            logger.error(error_msg)
                                            error_messages.append(error_msg)
                                        continue  # Skip relation creation for this non-word component
                                    # --- END TARGETED CHANGE (Store Metadata for Non-Words) ---

                                    if (
                                        component_word_clean
                                        and component_word_clean.lower() != word.lower()
                                    ):
                                        # Blends are usually within the same language
                                        component_lang = language_code
                                        component_cache_key = f"{component_word_clean.lower()}|{component_lang}"

                                        if component_cache_key in word_cache:
                                            related_word_id = word_cache[
                                                component_cache_key
                                            ]
                                        else:
                                            related_word_id = get_or_create_word_id(
                                                cur,
                                                component_word_clean,
                                                component_lang,
                                                source_identifier,
                                            )
                                            if related_word_id:
                                                word_cache[component_cache_key] = (
                                                    related_word_id
                                                )

                                        if (
                                            related_word_id
                                            and related_word_id != word_id
                                        ):
                                            metadata = {
                                                "from_etymology": True,
                                                "template": template_name,
                                                "relation_origin": "blend",  # Indicate blend source
                                                "confidence": RelationshipType.RELATED.strength,
                                            }
                                            # Insert relation (blend component -> word) - Using RELATED as type
                                            # RELATED is bidirectional, so one insert handles both directions implicitly if insert_relation handles it,
                                            # otherwise, the second call is needed if insert_relation doesn't automatically add inverse for bidirectional.
                                            # Assuming insert_relation handles bidirectionality correctly based on RelationshipType enum.
                                            rel_id = insert_relation(
                                                cur,
                                                related_word_id,
                                                word_id,
                                                RelationshipType.RELATED.value,  # Use RELATED.value for components
                                                source_identifier,
                                                metadata,
                                            )
                                            if rel_id:
                                                local_stats["ety_relations"] += 1
                                                local_stats["relations"] += 1

                            elif (
                                template_name == "inh"
                            ):  # Inherited from proto-language
                                proto_lang = args.get("2", "")
                                proto_word = args.get("3", "")
                                if (
                                    proto_lang
                                    and proto_word
                                    and isinstance(proto_word, str)
                                ):
                                    proto_word_clean = clean_html(proto_word).lstrip(
                                        "*"
                                    )  # Clean and remove leading asterisk
                                    # --- BEGIN NON-WORD CHECK ---
                                    if proto_word_clean.lower() in NON_WORD_STRINGS:
                                        try:
                                            note_type = get_non_word_note_type(
                                                proto_word_clean
                                            )  # Assumes helper function exists
                                            add_linguistic_note(
                                                cur,
                                                word_id,
                                                note_type,
                                                proto_word_clean,
                                                f"etymology_{template_name}_{proto_lang}",
                                            )  # Assumes helper function exists
                                            logger.debug(
                                                f"Stored non-word proto-word '{proto_word_clean}' (type: {note_type}) from inh template for '{word}' into metadata."
                                            )
                                        except Exception as meta_e:
                                            error_messages.append(
                                                f"Error storing non-word metadata for proto-word '{proto_word_clean}' in inh template: {meta_e}"
                                            )
                                        continue  # Skip relation creation for this non-word
                                    # --- END NON-WORD CHECK ---
                                    if (
                                        proto_word_clean
                                        and proto_word_clean.lower() != word.lower()
                                    ):
                                        # Get ID for the proto-word
                                        proto_cache_key = (
                                            f"{proto_word_clean.lower()}|{proto_lang}"
                                        )
                                        if proto_cache_key in word_cache:
                                            related_word_id = word_cache[
                                                proto_cache_key
                                            ]
                                        else:
                                            # Note: Might need to adjust get_or_create_word_id if it needs special handling
                                            # for proto-languages (e.g., 'poz-pro') or proto-words ('*giliŋ')
                                            related_word_id = get_or_create_word_id(
                                                cur,
                                                proto_word_clean,
                                                proto_lang,
                                                source_identifier,
                                                # , is_proto_word=True # Add flag if needed by get_or_create_word_id
                                            )
                                            if related_word_id:
                                                word_cache[proto_cache_key] = (
                                                    related_word_id
                                                )

                                        if (
                                            related_word_id
                                            and related_word_id != word_id
                                        ):
                                            metadata = {
                                                "from_etymology": True,
                                                "template": template_name,
                                                "confidence": RelationshipType.DERIVED_FROM.strength,
                                            }
                                            # Insert relation: current word DERIVED_FROM proto-word
                                            rel_id = insert_relation(
                                                cur,
                                                word_id,
                                                related_word_id,
                                                RelationshipType.DERIVED_FROM.value,
                                                source_identifier,
                                                metadata,
                                            )
                                            if rel_id:
                                                local_stats["ety_relations"] += 1
                                                local_stats["relations"] += 1
                                            # Add inverse: proto-word ROOT_OF current word
                                            insert_relation(
                                                cur,
                                                related_word_id,
                                                word_id,
                                                RelationshipType.ROOT_OF.value,
                                                source_identifier,
                                                metadata,
                                            )

                            elif (
                                template_name == "cog"
                            ):  # Cognate with word in another language
                                cognate_lang = args.get("1", "")
                                cognate_word = args.get("2", "")
                                if (
                                    cognate_lang
                                    and cognate_word
                                    and isinstance(cognate_word, str)
                                ):
                                    cognate_word_clean = clean_html(cognate_word)

                                    # --- BEGIN NON-WORD CHECK ---
                                    if cognate_word_clean.lower() in NON_WORD_STRINGS:
                                        try:
                                            note_type = get_non_word_note_type(
                                                cognate_word_clean
                                            )  # Assumes helper function exists
                                            add_linguistic_note(
                                                cur,
                                                word_id,
                                                note_type,
                                                cognate_word_clean,
                                                f"etymology_{template_name}_{cognate_lang}",
                                            )  # Assumes helper function exists
                                            logger.debug(
                                                f"Stored non-word cognate '{cognate_word_clean}' (type: {note_type}) from cog template for '{word}' into metadata."
                                            )
                                        except Exception as meta_e:
                                            error_messages.append(
                                                f"Error storing non-word metadata for cognate '{cognate_word_clean}' in cog template: {meta_e}"
                                            )
                                        continue  # Skip relation creation for this non-word
                                    # --- END NON-WORD CHECK ---
                                    if (
                                        cognate_word_clean
                                        and cognate_word_clean.lower() != word.lower()
                                    ):
                                        # Get ID for the cognate word
                                        cognate_cache_key = f"{cognate_word_clean.lower()}|{cognate_lang}"
                                        if cognate_cache_key in word_cache:
                                            related_word_id = word_cache[
                                                cognate_cache_key
                                            ]
                                        else:
                                            related_word_id = get_or_create_word_id(
                                                cur,
                                                cognate_word_clean,
                                                cognate_lang,
                                                source_identifier,
                                            )
                                            if related_word_id:
                                                word_cache[cognate_cache_key] = (
                                                    related_word_id
                                                )

                                        if (
                                            related_word_id
                                            and related_word_id != word_id
                                        ):
                                            metadata = {
                                                "from_etymology": True,
                                                "template": template_name,
                                                "confidence": RelationshipType.COGNATE_OF.strength,
                                            }
                                            # --- NEW: Extract Gloss ---
                                            target_gloss = args.get("4") or args.get(
                                                "t"
                                            )  # Check arg 4 or 't'
                                            if (
                                                target_gloss
                                                and isinstance(target_gloss, str)
                                                and target_gloss.strip()
                                            ):
                                                metadata["target_gloss"] = (
                                                    target_gloss.strip()
                                                )
                                            # --- END: Extract Gloss ---
                                            # Insert relation: current word COGNATE_OF related_word
                                            # COGNATE_OF is bidirectional, assume insert_relation handles the inverse.
                                            rel_id = insert_relation(
                                                cur,
                                                word_id,
                                                related_word_id,
                                                RelationshipType.COGNATE_OF.value,
                                                source_identifier,
                                                metadata,
                                            )
                                            # Explicitly add inverse since insert_relation doesn't do it for bidirectional types
                                            insert_relation(
                                                cur,
                                                related_word_id,
                                                word_id,
                                                RelationshipType.COGNATE_OF.value,
                                                source_identifier,
                                                metadata,
                                            )
                                            if rel_id:
                                                local_stats["ety_relations"] += 1
                                                local_stats["relations"] += 1

                            elif (
                                template_name == "doublet"
                            ):  # Doublet within the same language
                                doublet_lang = (
                                    args.get("1", "") or language_code
                                )  # Should be same language
                                doublet_word = args.get("2", "")
                                if doublet_word and isinstance(doublet_word, str):
                                    doublet_word_clean = clean_html(doublet_word)
                                    # --- BEGIN NON-WORD CHECK ---
                                    if doublet_word_clean.lower() in NON_WORD_STRINGS:
                                        try:
                                            note_type = get_non_word_note_type(
                                                doublet_word_clean
                                            )  # Assumes helper function exists
                                            add_linguistic_note(
                                                cur,
                                                word_id,
                                                note_type,
                                                doublet_word_clean,
                                                f"etymology_{template_name}",
                                            )  # Assumes helper function exists
                                            logger.debug(
                                                f"Stored non-word doublet '{doublet_word_clean}' (type: {note_type}) from doublet template for '{word}' into metadata."
                                            )
                                        except Exception as meta_e:
                                            error_messages.append(
                                                f"Error storing non-word metadata for doublet '{doublet_word_clean}' in doublet template: {meta_e}"
                                            )
                                        continue  # Skip relation creation for this non-word
                                    # --- END NON-WORD CHECK ---

                                    if (
                                        doublet_word_clean
                                        and doublet_word_clean.lower() != word.lower()
                                    ):
                                        # Get ID for the doublet word
                                        doublet_cache_key = f"{doublet_word_clean.lower()}|{doublet_lang}"
                                        if doublet_cache_key in word_cache:
                                            related_word_id = word_cache[
                                                doublet_cache_key
                                            ]
                                        else:
                                            related_word_id = get_or_create_word_id(
                                                cur,
                                                doublet_word_clean,
                                                doublet_lang,
                                                source_identifier,
                                            )
                                            if related_word_id:
                                                word_cache[doublet_cache_key] = (
                                                    related_word_id
                                                )

                                        if (
                                            related_word_id
                                            and related_word_id != word_id
                                        ):
                                            metadata = {
                                                "from_etymology": True,
                                                "template": template_name,
                                                "confidence": RelationshipType.DOUBLET_OF.strength,
                                            }
                                            # --- NEW: Extract Gloss (Doublet templates usually don't have glosses, but check just in case) ---
                                            target_gloss = args.get(
                                                "t"
                                            )  # Less common for doublets, but check 't'
                                            if (
                                                target_gloss
                                                and isinstance(target_gloss, str)
                                                and target_gloss.strip()
                                            ):
                                                metadata["target_gloss"] = (
                                                    target_gloss.strip()
                                                )
                                            # --- END: Extract Gloss ---
                                            # Insert relation: current word DOUBLET_OF related_word
                                            # DOUBLET_OF is bidirectional, assume insert_relation handles the inverse.
                                            rel_id = insert_relation(
                                                cur,
                                                word_id,
                                                related_word_id,
                                                RelationshipType.DOUBLET_OF.value,
                                                source_identifier,
                                                metadata,
                                            )
                                            # Explicitly add inverse since insert_relation doesn't do it for bidirectional types
                                            insert_relation(
                                                cur,
                                                related_word_id,
                                                word_id,
                                                RelationshipType.DOUBLET_OF.value,
                                                source_identifier,
                                                metadata,
                                            )
                                            if rel_id:
                                                local_stats["ety_relations"] += 1
                                                local_stats["relations"] += 1
                            # --- End additions ---

                            if template_name in [
                                "derived",
                                "borrowing",
                                "derived from",
                                "borrowed from",
                                "bor",
                                "der",
                                "bor+",
                            ]:
                                rel_type = RelationshipType.DERIVED_FROM
                                # Adjusted logic for source_lang and source_word extraction
                                source_lang = ""
                                source_word = ""
                                if template_name in [
                                    "bor",
                                    "bor+",
                                    "borrowing",
                                    "borrowed from",
                                ]:  # Borrowing templates
                                    source_lang = args.get(
                                        "2", ""
                                    )  # Language code is usually arg 2
                                    source_word = args.get(
                                        "3", ""
                                    )  # Source word is usually arg 3
                                    if (
                                        not source_lang
                                    ):  # Fallback if arg 2 is missing/empty
                                        source_lang = args.get("1", "")
                                elif template_name in [
                                    "der",
                                    "derived",
                                    "derived from",
                                ]:  # Derivation templates
                                    source_lang = args.get(
                                        "1", ""
                                    )  # Language code is usually arg 1
                                    source_word = args.get(
                                        "2", ""
                                    )  # Source word is usually arg 2

                                # Ensure source_lang has a reasonable default if empty
                                if not source_lang:
                                    source_lang = language_code  # Default to the entry's language if not specified

                                if source_word and isinstance(source_word, str):
                                    source_word_clean = clean_html(source_word)

                                    # --- BEGIN NON-WORD CHECK ---
                                    if source_word_clean.lower() in NON_WORD_STRINGS:
                                        try:
                                            note_type = get_non_word_note_type(
                                                source_word_clean
                                            )  # Assumes helper function exists
                                            # Use the specific source_lang if available, otherwise default
                                            note_source_lang = (
                                                source_lang
                                                if source_lang
                                                else "unknown"
                                            )
                                            add_linguistic_note(
                                                cur,
                                                word_id,
                                                note_type,
                                                source_word_clean,
                                                f"etymology_{template_name}_{note_source_lang}",
                                            )  # Assumes helper function exists
                                            logger.debug(
                                                f"Stored non-word source '{source_word_clean}' (type: {note_type}, lang: {note_source_lang}) from {template_name} template for '{word}' into metadata."
                                            )
                                        except Exception as meta_e:
                                            error_messages.append(
                                                f"Error storing non-word metadata for source '{source_word_clean}' in {template_name} template: {meta_e}"
                                            )
                                        continue  # Skip relation creation for this non-word source
                                    # --- END NON-WORD CHECK ---

                                    # Skip if the cleaned source word is empty or just a placeholder like '-'
                                    if (
                                        source_word_clean
                                        and source_word_clean != "-"
                                        and source_word_clean.lower() != word.lower()
                                    ):
                                        # Use extracted source_lang, fall back to entry lang if needed
                                        source_word_lang = (
                                            source_lang
                                            if source_lang
                                            else language_code
                                        )

                                        # Check cache for source word
                                        source_cache_key = f"{source_word_clean.lower()}|{source_word_lang}"
                                        if source_cache_key in word_cache:
                                            related_word_id = word_cache[
                                                source_cache_key
                                            ]
                                        else:
                                            related_word_id = get_or_create_word_id(
                                                cur,
                                                source_word_clean,
                                                source_word_lang,
                                                source_identifier,
                                            )
                                            if related_word_id:
                                                word_cache[source_cache_key] = (
                                                    related_word_id
                                                )

                                        if (
                                            related_word_id
                                            and related_word_id != word_id
                                        ):
                                            metadata = {
                                                "from_etymology": True,
                                                "template": template_name,
                                                "confidence": rel_type.strength,
                                            }
                                            # --- NEW: Extract Gloss ---
                                            target_gloss = None
                                            # Check standard gloss args for borrowing templates
                                            if template_name in [
                                                "bor",
                                                "bor+",
                                                "borrowing",
                                                "borrowed from",
                                            ]:
                                                target_gloss = args.get(
                                                    "t"
                                                ) or args.get(
                                                    "5"
                                                )  # Common args 't' or '5'
                                            # Add checks for derivation templates if they sometimes carry glosses
                                            # elif template_name in ["der", "derived", "derived from"]:
                                            #    target_gloss = args.get("t") # Or check appropriate arg number like '4' if applicable

                                            if (
                                                target_gloss
                                                and isinstance(target_gloss, str)
                                                and target_gloss.strip()
                                            ):
                                                metadata["target_gloss"] = (
                                                    target_gloss.strip()
                                                )
                                            # --- END: Extract Gloss ---

                                            # Insert relation
                                            rel_id = insert_relation(
                                                cur,
                                                word_id,
                                                related_word_id,
                                                rel_type.value, # Use .value here
                                                source_identifier,
                                                metadata,
                                            )

                                            if rel_id:
                                                local_stats["ety_relations"] += 1
                                                local_stats["relations"] += 1

                                            # Add inverse relation
                                            inverse_rel = rel_type.get_inverse()
                                            if inverse_rel:
                                                insert_relation(
                                                    cur,
                                                    related_word_id,
                                                    word_id,
                                                    inverse_rel.value, # Use .value here
                                                    source_identifier,
                                                    metadata,
                                                )
                except Exception as e:
                    error_messages.append(f"Etymology error: {str(e)}")

                    # --- Improvement 3: Process Top-Level Relation Fields ---

            # --- Improvement 3: Process Top-Level Relation Fields ---

            top_level_relations_map = {
                "derived": RelationshipType.ROOT_OF,  # Word is ROOT_OF the derived term
                "related": RelationshipType.RELATED,
                "synonyms": RelationshipType.SYNONYM,
                "antonyms": RelationshipType.ANTONYM,
            }

            for field, rel_type in top_level_relations_map.items():
                if field in entry and isinstance(entry[field], list):
                    for related_item in entry[field]:
                        related_word_str = None
                        item_metadata = {}  # Store metadata from the item itself

                        if isinstance(related_item, str):
                            related_word_str = related_item
                        elif (
                            isinstance(related_item, dict)
                            and "word" in related_item
                            and isinstance(related_item["word"], str)
                        ):
                            related_word_str = related_item["word"]
                            # --- BEGIN ADDED METADATA EXTRACTION ---
                            if "tags" in related_item and isinstance(
                                related_item["tags"], list
                            ):
                                item_metadata["tags"] = related_item["tags"]
                            if "qualifier" in related_item and isinstance(
                                related_item["qualifier"], str
                            ):
                                item_metadata["qualifier"] = related_item["qualifier"]
                            if "english" in related_item and isinstance(
                                related_item["english"], str
                            ):
                                # Often used for dialect/region info in Kaikki synonyms/related
                                item_metadata["note"] = related_item["english"]
                            if "topics" in related_item and isinstance(
                                related_item["topics"], list
                            ):
                                item_metadata["topics"] = related_item["topics"]
                            # Add extraction for other potential fields if needed
                            # --- END ADDED METADATA EXTRACTION ---

                        if related_word_str:
                            related_word_clean = clean_html(related_word_str)
                            # Check for non-words (existing logic)
                            if (
                                related_word_clean.lower()
                                == "spanish-based orthography"
                            ):
                                logger.debug(
                                    f"Found non-word string '{related_word_clean}' in top-level field '{field}' for word '{word}'"
                                )
                                try:
                                    note_type = "Orthography"
                                    # Use the more robust jsonb_set update logic
                                    cur.execute(
                                        """
                                        UPDATE words
                                        SET word_metadata = jsonb_set(
                                            COALESCE(word_metadata, '{}'::jsonb),
                                            '{linguistic_notes}',
                                            COALESCE(word_metadata->'linguistic_notes', '[]'::jsonb) ||
                                            jsonb_build_object('type', %s, 'value', %s, 'source', %s)::jsonb
                                        )
                                        WHERE id = %s
                                    """,
                                        (
                                            note_type,
                                            related_word_clean,
                                            f"top_level_{field}",
                                            word_id,
                                        ),
                                    )

                                    logger.info(
                                        f"Added {note_type} '{related_word_clean}' from top-level field '{field}' to word_metadata for '{word}' (ID: {word_id})"
                                    )
                                except Exception as meta_e:
                                    error_msg = f"Error storing metadata for non-word '{related_word_clean}' from field '{field}' for word '{word}': {meta_e}"
                                    logger.error(error_msg)
                                    error_messages.append(error_msg)
                            if (
                                related_word_clean.lower() in NON_WORD_STRINGS
                                and (
                                    len(related_word_clean) == 2
                                    or len(related_word_clean) == 3
                                )
                                and related_word_clean.lower().isalpha()
                            ):
                                try:
                                    note_type = "Language Code"
                                    # Use the more robust jsonb_set update logic
                                    cur.execute(
                                        """
                                        UPDATE words
                                        SET word_metadata = jsonb_set(
                                            COALESCE(word_metadata, '{}'::jsonb),
                                            '{linguistic_notes}',
                                            COALESCE(word_metadata->'linguistic_notes', '[]'::jsonb) ||
                                            jsonb_build_object('type', %s, 'value', %s, 'source', %s)::jsonb
                                        )
                                        WHERE id = %s
                                    """,
                                        (
                                            note_type,
                                            related_word_clean,
                                            f"top_level_{field}",
                                            word_id,
                                        ),
                                    )

                                    logger.info(
                                        f"Added {note_type} '{related_word_clean}' from top-level field '{field}' to word_metadata for '{word}' (ID: {word_id})"
                                    )
                                except Exception as meta_e:
                                    error_msg = f"Error storing metadata for non-word '{related_word_clean}' from field '{field}' for word '{word}': {meta_e}"
                                    logger.error(error_msg)
                                    error_messages.append(error_msg)

                                # Skip creating relationship for this non-word
                                continue
                                # --- End non-word handling ---

                            if (
                                related_word_clean
                                and related_word_clean.lower() != word.lower()
                            ):
                                # Assume related words are in the same language unless specified otherwise
                                related_lang = language_code
                                related_cache_key = (
                                    f"{related_word_clean.lower()}|{related_lang}"
                                )

                                if related_cache_key in word_cache:
                                    related_word_id = word_cache[related_cache_key]
                                else:
                                    related_word_id = get_or_create_word_id(
                                        cur,
                                        related_word_clean,
                                        related_lang,
                                        source_identifier,
                                    )
                                    if related_word_id:
                                        word_cache[related_cache_key] = related_word_id

                                if related_word_id and related_word_id != word_id:
                                    # --- MODIFIED METADATA MERGE ---
                                    relation_metadata = {
                                        "source": f"top_level_{field}",
                                        "confidence": rel_type.strength,
                                    }
                                    # Merge metadata extracted from the item itself
                                    relation_metadata.update(item_metadata)
                                    # --- END MODIFIED METADATA MERGE ---

                                    # Insert relation
                                    rel_id = insert_relation(
                                        cur,
                                        word_id,
                                        related_word_id,
                                        rel_type.value, # Use .value here
                                        source_identifier,
                                        relation_metadata,  # Use updated metadata
                                    )
                                    if rel_id:
                                        local_stats[
                                            "relations"
                                        ] += 1  # Increment general relations count
                                        # Approximate contribution to specific relation types for stats
                                        if field in ["synonyms", "antonyms", "related"]:
                                            local_stats[
                                                "sense_relations"
                                            ] += 1  # Approximate as sense-level
                                        elif field == "derived":
                                            local_stats[
                                                "ety_relations"
                                            ] += 1  # Approximate as ety-level

                                    # Add inverse relation if applicable
                                    # Assuming insert_relation handles bidirectionality based on rel_type
                                    # If not, explicit inverse insertion might be needed for non-bidirectional types.
                                    # For safety, let's assume explicit inverse is needed for non-bidirectional
                                    if not rel_type.bidirectional:
                                        inverse_rel_enum = rel_type.inverse # <-- Use .inverse
                                        if inverse_rel_enum:
                                            insert_relation(
                                                cur,
                                                related_word_id,
                                                word_id,
                                                inverse_rel_enum.value, # <-- Use .value of the enum
                                                source_identifier,
                                                relation_metadata,
                                            )
                                    # If relation IS bidirectional, insert_relation should ideally handle the inverse
                                    # or we need to explicitly call it again like:
                                    elif rel_type.bidirectional:
                                        insert_relation(cur, related_word_id, word_id, rel_type.value, source_identifier, relation_metadata) # Use .value

            # --- End Improvement 3 ---

            for field, rel_type in top_level_relations_map.items():
                if field in entry and isinstance(entry[field], list):
                    for related_item in entry[field]:
                        related_word_str = None
                        if isinstance(related_item, str):
                            related_word_str = related_item
                        elif (
                            isinstance(related_item, dict)
                            and "word" in related_item
                            and isinstance(related_item["word"], str)
                        ):
                            related_word_str = related_item["word"]

                        if related_word_str:
                            related_word_clean = clean_html(related_word_str)
                            if related_word_clean.lower() in NON_WORD_STRINGS:
                                logger.debug(
                                    f"Found non-word string '{related_word_clean}' for word '{word}'"
                                )

                                # Store as linguistic note in the word_metadata JSONB column directly
                                try:
                                    # Determine what type of information this is
                                    note_type = "Language Code"
                                    if (
                                        len(related_word_clean) <= 3
                                        and related_word_clean.isalpha()
                                    ):
                                        note_type = "Language Code"
                                    elif (
                                        related_word_clean.lower()
                                        == "spanish-based orthography"
                                    ):
                                        note_type = "Orthography"
                                    elif related_word_clean.lower() in [
                                        "obsolete",
                                        "archaic",
                                        "dialectal",
                                    ]:
                                        note_type = "Usage"

                                    # Simple SQL to update the word's metadata directly
                                    # This adds to the existing metadata without replacing it
                                    cur.execute(
                                        """
                                        UPDATE words
                                        SET word_metadata = CASE
                                            WHEN word_metadata IS NULL THEN 
                                                jsonb_build_object('linguistic_notes', jsonb_build_array(jsonb_build_object('type', %s, 'value', %s, 'source', %s)))
                                            WHEN word_metadata ? 'linguistic_notes' THEN
                                                jsonb_set(
                                                    word_metadata,
                                                    '{linguistic_notes}',
                                                    word_metadata->'linguistic_notes' || 
                                                    jsonb_build_array(jsonb_build_object('type', %s, 'value', %s, 'source', %s))
                                                )
                                            ELSE
                                                word_metadata || 
                                                jsonb_build_object('linguistic_notes', jsonb_build_array(jsonb_build_object('type', %s, 'value', %s, 'source', %s)))
                                        END
                                        WHERE id = %s
                                    """,
                                        (
                                            note_type,
                                            related_word_clean,
                                            field,
                                            note_type,
                                            related_word_clean,
                                            field,
                                            note_type,
                                            related_word_clean,
                                            field,
                                            word_id,
                                        ),
                                    )

                                    logger.info(
                                        f"Added {note_type} '{related_word_clean}' to word_metadata for '{word}'"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error storing metadata for '{related_word_clean}': {e}"
                                    )

                                # Skip creating relationship for this non-word
                                continue
                            if (
                                related_word_clean
                                and related_word_clean.lower() != word.lower()
                            ):
                                # Skip entries that are likely language codes or descriptive phrases
                                # Assume related words are in the same language unless specified otherwise (which Kaikki rarely does here)
                                related_lang = language_code
                                related_cache_key = (
                                    f"{related_word_clean.lower()}|{related_lang}"
                                )

                                if related_cache_key in word_cache:
                                    related_word_id = word_cache[related_cache_key]
                                else:
                                    related_word_id = get_or_create_word_id(
                                        cur,
                                        related_word_clean,
                                        related_lang,
                                        source_identifier,
                                    )
                                    if related_word_id:
                                        word_cache[related_cache_key] = related_word_id

                                if related_word_id and related_word_id != word_id:
                                    metadata = {
                                        "source": f"top_level_{field}",
                                        "confidence": rel_type.strength,
                                    }
                                    # Insert relation
                                    rel_id = insert_relation(
                                        cur,
                                        word_id,
                                        related_word_id,
                                        rel_type.value, # Use .value here
                                        source_identifier,
                                        metadata,
                                    )
                                    if rel_id:
                                        local_stats[
                                            "relations"
                                        ] += 1  # Increment general relations count
                                        # Approximate contribution to sense relations for stats
                                        if field in ["synonyms", "antonyms", "related"]:
                                            local_stats["sense_relations"] += 1
                                        elif field == "derived":
                                            local_stats["ety_relations"] += 1

                                    # Add inverse relation if applicable (insert_relation should ideally handle bidirectionality)
                                    # If insert_relation doesn't handle it automatically based on rel_type.bidirectional:
                                    if not rel_type.bidirectional:
                                        inverse_rel_enum = rel_type.inverse # <-- Use .inverse
                                        if inverse_rel_enum:
                                            insert_relation(
                                                cur,
                                                related_word_id,
                                                word_id,
                                                inverse_rel_enum.value, # <-- Use .value of the enum
                                                source_identifier,
                                                relation_metadata,
                                            )
                                    # If insert_relation DOES handle bidirectionality, the inverse logic is not needed here.
                                    # Let's assume insert_relation needs explicit inverse insertion for non-bidirectional types for safety.

            # --- End Improvement 3 ---
            # 4. Process templates
            if "head_templates" in entry and entry["head_templates"]:
                try:
                    for template in entry["head_templates"]:
                        if not isinstance(template, dict) or "name" not in template:
                            continue

                        template_name = template["name"]
                        args = template.get("args")
                        expansion = template.get("expansion")

                        cur.execute(
                            """
                            INSERT INTO word_templates (word_id, template_name, args, expansion) 
                            VALUES (%s, %s, %s, %s) 
                            ON CONFLICT (word_id, template_name) DO UPDATE 
                            SET args = EXCLUDED.args, 
                                expansion = EXCLUDED.expansion, 
                                updated_at = CURRENT_TIMESTAMP
                            """,
                            (
                                word_id,
                                template_name,
                                Json(args) if args else None,
                                expansion,
                            ),
                        )

                        if cur.rowcount > 0:
                            local_stats["templates"] += 1
                except Exception as e:
                    error_messages.append(f"Template error: {str(e)}")

            # 5. Process senses/definitions
            # 3. Process senses (definitions) and their nested data
            if "senses" in entry and isinstance(entry["senses"], list):
                # Track if we have successfully processed at least one sense
                processed_sense_count = 0
                
                for sense_idx, sense in enumerate(entry["senses"]):
                    if not isinstance(sense, dict):
                        error_messages.append(f"Non-dictionary sense at index {sense_idx}. Skipping.")
                        continue

                    # Get the definition text - prioritize glosses for consistent structure
                    definition_text = None
                    sense_id = sense.get("id", "")  # Capture sense ID for tracking
                    
                    # First, try to extract from glosses (preferred)
                    if "glosses" in sense and isinstance(sense["glosses"], list) and sense["glosses"]:
                        # Join all glosses with a semicolon
                        glosses = [g for g in sense["glosses"] if g and isinstance(g, str)]
                        if glosses:
                            definition_text = "; ".join(clean_html(g) for g in glosses)
                        
                    # If no glosses, fall back to 'raw_glosses' or 'definition'
                    if not definition_text and "raw_glosses" in sense and isinstance(sense["raw_glosses"], list) and sense["raw_glosses"]:
                        raw_glosses = [g for g in sense["raw_glosses"] if g and isinstance(g, str)]
                        if raw_glosses:
                            definition_text = "; ".join(clean_html(g) for g in raw_glosses)
                    
                    if not definition_text:
                        # Try definition field as final fallback
                        definition = sense.get("definition")
                        if definition and isinstance(definition, str):
                            definition_text = clean_html(definition)
                        else:
                            # Skip sense items with no valid definition text
                            error_messages.append(f"Sense at index {sense_idx} has no usable definition (Word: '{word}'). Skipping.")
                            continue
                        
                    # Get POS information - inherit from top-level or use sense-specific if available
                    original_pos = sense.get("pos", pos)  # Use sense pos if available, otherwise top-level pos

                    # Create metadata to store additional sense information
                    metadata = {}
                    
                    # Store sense ID if available
                    if sense_id:
                        metadata["sense_id"] = sense_id
                    
                    # Track sense index
                    metadata["sense_index"] = sense_idx
                    
                    # Detect sense type
                    if "alt_of" in sense:
                        metadata["sense_type"] = "alternative_form"
                        
                        # Capture reference info
                        if isinstance(sense["alt_of"], list) and sense["alt_of"]:
                            alt_of_data = sense["alt_of"][0]
                            if isinstance(alt_of_data, str):
                                metadata["alt_of"] = alt_of_data
                            elif isinstance(alt_of_data, dict) and "word" in alt_of_data:
                                metadata["alt_of"] = alt_of_data["word"]
                        elif isinstance(sense["alt_of"], str):
                            metadata["alt_of"] = sense["alt_of"]
                        elif isinstance(sense["alt_of"], dict) and "word" in sense["alt_of"]:
                            metadata["alt_of"] = sense["alt_of"]["word"]
                    
                    elif "form_of" in sense:
                        metadata["sense_type"] = "variant_form"
                        
                        # Capture reference info
                        if isinstance(sense["form_of"], list) and sense["form_of"]:
                            form_of_data = sense["form_of"][0]
                            if isinstance(form_of_data, str):
                                metadata["form_of"] = form_of_data
                            elif isinstance(form_of_data, dict) and "word" in form_of_data:
                                metadata["form_of"] = form_of_data["word"]
                        elif isinstance(sense["form_of"], str):
                            metadata["form_of"] = sense["form_of"]
                        elif isinstance(sense["form_of"], dict) and "word" in sense["form_of"]:
                            metadata["form_of"] = sense["form_of"]["word"]
                    
                    # Store other relevant metadata like qualifiers, domains
                    if "qualifier" in sense and sense["qualifier"]:
                        metadata["qualifier"] = sense["qualifier"]
                        
                    if "topics" in sense and isinstance(sense["topics"], list) and sense["topics"]:
                        metadata["topics"] = sense["topics"]

                    # --- MODIFICATION START: Extract tags, merge topics ---
                    tags = sense.get("tags", [])
                    # Ensure tags is initially a list
                    if not isinstance(tags, list):
                        tags = (
                            [str(tags)] if tags else []
                        )  # Convert single tag to list, handle non-list input

                    # Extract topics and merge into tags
                    topics = sense.get("topics", [])
                    if topics and isinstance(topics, list):
                        # Add topics, ensuring they are strings and non-empty
                        tags.extend(
                            [str(t).strip() for t in topics if t and isinstance(t, str)]
                        )

                    # Clean up and deduplicate the final tags list
                    cleaned_tags = []
                    seen_tags = set()
                    for tag in tags:
                        if tag and isinstance(tag, str):
                            stripped_tag = tag.strip()
                            if stripped_tag and stripped_tag not in seen_tags:
                                cleaned_tags.append(stripped_tag)
                                seen_tags.add(stripped_tag)

                    # Ensure tags is None if empty, otherwise use the cleaned list
                    final_tags = cleaned_tags if cleaned_tags else []
                    # --- MODIFICATION END ---

                    # Extract examples
                    examples = sense.get("examples")
                    # Ensure examples is a list of dictionaries before dumping
                    if examples and isinstance(examples, list):
                        valid_examples = [ex for ex in examples if isinstance(ex, dict)]
                        # Store examples in metadata for reference
                        if valid_examples:
                            metadata["example_count"] = len(valid_examples)
                        examples_json = (
                            json.dumps(valid_examples) if valid_examples else None
                        )
                    else:
                        examples_json = None

                    if examples_json:
                        local_stats["examples"] += len(
                            valid_examples
                        )  # Count examples based on valid input

                    # Extract usage notes
                    usage_notes = sense.get("usage_notes")
                    if isinstance(usage_notes, list):
                        # Filter out non-string elements and join
                        usage_notes = "\n".join(
                            str(n).strip()
                            for n in usage_notes
                            if n and isinstance(n, str)
                        )
                    elif isinstance(usage_notes, str):
                        usage_notes = usage_notes.strip()
                    else:
                        usage_notes = None  # Set to None if not list or string

                    try:
                        # Insert the definition into the database
                        def_id = insert_definition(
                            cur,
                            word_id,
                            definition_text,
                            sources=source_identifier,  # Correct argument name is 'sources'
                            part_of_speech=original_pos,  # Pass the ORIGINAL string here using the correct argument name
                            usage_notes=usage_notes if usage_notes else None,
                            tags=final_tags,
                            metadata=metadata  # Pass the collected metadata to the definition insert
                        )

                        if (
                            def_id
                        ):  # Only proceed if definition was successfully inserted/found
                            processed_sense_count += 1
                            # Increment definition count (moved here from original spot for clarity)
                            local_stats[
                                "definitions"
                            ] += 1  # Correct place to count successful def insertions

                            # --- BEGIN: Process sense-level links ---
                            if "links" in sense and isinstance(sense["links"], list):
                                for link_item_idx, link_item in enumerate(
                                    sense["links"]
                                ):
                                    try:
                                        # Skip empty/null links
                                        if not link_item:
                                            continue
                                        
                                        # --- START HANDLING LIST WITHIN LIST ---
                                        items_to_process = []
                                        if isinstance(link_item, list):
                                            # If the item itself is a list, iterate through its contents
                                            logger.warning(f"Found nested list in links at index {link_item_idx} for definition {def_id}. Processing items within.")
                                            items_to_process.extend(link_item)
                                        else:
                                            # Otherwise, process the item directly
                                            items_to_process.append(link_item)
                                        # --- END HANDLING LIST WITHIN LIST ---

                                        for item in items_to_process: # Process each item (original or from nested list)
                                            link_data = None
                                            if isinstance(item, str):
                                                link_data = {"link": item.strip(), "type": "external"}
                                            elif isinstance(item, dict):
                                                # Use the existing dictionary structure
                                                link_data = item
                                                # Ensure link text exists
                                                link_text = link_data.get("link") or link_data.get("text")
                                                if not link_text:
                                                    # Try numbered keys if text/link missing
                                                    for i in range(1, 5): # Check keys '1' to '4'
                                                        if str(i) in link_data and isinstance(link_data[str(i)], str):
                                                            link_text = link_data[str(i)]
                                                            break
                                                if link_text:
                                                     link_data["link"] = link_text # Ensure 'link' key holds the text
                                                else:
                                                    logger.warning(f"Link dictionary missing usable text/link field for def ID {def_id}: {link_data}")
                                                    continue # Skip if no usable text found
                                            else:
                                                logger.warning(f"Unexpected link format for definition {def_id}: {type(item)}")
                                                continue
                                            
                                            if not link_data or not link_data.get("link"):
                                                 logger.debug(f"Skipping empty or invalid link data for def ID {def_id}: {link_data}")
                                                 continue
                                                 
                                            # Pass the link_item (string or dict) directly to the insert function
                                            link_db_id = insert_definition_link(
                                                cur,
                                                def_id,
                                                link_data,  # Pass the normalized link data
                                                source_identifier=source_identifier,
                                            )
                                            if link_db_id:
                                                local_stats["links"] += 1
                                                
                                    except Exception as link_e:
                                        # Log error but continue processing other links/categories
                                        error_msg = f"Error storing sense link #{link_item_idx} for def ID {def_id} (Word: '{word}'): {str(link_e)}. Data: {link_item}"
                                        logger.error(error_msg, exc_info=True)
                                        error_messages.append(error_msg)
                            # --- END: Process sense-level links ---

                            # --- BEGIN: Process sense-level categories ---
                            if "categories" in sense and isinstance(
                                sense["categories"], list
                            ):
                                for category_item_idx, category_item in enumerate(
                                    sense["categories"]
                                ):
                                    try:
                                        # Skip empty/null categories
                                        if not category_item:
                                            continue
                                            
                                        # Handle both string categories and object categories properly  
                                        category_data = None
                                        if isinstance(category_item, str):
                                            category_data = {"name": category_item.strip()}
                                        elif isinstance(category_item, dict):
                                            # Use the existing dictionary structure
                                            category_data = category_item
                                            # Ensure category name exists
                                            if not category_data.get("name") and category_data.get("text"):
                                                category_data["name"] = category_data["text"]
                                        else:
                                            logger.warning(f"Unexpected category format at index {category_item_idx} for definition {def_id}: {type(category_item)}")
                                            continue
                                            
                                        if not category_data or not category_data.get("name"):
                                            continue
                                        # Pass the category_item (string or dict) directly to the insert function
                                        cat_db_id = insert_definition_category(
                                            cur,
                                            def_id,
                                            category_data,  # Pass the normalized category data
                                            source_identifier=source_identifier,
                                        )
                                        if cat_db_id:
                                            local_stats["categories"] += 1
                                    except Exception as cat_e:
                                        # Log error but continue processing other links/categories
                                        error_msg = f"Error storing sense category #{category_item_idx} for def ID {def_id} (Word: '{word}'): {str(cat_e)}. Data: {category_item}"
                                        logger.error(error_msg, exc_info=True)
                                        error_messages.append(error_msg)
                            # --- END: Process sense-level categories ---

                            # --- Process examples (Keep existing logic) ---
                            if "examples" in sense and isinstance(
                                sense["examples"], list
                            ):
                                for example_data in sense["examples"]:
                                    # ... (Your existing example processing logic) ...
                                    # Ensure insert_definition_example is checked for existence and called correctly
                                    if isinstance(
                                        example_data, dict
                                    ) and example_data.get("text"):
                                        try:
                                            if (
                                                "insert_definition_example" in globals()
                                                and callable(
                                                    globals()[
                                                        "insert_definition_example"
                                                    ]
                                                )
                                            ):
                                                ex_id = insert_definition_example(
                                                    cur,
                                                    def_id,
                                                    example_data,
                                                    source_identifier,
                                                )
                                                if ex_id:
                                                    local_stats["examples"] += 1
                                            else:  # Function missing case
                                                if (
                                                    "insert_definition_example_missing_logged"
                                                    not in locals()
                                                ):
                                                    logger.error(
                                                        f"Function 'insert_definition_example' not defined. Cannot store examples."
                                                    )
                                                    insert_definition_example_missing_logged = (
                                                        True  # Prevent repeated logging
                                                    )
                                        except Exception as ex_e:
                                            error_msg = f"Error storing example for def ID {def_id}: {ex_e}"
                                            logger.error(error_msg)
                                            error_messages.append(error_msg)

                            # --- Process sense-level relations (Keep existing logic) ---
                            sense_relations_map = {
                                "synonyms": RelationshipType.SYNONYM,
                                "antonyms": RelationshipType.ANTONYM,
                                "hypernyms": RelationshipType.HYPERNYM,
                                "hyponyms": RelationshipType.HYPONYM,
                                "meronyms": RelationshipType.MERONYM,
                                "holonyms": RelationshipType.HOLONYM,
                                "related": RelationshipType.RELATED,
                            }
                            for field, rel_type in sense_relations_map.items():
                                if field in sense and isinstance(sense[field], list):
                                    for related_item in sense[field]:
                                        # ... (Your existing sense relation processing logic, including non-word checks) ...
                                        # Ensure insert_relation is called correctly
                                        related_word_str = None
                                        item_metadata = {}
                                        # ... (extract related_word_str and item_metadata) ...
                                        if related_word_str and isinstance(
                                            related_word_str, str
                                        ):
                                            related_word_clean = clean_html(
                                                related_word_str
                                            )
                                            # Non-word check
                                            if (
                                                related_word_clean.lower()
                                                in NON_WORD_STRINGS
                                            ):
                                                try:
                                                    note_type = get_non_word_note_type(
                                                        related_word_clean
                                                    )
                                                    add_linguistic_note(
                                                        cur,
                                                        word_id,
                                                        note_type,
                                                        related_word_clean,
                                                        f"sense_{field}",
                                                    )
                                                except Exception as meta_e:
                                                    error_messages.append(
                                                        f"MetaErr sense rel: {meta_e}"
                                                    )
                                                continue  # Skip non-word

                                            if (
                                                related_word_clean
                                                and related_word_clean.lower()
                                                != word.lower()
                                            ):
                                                # ... (get related_word_id using cache/get_or_create_word_id) ...
                                                related_word_id = word_cache.get(
                                                    f"{related_word_clean.lower()}|{language_code}"
                                                )  # Example cache lookup
                                                if not related_word_id:
                                                    related_word_id = (
                                                        get_or_create_word_id(
                                                            cur,
                                                            related_word_clean,
                                                            language_code,
                                                            source_identifier,
                                                        )
                                                    )
                                                    if related_word_id:
                                                        word_cache[
                                                            f"{related_word_clean.lower()}|{language_code}"
                                                        ] = related_word_id

                                                if (
                                                    related_word_id
                                                    and related_word_id != word_id
                                                ):
                                                    # Build metadata, including def_id
                                                    relation_metadata = {
                                                        "source": f"sense_{field}",
                                                        "definition_id": def_id,
                                                        "confidence": rel_type.strength,
                                                    }
                                                    relation_metadata.update(
                                                        item_metadata
                                                    )
                                                    # Insert relation
                                                    try:
                                                        rel_id = insert_relation(
                                                            cur,
                                                            word_id,
                                                            related_word_id,
                                                            rel_type.value, # Use .value here
                                                            source_identifier,
                                                            relation_metadata,
                                                        )
                                                        if rel_id:
                                                            local_stats[
                                                                "sense_relations"
                                                            ] += 1
                                                            local_stats[
                                                                "relations"
                                                            ] += 1
                                                        # Handle inverse if needed
                                                        if not rel_type.bidirectional:
                                                            inverse_rel_enum = rel_type.inverse # <-- Use .inverse
                                                            if inverse_rel_enum:
                                                                insert_relation(
                                                                    cur,
                                                                    related_word_id,
                                                                    word_id,
                                                                    inverse_rel_enum.value, # <-- Use .value of the enum
                                                                    source_identifier,
                                                                    relation_metadata,
                                                                )
                                                        elif (
                                                            rel_type.bidirectional
                                                        ):  # Explicitly add bidirectional inverse if insert_relation doesn't
                                                            insert_relation(
                                                                cur,
                                                                related_word_id,
                                                                word_id,
                                                                rel_type.value, # Use .value here
                                                                source_identifier,
                                                                relation_metadata,
                                                            )
                                                    except Exception as rel_e:
                                                        error_msg = f"Error inserting sense relation ({field}) for def ID {def_id}: {rel_e}"
                                                        logger.error(error_msg)
                                                        error_messages.append(error_msg)

                        else:  # def_id was None
                            logger.warning(
                                f"Skipping sense details (links, categories, examples, relations) for index {sense_idx} of word '{word}' because definition ID could not be obtained."
                            )

                    except Exception as def_e:
                        # Log error during definition insertion or post-processing
                        error_msg = f"Error processing sense index {sense_idx} for word '{word}': {str(def_e)}. Def Text: '{definition_text[:100]}...'"
                        logger.error(
                            error_msg, exc_info=True
                        )  # Include traceback for debugging
                        error_messages.append(error_msg)
                        # Optionally, rollback this specific definition's changes if possible,
                        # but the main transaction handles overall rollback on failure.

            # --- End Sense Processing ---
            # 6. Process top-level relations
            try:
                top_level_rels = {
                    "derived": RelationshipType.ROOT_OF,
                    "related": RelationshipType.RELATED,
                }

                for rel_key, rel_enum in top_level_rels.items():
                    if rel_key in entry and isinstance(entry[rel_key], list):
                        for item in entry[rel_key]:
                            related_word = None

                            if isinstance(item, dict) and "word" in item:
                                related_word = item["word"]
                            elif isinstance(item, str):
                                related_word = item
                            else:
                                continue

                            related_word_clean = clean_html(related_word)
                            if (
                                not related_word_clean
                                or related_word_clean.lower() == word.lower()
                            ):
                                continue

                            # Get or create related word
                            related_cache_key = (
                                f"{related_word_clean.lower()}|{language_code}"
                            )
                            if related_cache_key in word_cache:
                                related_word_id = word_cache[related_cache_key]
                            else:
                                related_word_id = get_or_create_word_id(
                                    cur,
                                    related_word_clean,
                                    language_code,
                                    source_identifier=source_identifier,
                                )
                                if related_word_id:
                                    word_cache[related_cache_key] = related_word_id

                            if related_word_id and related_word_id != word_id:
                                metadata = {
                                    "confidence": rel_enum.strength,
                                    "from_sense": False,
                                }

                                if isinstance(item, dict) and "tags" in item:
                                    metadata["tags"] = ",".join(item["tags"])

                                # Insert relation
                                rel_id = insert_relation(
                                    cur,
                                    word_id,
                                    related_word_id,
                                    rel_enum.value,
                                    source_identifier,
                                    metadata,
                                )

                                if rel_id:
                                    local_stats["relations"] += 1

                                # Add inverse/bidirectional relation if needed
                                if rel_enum.bidirectional:
                                    insert_relation(
                                        cur,
                                        related_word_id,
                                        word_id,
                                        rel_enum.value,
                                        source_identifier,
                                        metadata,
                                    )
                                else:
                                    # Correctly access the inverse enum member
                                    inv_enum = rel_enum.inverse 
                                    if inv_enum:
                                        insert_relation(
                                            cur,
                                            related_word_id,
                                            word_id,
                                            inv_enum.value, # Use the value of the inverse enum member
                                            source_identifier,
                                            metadata,
                                        )
            except Exception as e:
                error_messages.append(f"Top-level relation error: {str(e)}")

            # Update global statistics
            for key, value in local_stats.items():
                entry_stats[key] += value

            # Log warning if no senses were processed for an entry that had senses
            if len(entry["senses"]) > 0 and processed_sense_count == 0:
                logger.warning(f"Failed to process any senses for entry '{word}'")
                error_messages.append(f"No senses processed for '{word}'")

            return {
                "word_id": word_id,
                "stats": local_stats,
                "success": True
            }, "; ".join(error_messages) if error_messages else None

        except Exception as e:
            logger.error(f"Unexpected error processing entry for '{word}': {str(e)}")
            # Ensure word_id is None if the core processing failed before ID acquisition
            current_word_id = (
                word_id if "word_id" in locals() and word_id is not None else None
            )
            # Return the standard dictionary structure indicating failure
            return {
                "word_id": current_word_id,
                "stats": {},
                "success": False,
            }, f"Unhandled exception: {str(e)}"

    # Main processing logic
    stats = {
        "total_entries": total_lines if total_lines >= 0 else 0,
        "processed_ok": 0,
        "processed_with_errors": 0,
        "failed_entries": 0,
        "skipped_json_errors": 0,
    }
    error_summary = {}

    try:
        # Process file in smaller chunks for better performance
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            entry_count = 0
            progress_bar = None
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_lines if total_lines >= 0 else None, desc="Processing entries")
            except (ImportError, ModuleNotFoundError):
                logger.info("tqdm not available, using simple progress logging")
            
            # Prepare for chunked processing
            current_chunk = []
            chunk_index = 0
            
            for line in f:
                if not line.strip():
                    continue  # Skip empty lines
                
                entry_count += 1
                stats["total_entries"] = entry_count
                
                if progress_bar:
                    progress_bar.update(1)
                elif entry_count % PROGRESS_REPORT_INTERVAL == 0:
                    logger.info(f"Processed {entry_count} entries...")
                
                # Add to current chunk
                current_chunk.append(line.strip())
                
                # Process chunk when it reaches the desired size
                if len(current_chunk) >= CHUNK_SIZE:
                    # Parse JSON entries since we're receiving text lines
                    parsed_entries = []
                    for line in current_chunk:
                        try:
                            entry = json.loads(line)
                            parsed_entries.append(entry)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in chunk {chunk_index}: {line[:100]}...")
                            stats["skipped_json_errors"] += 1
                    
                    if parsed_entries:
                        chunk_result = process_entry_chunk(parsed_entries, chunk_index)
                        
                        # Update statistics
                        stats["processed_ok"] += chunk_result["processed_ok"]
                        stats["processed_with_errors"] += chunk_result["processed_with_errors"]
                        stats["failed_entries"] += chunk_result["failed_entries"]
                        
                        # Update error summary
                        for error_key, error_count in chunk_result["error_details"].items():
                            error_summary[error_key] = error_summary.get(error_key, 0) + error_count
                        
                        # Update entry stats
                        for stat_key, stat_value in chunk_result["stats"].items():
                            entry_stats[stat_key] += stat_value
                        
                        # Reset for next chunk
                        current_chunk = []
                        chunk_index += 1

            if progress_bar:
                progress_bar.close()

            # Process any remaining entries in the final chunk
            if current_chunk:
                # Parse JSON entries since we're receiving text lines
                parsed_entries = []
                for line in current_chunk:
                    try:
                        entry = json.loads(line)
                        parsed_entries.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in final chunk: {line[:100]}...")
                        stats["skipped_json_errors"] += 1
                
                if parsed_entries:
                    chunk_result = process_entry_chunk(parsed_entries, chunk_index)
                    
                    # Update statistics 
                    stats["processed_ok"] += chunk_result["processed_ok"]
                    stats["processed_with_errors"] += chunk_result["processed_with_errors"]
                    stats["failed_entries"] += chunk_result["failed_entries"]
                    
                    # Update error summary
                    for error_key, error_count in chunk_result["error_details"].items():
                        error_summary[error_key] = error_summary.get(error_key, 0) + error_count
                    
                    # Update entry stats
                    for stat_key, stat_value in chunk_result["stats"].items():
                        entry_stats[stat_key] += stat_value

    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return {
            "total_entries": stats.get("total_entries", 0),
            "processed_entries": 0,
            "error_entries": 1,
            "skipped_entries": 0,
            "detail_stats": entry_stats,
        }
    except Exception as e:
        logger.error(f"Fatal error processing {filename}: {str(e)}", exc_info=True)
        stats["failed_entries"] = (
            stats.get("total_entries", 0)
            - stats["processed_ok"]
            - stats["processed_with_errors"]
            - stats["skipped_json_errors"]
        )
        if stats["failed_entries"] < 0:
            stats["failed_entries"] = 0

    # Final logging
    logger.info(f"Completed processing {filename}:")
    logger.info(f"  Total lines: {total_lines if total_lines >= 0 else 'Unknown'}")
    logger.info(f"  Successfully processed: {stats['processed_ok']}")
    logger.info(
        f"  Processed with non-critical errors: {stats['processed_with_errors']}"
    )
    logger.info(f"  Failed entries: {stats['failed_entries']}")
    logger.info(f"  Skipped JSON errors: {stats['skipped_json_errors']}")
    logger.info(f"  --- Data processed: ---")
    for key, value in entry_stats.items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")

    if error_summary:
        logger.warning(
            f"Error summary for {filename}: {json.dumps(error_summary, indent=2)}"
        )

    return {
        "total_entries": stats["total_entries"],
        "processed_entries": stats["processed_ok"] + stats["processed_with_errors"],
        "error_entries": stats["failed_entries"],
        "skipped_entries": stats["skipped_json_errors"],
        "detail_stats": entry_stats,
    }


