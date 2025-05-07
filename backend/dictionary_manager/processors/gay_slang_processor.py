#!/usr/bin/env python3
"""
Processor for the Gay Slang JSON dictionary file.
"""

import json
import logging
import os
import re # Added re import
from typing import Tuple, Dict, List, Any, Optional

# Third-party imports
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # Optional dependency
from psycopg2.extras import Json # ADDED: For wrapping JSON data

# Project-specific imports (using absolute paths)
from backend.dictionary_manager.db_helpers import (
    get_or_create_word_id,
    insert_definition,
    insert_relation,
    insert_etymology,
    get_standardized_pos_id,
    get_uncategorized_pos_id,
    insert_definition_example,
)
from backend.dictionary_manager.text_helpers import (
    SourceStandardization,
    standardize_source_identifier,
    normalize_lemma,
    get_standard_code
)
from backend.dictionary_manager.enums import RelationshipType

logger = logging.getLogger(__name__)

def process_gay_slang_json(cur, filename: str) -> Tuple[int, int]:
    """
    Processes entries from the gay-slang.json file.
    Handles JSON input that is a list of entry objects.
    Manages transactions manually using savepoints for individual entry resilience.
    FIXED: Correctly handles examples via insert_definition_example.
    FIXED: Stores English synonyms as relations instead of appending to definition text.

    Args:
        cur: Database cursor
        filename: Path to the gay-slang.json file

    Returns:
        Tuple: (number_of_entries_processed_successfully, number_of_entries_with_errors)
    """
    # Standardize source identifier consistently
    raw_source_identifier = os.path.basename(filename)
    source_identifier_map = {
        "gay-slang.json": "Philippine Slang and Gay Dictionary (2023)"
    }
    source_identifier = source_identifier_map.get(
        raw_source_identifier,
        standardize_source_identifier(raw_source_identifier)
    )
    if not source_identifier:
        source_identifier = "Gay Slang Dictionary"  # Default fallback

    logger.info(f"Processing Gay Slang file: {filename}")
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    conn = cur.connection
    if conn.closed:
        logger.error("Connection is closed. Cannot proceed with processing.")
        return 0, 1  # Indicate 0 processed, 1 issue (connection error)

    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return 0, 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        raise RuntimeError(f"Invalid JSON in file {filename}: {e}") from e
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        raise RuntimeError(f"Error reading file {filename}: {e}") from e

    if not isinstance(data, list):
        logger.error(f"File {filename} does not contain a list of entries as expected.")
        return 0, 1

    entries_in_file = len(data)
    if entries_in_file == 0:
        logger.info(f"Found 0 entries in {filename}. Skipping file.")
        return 0, 0

    logger.info(f"Found {entries_in_file} entries in {filename}")

    # Statistics tracking
    stats = {
        "processed": 0,
        "definitions": 0,
        "relations": 0,
        "synonyms": 0,
        "eng_synonyms": 0,
        "variants": 0,
        "etymologies": 0,
        "examples": 0,
        "skipped_invalid": 0,
        "errors": 0,
    }
    error_types = {}

    pbar = None
    try:
        pbar = tqdm(
            total=entries_in_file,
            desc=f"Processing {source_identifier}",
            unit="entry",
            leave=False,
        )

        for entry_index, entry in enumerate(data):
            savepoint_name = f"gayslang_entry_{entry_index}"
            lemma = ""
            word_id = None

            try:
                cur.execute(f"SAVEPOINT {savepoint_name}")

                if not isinstance(entry, dict):
                    logger.warning(
                        f"Skipping non-dictionary item at index {entry_index} in {filename}"
                    )
                    stats["skipped_invalid"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    pbar.update(1)
                    continue

                lemma = entry.get("headword", "").strip()
                if not lemma:
                    logger.warning(
                        f"Skipping entry at index {entry_index} due to missing/empty 'headword' field"
                    )
                    stats["skipped_invalid"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    pbar.update(1)
                    continue

                # Assume Tagalog ('tl') as the base language for this dictionary
                language_code = "tl"

                # --- Extract Metadata ---
                entry_metadata = entry.get("metadata", {})
                word_metadata_to_store = {}
                if isinstance(entry_metadata, dict) and "page" in entry_metadata:
                    word_metadata_to_store["page"] = entry_metadata["page"]
                word_metadata_to_store["source_file"] = (
                    raw_source_identifier  # Store original filename
                )

                # --- Get or Create Word ---
                try:
                    word_id = get_or_create_word_id(
                        cur,
                        lemma=lemma,
                        language_code=language_code,
                        source_identifier=source_identifier,
                        word_metadata=word_metadata_to_store,
                    )
                    if not word_id:
                        raise ValueError("get_or_create_word_id returned None")
                    logger.debug(f"Word '{lemma}' ({language_code}) -> ID: {word_id}")
                except Exception as word_err:
                    logger.error(
                        f"CRITICAL FAILURE creating word '{lemma}': {word_err}"
                    )
                    raise word_err  # Re-raise critical error to trigger outer catch

                # --- Process Etymology ---
                etymology_text = entry.get("etymology", "").strip()
                if etymology_text:
                    try:
                        ety_id = insert_etymology(
                            cur, word_id, etymology_text, source_identifier
                        )
                        if ety_id:
                            stats["etymologies"] += 1
                    except Exception as ety_err:
                        logger.warning(
                            f"Error inserting etymology for '{lemma}' (ID: {word_id}): {ety_err}"
                        )
                        error_key = f"EtymologyInsertError: {type(ety_err).__name__}"
                        error_types[error_key] = error_types.get(error_key, 0) + 1

                # --- Process POS ---
                # Use the first POS found, standardize it
                pos_list = entry.get("partOfSpeech", [])
                raw_pos_str = (
                    pos_list[0].strip()
                    if pos_list and isinstance(pos_list[0], str) and pos_list[0].strip()
                    else None
                )
                
                # --- End POS Mapping ---

                # --- Process Usage Labels as Word Tags ---
                # Add usage labels to the word's tags column
                usage_labels = entry.get("usageLabels", [])
                if usage_labels and isinstance(usage_labels, list):
                    cleaned_labels = [
                        str(l).strip() for l in usage_labels if l and isinstance(l, str)
                    ]
                    cleaned_labels = [
                        l for l in cleaned_labels if l
                    ]  # Ensure not empty after strip
                    if cleaned_labels:
                        labels_str = ",".join(cleaned_labels)
                        try:
                            # Append labels to existing tags (comma-separated)
                            # Use COALESCE to handle NULL tags and append with a comma
                            # Check if words.tags column exists before updating
                            cur.execute(
                                """UPDATE words
                                    SET tags = COALESCE(tags || ',', '') || %s
                                    WHERE id = %s""",
                                (labels_str, word_id),
                            )
                            logger.debug(
                                f"Appended usage labels {cleaned_labels} to tags for word ID {word_id}"
                            )
                        except Exception as tag_update_err:
                            logger.warning(
                                f"Failed to update tags with usage labels for word ID {word_id}: {tag_update_err}"
                            )

                # --- Process Definitions and Examples ---
                definitions_raw = entry.get("definitions", [])
                examples_raw = entry.get("examples", [])

                if isinstance(definitions_raw, list):
                    for def_item in definitions_raw:
                        if isinstance(def_item, dict):
                            def_lang = def_item.get(
                                "language", ""
                            ).lower()  # Use language from definition item
                            def_meaning = def_item.get("meaning", "").strip()

                            if not def_meaning:
                                continue  # Skip empty definitions

                            def_id = None  # Reset def_id for each definition attempt
                            try:
                                # Insert the definition WITHOUT examples argument
                                def_id = insert_definition(
                                    cur,
                                    word_id,
                                    def_meaning,
                                    sources=source_identifier,
                                    part_of_speech=get_standard_code(raw_pos_str),
                                )
                                if def_id:
                                    stats["definitions"] += 1

                                    # --- FIX: Process Examples AFTER getting def_id ---
                                    if isinstance(examples_raw, list):
                                        for ex_item in examples_raw:
                                            # Find example matching the definition's language
                                            if (
                                                isinstance(ex_item, dict)
                                                and ex_item.get("language", "").lower()
                                                == def_lang
                                            ):
                                                example_text = (
                                                    ex_item.get("example") or ""
                                                ).strip()
                                                if example_text:
                                                    try:
                                                        # Call insert_definition_example
                                                        ex_db_id = (
                                                            insert_definition_example(
                                                                cur,
                                                                def_id,
                                                                {
                                                                    "text": example_text
                                                                },  # Pass data as dict
                                                                source_identifier,
                                                            )
                                                        )
                                                        if ex_db_id:
                                                            stats["examples"] += 1
                                                    except Exception as ex_err:
                                                        logger.warning(
                                                            f"Error inserting example for def ID {def_id} ('{lemma}'): {ex_err}"
                                                        )
                                                        # Don't let example failure stop other processing for this definition

                            except Exception as def_err:
                                logger.warning(
                                    f"Error inserting definition for '{lemma}' ({def_lang}): {def_err}"
                                )
                                error_key = (
                                    f"DefinitionInsertError: {type(def_err).__name__}"
                                )
                                error_types[error_key] = (
                                    error_types.get(error_key, 0) + 1
                                )
                                # Continue to next definition if one fails

                # --- Process Relations ---
                # Variations
                variations = entry.get("variations", [])
                if isinstance(variations, list):
                    for var_word in variations:
                        if (
                            isinstance(var_word, str)
                            and var_word.strip()
                            and var_word.strip().lower() != lemma.lower()
                        ):
                            try:
                                var_id = get_or_create_word_id(
                                    cur,
                                    var_word.strip(),
                                    language_code,
                                    source_identifier=source_identifier,
                                )
                                if var_id and var_id != word_id:
                                    # VARIANT is bidirectional
                                    rel_id_1 = insert_relation(
                                        cur,
                                        word_id,
                                        var_id,
                                        RelationshipType.VARIANT.value,
                                        source_identifier,
                                    )
                                    rel_id_2 = insert_relation(
                                        cur,
                                        var_id,
                                        word_id,
                                        RelationshipType.VARIANT.value,
                                        source_identifier,
                                    )
                                    if rel_id_1:
                                        stats["relations"] += 1
                                        stats["variants"] += 1  # Count once per pair
                            except Exception as rel_err:
                                logger.warning(
                                    f"Error creating VARIANT relation for '{lemma}' -> '{var_word}': {rel_err}"
                                )
                                error_key = (
                                    f"RelationInsertError: {type(rel_err).__name__}"
                                )
                                error_types[error_key] = (
                                    error_types.get(error_key, 0) + 1
                                )

                # Filipino Synonyms (sangkahulugan)
                fil_synonyms = entry.get("sangkahulugan", [])
                if isinstance(fil_synonyms, list):
                    for syn_word in fil_synonyms:
                        if (
                            isinstance(syn_word, str)
                            and syn_word.strip()
                            and syn_word.strip().lower() != lemma.lower()
                        ):
                            try:
                                # Assume Filipino synonyms are 'tl'
                                syn_id = get_or_create_word_id(
                                    cur,
                                    syn_word.strip(),
                                    language_code,
                                    source_identifier=source_identifier,
                                )
                                if syn_id and syn_id != word_id:
                                    # SYNONYM is bidirectional
                                    rel_id_1 = insert_relation(
                                        cur,
                                        word_id,
                                        syn_id,
                                        RelationshipType.SYNONYM.value,
                                        source_identifier,
                                    )
                                    rel_id_2 = insert_relation(
                                        cur,
                                        syn_id,
                                        word_id,
                                        RelationshipType.SYNONYM.value,
                                        source_identifier,
                                    )
                                    if rel_id_1:
                                        stats["relations"] += 1
                                        stats["synonyms"] += 1  # Count once per pair
                            except Exception as rel_err:
                                logger.warning(
                                    f"Error creating Filipino SYNONYM relation for '{lemma}' -> '{syn_word}': {rel_err}"
                                )
                                error_key = (
                                    f"RelationInsertError: {type(rel_err).__name__}"
                                )
                                error_types[error_key] = (
                                    error_types.get(error_key, 0) + 1
                                )

                # --- FIX: Process English Synonyms as Relations ---
                eng_synonyms = entry.get("synonyms", [])
                if isinstance(eng_synonyms, list):
                    for syn_word in eng_synonyms:
                        if isinstance(syn_word, str) and syn_word.strip():
                            # We assume these are English words
                            eng_syn_lang_code = "en"
                            # Check if the English synonym is the same as the headword (case-insensitive)
                            if (
                                syn_word.strip().lower() != lemma.lower()
                            ):  # Avoid self-relation
                                try:
                                    syn_id = get_or_create_word_id(
                                        cur,
                                        syn_word.strip(),
                                        eng_syn_lang_code,
                                        source_identifier=source_identifier,
                                    )
                                    if (
                                        syn_id and syn_id != word_id
                                    ):  # Ensure we got an ID and it's not the same word
                                        # SYNONYM is bidirectional
                                        rel_id_1 = insert_relation(
                                            cur,
                                            word_id,
                                            syn_id,
                                            RelationshipType.SYNONYM.value,
                                            source_identifier,
                                        )
                                        rel_id_2 = insert_relation(
                                            cur,
                                            syn_id,
                                            word_id,
                                            RelationshipType.SYNONYM.value,
                                            source_identifier,
                                        )
                                        if rel_id_1:
                                            stats["relations"] += 1
                                            stats[
                                                "eng_synonyms"
                                            ] += 1  # Count once per pair
                                except Exception as rel_err:
                                    logger.warning(
                                        f"Error creating English SYNONYM relation for '{lemma}' -> '{syn_word}': {rel_err}"
                                    )
                                    error_key = (
                                        f"RelationInsertError: {type(rel_err).__name__}"
                                    )
                                    error_types[error_key] = (
                                        error_types.get(error_key, 0) + 1
                                    )

                # --- Finish Entry ---
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                stats["processed"] += 1

                # Commit every 100 entries to avoid transaction bloat
                if stats["processed"] % 100 == 0:
                    conn.commit()
                    logger.debug(f"Committed after processing {stats['processed']} entries")

            except Exception as entry_err:
                logger.error(
                    f"Failed processing entry #{entry_index} ('{lemma}') in {filename}: {entry_err}",
                    exc_info=True,
                )
                stats["errors"] += 1
                error_key = f"EntryProcessingError: {type(entry_err).__name__}"
                error_types[error_key] = error_types.get(error_key, 0) + 1
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                except Exception as rb_err:
                    logger.critical(
                        f"Failed to rollback/release savepoint {savepoint_name} after entry error: {rb_err}",
                        exc_info=True,
                    )
                    try:
                        conn.rollback()
                        logger.info("Performed full transaction rollback after savepoint failure")
                    except Exception as full_rb_err:
                        logger.critical(f"CRITICAL: Failed both savepoint and full rollback: {full_rb_err}")
                        raise  # Re-raise to stop processing
            finally:
                pbar.update(1)

    except Exception as e:
        logger.critical(f"Critical error during processing: {e}", exc_info=True)
        error_types["CriticalProcessingError"] = error_types.get("CriticalProcessingError", 0) + 1
        try:
            conn.rollback()
            logger.info("Rolled back transaction after critical error")
        except Exception as rb_error:
            logger.critical(f"CRITICAL: Failed to rollback after critical error: {rb_error}")
    finally:
        if pbar:
            pbar.close()

    # Log final stats
    logger.info(
        f"Finished processing {filename}. Processed: {stats['processed']}, Skipped: {stats['skipped_invalid']}, Errors: {stats['errors']}"
    )
    logger.info(
        f"  Stats => Defs: {stats['definitions']}, Examples: {stats['examples']}, Etys: {stats['etymologies']}, Relations: {stats['relations']}"
    )
    if error_types:
        logger.warning(f"Error summary for {filename}: {error_types}")

    total_issues = stats["skipped_invalid"] + stats["errors"]
    return stats["processed"], total_issues



