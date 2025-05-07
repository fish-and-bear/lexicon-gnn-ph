#!/usr/bin/env python3
"""
Process Sofronio G. Calderón's DICCIONARIO INGLES-ESPAÑOL-TAGALOG (1915) data.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Optional dependency

# Project-specific imports
from backend.dictionary_manager.db_helpers import (
    DBConnection,
    get_or_create_word_id,
    insert_definition,
    insert_relation,
    insert_pronunciation,
    # Add other necessary imports:
    # insert_etymology, insert_word_form, etc.
)
from backend.dictionary_manager.text_helpers import (
    clean_html,
    normalize_lemma,
    standardize_source_identifier,
    get_standard_code
    # Add other necessary imports:
    # extract_parenthesized_text, etc.
)
from backend.dictionary_manager.enums import RelationshipType

logger = logging.getLogger(__name__)

# Define the source identifier for this dictionary
SOURCE_IDENTIFIER_CALDERON = "Calderon Diccionario 1915"

def _process_single_calderon_entry(cur, entry_dict: Dict[str, Any], source_identifier: str, word_id_cache: Dict) -> Dict:
    """Processes a single entry object from the Calderon dictionary JSON list."""
    stats = {
        "definitions": 0,
        "relations": 0,
        "pronunciations": 0,
        "words_created": 0,
        "errors": 0,
        "pos_standardized": 0, # New stat
        "skipped_equivalents": 0, # New stat
    }
    error_messages = []

    # --- 1. Extract English Entry Data ---
    english_entry = entry_dict.get("english_entry")
    if not english_entry or not isinstance(english_entry, dict):
        logger.warning(f"Skipping entry due to missing or invalid 'english_entry' object: {entry_dict}")
        return {"errors": 1, "error_details": ["Missing english_entry"]}

    headword = english_entry.get("headword")
    language_code = "en" # English headwords

    if not headword or not isinstance(headword, str):
        logger.warning(f"Skipping entry due to missing/invalid headword within english_entry: {english_entry}")
        return {"errors": 1, "error_details": ["Missing headword in english_entry"]}

    cleaned_headword = clean_html(headword).strip()
    if not cleaned_headword:
        logger.warning(f"Skipping entry due to empty headword after cleaning: Original '{headword}'")
        return {"errors": 1, "error_details": ["Empty headword after cleaning"]}

    logger.debug(f"[Calderon:{cleaned_headword}] Processing entry...")

    # --- 2. Get or Create English Word ID & Handle POS/Pronunciation ---
    english_word_id = None
    original_pos = None         # Will store the raw POS string like "v.", "n."
    standardized_pos = None     # Will store the mapped POS string like "verb", "noun"
    pronunciation_text = None

    try:
        # Extract POS and Pronunciation
        original_pos_str = english_entry.get("part_of_speech")
        pronunciation_str = english_entry.get("pronunciation")

        word_metadata = {}

        # Process POS
        if original_pos_str and isinstance(original_pos_str, str):
            original_pos = original_pos_str.strip() # Store the raw original POS
            if original_pos:
                word_metadata['calderon_pos'] = original_pos
                # Standardize POS using get_standard_code
                standardized_pos = get_standard_code(original_pos)
                if standardized_pos != 'unc': # Check if mapping was successful (not uncategorized)
                     stats["pos_standardized"] += 1
                     word_metadata['standardized_pos'] = standardized_pos
                else:
                     if original_pos.lower() != 'unc': # Avoid logging if original was already 'unc' or similar
                         logger.debug(f"[Calderon:{cleaned_headword}] POS '{original_pos}' mapped to 'unc' by get_standard_code.")
                     word_metadata['standardized_pos'] = None # Explicitly mark as not mapped or mapped to unc
            # If original_pos was empty after strip, original_pos remains None here

        # Process Pronunciation (store as-is after stripping)
        if pronunciation_str and isinstance(pronunciation_str, str):
            pronunciation_text = pronunciation_str.strip()
            if pronunciation_text:
                word_metadata['calderon_pronunciation'] = pronunciation_text

        # Get or Create Word ID
        english_word_id = get_or_create_word_id(
            cur,
            cleaned_headword,
            language_code,
            source_identifier,
            word_metadata=word_metadata # Pass metadata with original POS, standardized POS, and Pron
        )
        if english_word_id is None:
            raise ValueError(f"Failed to get/create word ID for '{cleaned_headword}' ({language_code})")
        # Increment word count only if it was newly created (assuming get_or_create returns ID even if exists)
        # The current db_helper doesn't easily return creation status, so we approximate
        # Use a language-prefixed key for the cache
        cache_key_en = f"en:{cleaned_headword}"
        if cache_key_en not in word_id_cache:
             stats["words_created"] += 1
             word_id_cache[cache_key_en] = english_word_id # Basic caching

        # Insert pronunciation separately (if extracted)
        if pronunciation_text:
            try:
                pron_id = insert_pronunciation(
                    cur,
                    english_word_id,
                    pronunciation_type='text', # Label it as text pronunciation
                    value=pronunciation_text, # Store the potentially corrupted text
                    source_identifier=source_identifier,
                    metadata={"context": "Calderon Dictionary Pronunciation (potential encoding issues)"}
                )
                if pron_id:
                    stats["pronunciations"] += 1
            except Exception as pron_e:
                logger.warning(f"[Calderon:{cleaned_headword}] Failed to insert pronunciation '{pronunciation_text}': {pron_e}")
                error_messages.append(f"Pronunciation insert error: {pron_e}")

    except Exception as e:
        logger.error(f"[Calderon:{cleaned_headword}] Error getting/creating word ID or processing base info: {e}", exc_info=True)
        stats["errors"] += 1
        error_messages.append(f"Word ID/Base Info error: {e}")
        return {**stats, "error_details": error_messages}

    # --- 3. Process Tagalog Equivalents ---
    tagalog_translation_str = entry_dict.get("tagalog_translation") # Correct key
    if tagalog_translation_str and isinstance(tagalog_translation_str, str):
        # Split only by comma now
        tagalog_equivalents = [p.strip() for p in tagalog_translation_str.split(',') if p.strip()]

        processed_count_tl = 0
        for tagalog_word in tagalog_equivalents:
            cleaned_tagalog = clean_html(tagalog_word).strip()

            # Basic Filtering: skip empty strings or very short strings unlikely to be words
            # Allows multi-word expressions. Consider refining filter if needed.
            if cleaned_tagalog and len(cleaned_tagalog) > 1: # Simple length filter
                try:
                    tagalog_word_id = get_or_create_word_id(cur, cleaned_tagalog, "tl", source_identifier)
                    if tagalog_word_id:
                        # Increment word count only if new
                        cache_key_tl = f"tl:{cleaned_tagalog}"
                        if cache_key_tl not in word_id_cache:
                             stats["words_created"] += 1
                             word_id_cache[cache_key_tl] = tagalog_word_id

                        # Add standardized English POS to relation metadata
                        relation_metadata_tl = {
                            "context": "Calderon Dictionary Tagalog Equivalent"
                        }
                        if standardized_pos: # Use the mapped POS here
                            relation_metadata_tl["english_pos_context"] = standardized_pos
                        elif original_pos: # Fallback to original if not standardized
                             relation_metadata_tl["english_pos_context"] = original_pos

                        rel_id_en_tl = insert_relation(
                            cur, english_word_id, tagalog_word_id,
                            RelationshipType.HAS_TRANSLATION.value, source_identifier,
                            metadata=relation_metadata_tl
                        )

                        relation_metadata_en = {
                            "context": "Calderon Dictionary English Equivalent"
                        }
                        if standardized_pos:
                            relation_metadata_en["english_pos_context"] = standardized_pos
                        elif original_pos:
                             relation_metadata_en["english_pos_context"] = original_pos

                        rel_id_tl_en = insert_relation(
                            cur, tagalog_word_id, english_word_id,
                            RelationshipType.TRANSLATION_OF.value, source_identifier,
                            metadata=relation_metadata_en
                        )
                        if rel_id_en_tl or rel_id_tl_en:
                             stats["relations"] += 1 # Count pair as one relation added in total stats
                             processed_count_tl += 1

                except Exception as e:
                    # Improved logging: Include exception type and full traceback info
                    logger.error(
                        f"[Calderon:{cleaned_headword}] Error processing Tagalog equivalent '{cleaned_tagalog}'",
                        exc_info=True # Automatically adds traceback
                    )
                    # Add exception type and message to error_messages for summary
                    error_detail = f"Tagalog equivalent error ('{cleaned_tagalog}'): {type(e).__name__}: {e}"
                    error_messages.append(error_detail)
            elif cleaned_tagalog:
                # Log potentially skipped equivalents
                logger.debug(f"[Calderon:{cleaned_headword}] Skipping potential short/invalid Tagalog equivalent: '{cleaned_tagalog}'")
                stats["skipped_equivalents"] += 1
        # Log if no equivalents were processed from a non-empty string
        if not processed_count_tl and tagalog_translation_str.strip(): # Check original string wasn't just whitespace
             logger.debug(f"[Calderon:{cleaned_headword}] No valid Tagalog equivalents extracted from: '{tagalog_translation_str}'")

    # --- 4. Process Spanish Equivalents ---
    spanish_translation_str = entry_dict.get("spanish_translation") # Correct key
    tagalog_translation_str_for_spanish = entry_dict.get("tagalog_translation") # Need this to clean spanish part

    if spanish_translation_str and isinstance(spanish_translation_str, str):
        # Attempt to extract only the Spanish part
        spanish_only_str = spanish_translation_str.strip()
        if tagalog_translation_str_for_spanish and isinstance(tagalog_translation_str_for_spanish, str):
             tagalog_part = tagalog_translation_str_for_spanish.strip()
             if tagalog_part and spanish_only_str.endswith(tagalog_part):
                 # Remove the Tagalog part
                 potential_spanish = spanish_only_str[:-len(tagalog_part)].strip()
                 # Only use if it actually removed something and didn't result in empty string
                 if potential_spanish and len(potential_spanish) < len(spanish_only_str):
                     spanish_only_str = potential_spanish
                 else:
                      # Log if Tagalog part removal didn't work as expected (e.g., Spanish was identical to Tagalog)
                      logger.debug(f"[Calderon:{cleaned_headword}] Spanish/Tagalog parts might be identical or overlap issue. Using original Spanish string: '{spanish_translation_str}'")
             # If tagalog part wasn't found at the end, assume the original string is just Spanish

        # Split only by comma now, using the potentially cleaned spanish_only_str
        spanish_equivalents = [p.strip() for p in spanish_only_str.split(',') if p.strip()]

        processed_count_es = 0
        for spanish_word in spanish_equivalents:
            cleaned_spanish = clean_html(spanish_word).strip()

            # Basic Filtering (same as Tagalog)
            if cleaned_spanish and len(cleaned_spanish) > 1:
                try:
                    spanish_word_id = get_or_create_word_id(cur, cleaned_spanish, "es", source_identifier)
                    if spanish_word_id:
                        # Increment word count only if new
                        cache_key_es = f"es:{cleaned_spanish}"
                        if cache_key_es not in word_id_cache:
                             stats["words_created"] += 1
                             word_id_cache[cache_key_es] = spanish_word_id

                        # Add standardized English POS to relation metadata
                        relation_metadata_es = {
                            "context": "Calderon Dictionary Spanish Equivalent"
                        }
                        if standardized_pos:
                            relation_metadata_es["english_pos_context"] = standardized_pos
                        elif original_pos:
                             relation_metadata_es["english_pos_context"] = original_pos

                        rel_id_en_es = insert_relation(
                            cur, english_word_id, spanish_word_id,
                            RelationshipType.HAS_TRANSLATION.value, source_identifier,
                            metadata=relation_metadata_es
                        )

                        relation_metadata_en_es = {
                            "context": "Calderon Dictionary English Equivalent"
                        }
                        if standardized_pos:
                            relation_metadata_en_es["english_pos_context"] = standardized_pos
                        elif original_pos:
                             relation_metadata_en_es["english_pos_context"] = original_pos

                        rel_id_es_en = insert_relation(
                            cur, spanish_word_id, english_word_id,
                            RelationshipType.TRANSLATION_OF.value, source_identifier,
                            metadata=relation_metadata_en_es
                        )
                        if rel_id_en_es or rel_id_es_en:
                            stats["relations"] += 1
                            processed_count_es += 1

                except Exception as e:
                    # Improved logging: Include exception type and full traceback info
                    logger.error(
                        f"[Calderon:{cleaned_headword}] Error processing Spanish equivalent '{cleaned_spanish}'",
                        exc_info=True # Automatically adds traceback
                    )
                    # Add exception type and message to error_messages for summary
                    error_detail = f"Spanish equivalent error ('{cleaned_spanish}'): {type(e).__name__}: {e}"
                    error_messages.append(error_detail)
            elif cleaned_spanish:
                logger.debug(f"[Calderon:{cleaned_headword}] Skipping potential short/invalid Spanish equivalent: '{cleaned_spanish}'")
                stats["skipped_equivalents"] += 1
        if not processed_count_es and spanish_translation_str.strip(): # Check original string wasn't just whitespace
             logger.debug(f"[Calderon:{cleaned_headword}] No valid Spanish equivalents extracted from: '{spanish_translation_str}' (Processed as: '{spanish_only_str}')")

    # --- 5. Process Definition Field ---
    definition_text = english_entry.get("definition")
    if definition_text and isinstance(definition_text, str):
        cleaned_definition = definition_text.strip()
        if cleaned_definition:
            # Determine POS to use for this definition entry
            # Prioritize standardized_pos, fallback to original_pos if standardization failed but original exists
            pos_for_definition = standardized_pos if standardized_pos else original_pos
            try:
                def_id = insert_definition(
                    cur,
                    english_word_id,
                    definition_text=cleaned_definition,
                    part_of_speech=pos_for_definition, # Pass the determined POS
                    sources=source_identifier, # Pass source identifier string directly
                    metadata={"context": "Calderon Dictionary Combined ES/TL Definition"}
                )
                if def_id:
                    stats["definitions"] += 1
            except Exception as def_e:
                 logger.warning(f"[Calderon:{cleaned_headword}] Failed to insert definition '{cleaned_definition[:50]}...': {def_e}")
                 # Treat as non-critical error
                 error_messages.append(f"Definition insert error: {def_e}")
                 if stats["errors"] == 0: stats["errors"] = 2 # Mark as having non-critical errors

    # --- 6. Process Other Fields --- Ignored: english_entry.raw
    # raw_text = english_entry.get("raw")

    if error_messages:
        # If a critical error occurred (word ID creation failed), stats["errors"] will be 1.
        # If only non-critical errors occurred, stats["errors"] is still 0 here.
        if stats["errors"] == 0:
             stats["errors"] = 2 # Use 2 to indicate non-critical errors occurred
        stats["error_details"] = error_messages
        logger.warning(f"[Calderon:{cleaned_headword}] Processed with {len(error_messages)} non-critical issue(s). Details: {error_messages}")
    # Ensure errors is 1 if word ID failed even if other errors occurred
    # This logic might be slightly redundant depending on when Word ID error is caught, but aims for clarity
    # elif stats.get("error_details") and stats.get("errors", 0) == 0:
    #      stats["errors"] = 1 # Should have error details, mark as error

    return stats


def process_calderon_json(cur, filename: str) -> Dict:
    """
    Processes the Calderon dictionary JSON file.
    Assumes the main structure is a JSON array: [ { "english_entry": {...} }, ... ]
    """
    logger.info(f"Starting processing for Calderon dictionary from {filename}")
    source_identifier = standardize_source_identifier(SOURCE_IDENTIFIER_CALDERON)
    overall_stats = {
        "total_entries": 0,
        "processed_ok": 0,
        "processed_with_errors": 0, # Entries with non-critical errors (code 2)
        "failed_entries": 0,        # Entries with critical errors (code 1)
        "definitions": 0,
        "relations": 0,
        "pronunciations": 0,
        "words_created": 0,
        "pos_standardized": 0, # New stat
        "skipped_equivalents": 0, # New stat
    }
    error_summary = {}
    word_id_cache = {} # Simple cache for this run: Dict[str, int] -> { "lang:word": word_id }

    try:
        with open(filename, "r", encoding="utf-8") as f: # Assume UTF-8 for reading the JSON file itself
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Calderon dictionary file not found: {filename}")
        return {**overall_stats, "failed_entries": 1, "error_details": ["File not found"]}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filename}: {e}")
        return {**overall_stats, "failed_entries": 1, "error_details": [f"JSON Decode Error: {e}"]}
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return {**overall_stats, "failed_entries": 1, "error_details": [f"File Read Error: {e}"]}

    # --- Check if data is a list ---
    if not isinstance(data, list):
        logger.error(f"Error: Expected a JSON array (list) at the root, but found {type(data)} in {filename}")
        if isinstance(data, dict) and "entries" in data and isinstance(data["entries"], list):
             logger.warning("Found a dictionary with an 'entries' key instead of a root array. Processing the 'entries' list.")
             entries_to_process = data["entries"]
        else:
             return {**overall_stats, "failed_entries": 1, "error_details": ["Root JSON structure is not a list"]}
    else:
         entries_to_process = data # The root is the list of entries

    total_entries = len(entries_to_process)
    overall_stats["total_entries"] = total_entries
    logger.info(f"Found {total_entries} entries in {filename}")

    # --- Processing Loop ---
    progress_bar = None
    if tqdm:
        progress_bar = tqdm(total=total_entries, desc=f"Processing {SOURCE_IDENTIFIER_CALDERON}", unit="entry")

    for entry_dict in entries_to_process:
        if not isinstance(entry_dict, dict):
            logger.warning(f"Skipping non-dictionary item in entries list: {entry_dict}")
            overall_stats["failed_entries"] += 1 # Count as failed
            if progress_bar: progress_bar.update(1)
            continue

        headword_for_log = entry_dict.get("english_entry", {}).get("headword", "UNKNOWN")

        try:
            savepoint_name = f"calderon_entry_{overall_stats['processed_ok'] + overall_stats['processed_with_errors'] + overall_stats['failed_entries']}"
            cur.execute(f"SAVEPOINT {savepoint_name}")

            entry_stats = _process_single_calderon_entry(cur, entry_dict, source_identifier, word_id_cache)

            # Aggregate stats based on the outcome
            entry_error_code = entry_stats.get("errors", 0)

            if entry_error_code == 1: # Critical error (e.g., word ID failure)
                 overall_stats["failed_entries"] += 1
                 if entry_stats.get("error_details"):
                     for detail in entry_stats["error_details"]:
                         error_key = detail.split(":")[0].strip() # Simple key extraction
                         error_summary[error_key] = error_summary.get(error_key, 0) + 1
            elif entry_error_code == 2: # Non-critical errors occurred
                 overall_stats["processed_with_errors"] += 1
                 if entry_stats.get("error_details"):
                     for detail in entry_stats["error_details"]:
                         error_key = detail.split(":")[0].strip()
                         error_summary[error_key] = error_summary.get(error_key, 0) + 1
            else: # Processed OK (errors == 0)
                overall_stats["processed_ok"] += 1

            # Aggregate other counts regardless of error status (if they were incremented before failure)
            overall_stats["definitions"] += entry_stats.get("definitions", 0)
            overall_stats["relations"] += entry_stats.get("relations", 0)
            overall_stats["pronunciations"] += entry_stats.get("pronunciations", 0)
            overall_stats["words_created"] += entry_stats.get("words_created", 0)
            overall_stats["pos_standardized"] += entry_stats.get("pos_standardized", 0)
            overall_stats["skipped_equivalents"] += entry_stats.get("skipped_equivalents", 0)

            cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")

        except Exception as e:
            logger.error(f"[Calderon:{headword_for_log}] CRITICAL error during main loop: {e}", exc_info=True)
            overall_stats["failed_entries"] += 1
            error_summary["CriticalLoopError"] = error_summary.get("CriticalLoopError", 0) + 1
            try:
                 cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                 cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release even after rollback
            except Exception as rb_err:
                 logger.error(f"Failed to rollback/release savepoint for entry '{headword_for_log}': {rb_err}. Transaction might be compromised.")

        if progress_bar:
            progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    # Final Logging
    logger.info(f"Finished processing {SOURCE_IDENTIFIER_CALDERON}.")
    logger.info(f"  Total Entries: {overall_stats['total_entries']}")
    logger.info(f"  Processed OK: {overall_stats['processed_ok']}")
    logger.info(f"  Processed with Issues: {overall_stats['processed_with_errors']}")
    logger.info(f"  Failed Entries: {overall_stats['failed_entries']}")
    logger.info(f"  Definitions Added: {overall_stats['definitions']}") # Now tracking definitions
    logger.info(f"  Relations Added: {overall_stats['relations']}")
    logger.info(f"  Pronunciations Added: {overall_stats['pronunciations']}")
    logger.info(f"  Words Created/Found: {overall_stats['words_created']}") # Note: This counts get_or_create calls, actual new rows might differ
    logger.info(f"  POS Standardized: {overall_stats['pos_standardized']}")
    logger.info(f"  Equivalents Skipped (Filter): {overall_stats['skipped_equivalents']}")
    if error_summary:
        logger.warning(f"  Error Summary: {json.dumps(error_summary, indent=2)}")

    return overall_stats 