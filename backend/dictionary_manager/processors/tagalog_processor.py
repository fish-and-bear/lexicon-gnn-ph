#!/usr/bin/env python3
"""
Processor for Tagalog dictionary files (specifically tagalog-words.json format).
"""

import json
import logging
import os
import re
from typing import Tuple, Dict, List, Any, Optional

# Third-party imports (ensure these are installed)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # Optional dependency

# Project-specific imports (using absolute paths)
from backend.dictionary_manager.db_helpers import (
    get_or_create_word_id,
    insert_definition,
    insert_etymology,
    insert_pronunciation,
    insert_definition_example,
    insert_relation, # Needed for inline relation processing
    process_relations_batch, # <-- ADDED import
    get_standardized_pos_id,
    with_transaction,
)
from backend.dictionary_manager.text_helpers import normalize_lemma, SourceStandardization

logger = logging.getLogger(__name__)

# Moved from dictionary_manager.py (originally around line 97 in old structure)
def process_definition_relations(cur, word_id: int, definition: str, source: str):
    """Process and create relationships from definition text."""
    synonym_patterns = [
        r"ka(singkahulugan|tulad|patulad|tumbas) ng\\s+(\\w+)",
        r"singkahulugan:\\s+(\\w+)",
    ]
    antonym_patterns = [
        r"(kasalungat|kabaligtaran) ng\\s+(\\w+)",
        r"kabaligtaran:\\s+(\\w+)",
    ]

    for pattern in synonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            syn = match.group(2).strip()
            syn_id = get_or_create_word_id(cur, syn, "tl")
            insert_relation(cur, word_id, syn_id, "synonym", source_identifier=source) # Use source_identifier

    for pattern in antonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            ant = match.group(2).strip()
            ant_id = get_or_create_word_id(cur, ant, "tl")
            insert_relation(cur, word_id, ant_id, "antonym", source_identifier=source) # Use source_identifier

# Moved from dictionary_manager.py (originally around line 852)
def _process_single_tagalog_word_entry(
    cur,  # <-- ADDED: Explicit cursor is now required
    entry_data: dict,
    word_key: str,
    source_identifier: str,
    language_code: str,  # Should be 'tl' for this processor
    add_etymology: bool,
    word_id_cache: dict,
    stats: dict,
    error_types: dict,  # Pass error_types to log specific errors
    table_column_map: dict,
    relations_batch: list,
    required_columns: dict,
):
    """
    Processes a single word entry from tagalog-words.json format using the provided cursor.
    Assumes it's running within a transaction managed by the caller.
    Raises exceptions on failure, which should be caught by the caller.
    Handles structure observed: top-level word, pronunciation, etymology obj, domains, senses list.
    Sense structure includes: definition, part_of_speech, example obj (raw, examples), references.
    """
    lemma = entry_data.get("word", word_key).strip()
    if not lemma:
        logger.warning(
            f"Skipping entry with missing/empty lemma (original key: '{word_key}')"
        )
        stats["skipped"] += 1
        return

    # --- Extract Top-Level Word Information ---
    top_level_pos_code = None
    top_level_pos_list = entry_data.get("part_of_speech", [])
    if isinstance(top_level_pos_list, list) and top_level_pos_list:
        # Expects format like [["pnl"]] or [["png"]]
        if isinstance(top_level_pos_list[0], list) and top_level_pos_list[0]:
            if (
                isinstance(top_level_pos_list[0][0], str)
                and top_level_pos_list[0][0].strip()
            ):
                top_level_pos_code = top_level_pos_list[0][0].strip()

    # --- Initial Word Insertion/Retrieval ---
    # Collect potential metadata to pass to get_or_create_word_id or update later
    word_creation_metadata = {"source_file": source_identifier}
    top_domains = entry_data.get("domains", [])  # Extract domains here for later use
    # REMOVED: Option 1 (adding domains to metadata) was here.

    # Add other top-level flags if they exist in the JSON (e.g., is_proper, is_abbr)
    # word_creation_metadata['is_proper_noun'] = entry_data.get('is_proper', False)

    # Note: Simplified call for now, only passing metadata. Add other args as needed.
    word_id = get_or_create_word_id(
        cur,
        lemma,
        language_code,
        source_identifier,
        word_metadata=word_creation_metadata, # Pass dict directly, Json() is handled internally
        # Pass other flags if extracted: has_baybayin, baybayin_form, etc.
    )
    if not word_id:
        # Error logged by get_or_create_word_id
        stats["errors"] += 1
        error_types["WordCreationFailed"] = error_types.get("WordCreationFailed", 0) + 1
        raise ValueError(f"Failed to get or create word_id for lemma: {lemma}")

    normalized_lemma_cache_key = normalize_lemma(lemma)
    cache_key = f"{normalized_lemma_cache_key}|{language_code}"
    word_id_cache[cache_key] = word_id

    # --- Store Top-Level Domains in Tags Column (Option 2) ---
    if isinstance(top_domains, list) and top_domains:
        cleaned_domains = [
            str(d).strip() for d in top_domains if d and isinstance(d, str)
        ]
        cleaned_domains = [
            d for d in cleaned_domains if d
        ]  # Ensure not empty after stripping
        if cleaned_domains:
            # Append domains to existing tags (comma-separated)
            domains_str = ",".join(cleaned_domains)
            try:
                # Use COALESCE to handle NULL tags and append with a comma
                # Ensure target column 'tags' exists in the 'words' table
                if "tags" in table_column_map.get("words", set()):
                    cur.execute(
                        """UPDATE words
                            SET tags = COALESCE(tags || ',', '') || %s
                            WHERE id = %s""",
                        (domains_str, word_id),
                    )
                    logger.debug(
                        f"Appended domains {cleaned_domains} to tags for word ID {word_id}"
                    )
                else:
                    logger.warning(
                        f"Cannot store domains: 'tags' column not found in 'words' table for word ID {word_id}."
                    )

            except Exception as tag_update_err:
                logger.warning(
                    f"Failed to update tags with domains for word ID {word_id}: {tag_update_err}"
                )
                # Log error but don't stop processing the entry

    # --- Process Top-Level Pronunciations ---
    # Uses 'pronunciation' and 'alternate_pronunciation' keys
    main_pron = entry_data.get("pronunciation")
    alt_pron = entry_data.get("alternate_pronunciation")

    if main_pron and isinstance(main_pron, str) and main_pron.strip():
        # Assume simple IPA string for now, could be enhanced to parse structure if needed
        # Use dict format for insert_pronunciation consistency
        pron_id = insert_pronunciation(
            cur, word_id, "ipa", main_pron.strip(), source_identifier=source_identifier
        )
        if pron_id:
            stats["pronunciations_added"] += 1

    if alt_pron and isinstance(alt_pron, str) and alt_pron.strip():
        # Treat alternate as a separate entry, maybe tag it?
        pron_id = insert_pronunciation(
            cur, word_id, "ipa", alt_pron.strip(), tags=["alternate"], source_identifier=source_identifier
        )
        if pron_id:
            stats["pronunciations_added"] += 1

    # --- Process Etymology ---
    etymology_obj = entry_data.get("etymology", {})
    if add_etymology and isinstance(etymology_obj, dict) and etymology_obj:
        etymology_text = etymology_obj.get("raw", "").strip()
        lang_codes = etymology_obj.get("languages", [])  # List like ["Esp"]
        # Convert list of codes to comma-separated string if needed by insert_etymology
        lang_codes_str = (
            ",".join(lc for lc in lang_codes if lc and isinstance(lc, str))
            if lang_codes
            else None
        )

        if etymology_text:
            ety_id = insert_etymology(
                cur,
                word_id,
                etymology_text,
                source_identifier,
                language_codes=lang_codes_str,
            )
            if ety_id:
                stats["etymologies_added"] += 1

        # Process terms/other_terms for potential relations
        etymology_terms = etymology_obj.get("terms", []) + etymology_obj.get(
            "other_terms", []
        )
        if etymology_terms and isinstance(etymology_terms, list):
            for term_data in etymology_terms:
                # Adapt based on actual structure of term_data (string? dict?)
                related_word = None
                rel_lang = None
                if isinstance(term_data, str):
                    related_word = term_data
                    # Try to guess language from lang_codes if only one source lang?
                    rel_lang = (
                        lang_codes[0] if lang_codes and len(lang_codes) == 1 else None
                    )
                elif isinstance(term_data, dict):
                    related_word = term_data.get("term")
                    rel_lang = term_data.get("lang")

                if (
                    related_word
                    and isinstance(related_word, str)
                    and related_word.strip()
                ):
                    # Add to batch (outside definition loop)
                    # Use a sensible default relationship like RELATED or DERIVED_FROM
                    relations_batch.append(
                        {
                            "from_word": word_id,  # Changed from "word_id" to "from_word"
                            "to_word": related_word.strip(),  # Changed from "related_word" to "to_word"
                            "relation_type": "derived_from",  # FIXED: Changed from rel_type to relation_type
                            "source": source_identifier,  # FIXED: Changed from source_identifier to source
                            "metadata": (
                                {"context": "etymology", "lang": rel_lang}
                                if rel_lang
                                else {"context": "etymology"}
                            ),
                            "def_id": None,  # Not linked to a specific definition
                        }
                    )

    # --- Process Top-Level Derivative ---
    derivative = entry_data.get("derivative")
    if derivative and isinstance(derivative, str) and derivative.strip():
        # Assume derivative is a word derived FROM the current word
        relations_batch.append(
            {
                "from_word": word_id,  # Changed from "word_id" to "from_word"
                "to_word": derivative.strip(),  # Changed from "related_word" to "to_word"
                "relation_type": "root_of",  # FIXED: Changed from rel_type to relation_type
                "source": source_identifier,  # FIXED: Changed from source_identifier to source
                "metadata": {"context": "derivative"},
                "def_id": None,
            }
        )

    # --- Process Senses / Definitions ---
    senses = entry_data.get("senses", [])
    if isinstance(senses, list):
        for sense_idx, sense in enumerate(senses):
            if not isinstance(sense, dict):
                logger.warning(
                    f"Skipping non-dict sense at index {sense_idx} for '{lemma}'"
                )
                continue

            # --- Definition Text ---
            definition_text = sense.get("definition", "").strip()
            if not definition_text:
                logger.debug(
                    f"Skipping sense {sense_idx} for '{lemma}' due to missing/empty definition text."
                )
                continue

            # --- Part of Speech ---
            sense_pos_code = None
            sense_pos_list = sense.get("part_of_speech", [])
            if isinstance(sense_pos_list, list) and sense_pos_list:
                if isinstance(sense_pos_list[0], list) and sense_pos_list[0]:
                    if (
                        isinstance(sense_pos_list[0][0], str)
                        and sense_pos_list[0][0].strip()
                    ):
                        sense_pos_code = sense_pos_list[0][0].strip()

            # Use sense POS if available, otherwise fallback to top-level POS
            pos_code_to_use = sense_pos_code if sense_pos_code else top_level_pos_code
            if not pos_code_to_use:
                logger.warning(
                    f"Missing POS for sense {sense_idx} of '{lemma}' (and no top-level POS). Skipping definition."
                )
                continue # Must have a POS to insert definition

            # Get standardized POS ID
            standardized_pos_id = get_standardized_pos_id(cur, pos_code_to_use)

            # --- Extract Usage Notes ---
            sense_notes = sense.get("notes") # Could be list or string
            usage_notes_str = None
            if isinstance(sense_notes, list):
                 usage_notes_str = "; ".join(str(n) for n in sense_notes if n)
            elif isinstance(sense_notes, str):
                 usage_notes_str = sense_notes.strip()

            # Create metadata dictionary
            metadata_dict = {}
            if pos_code_to_use:
                metadata_dict["original_pos"] = pos_code_to_use
            if usage_notes_str:
                 metadata_dict["usage_notes"] = usage_notes_str
            # Add other sense-level data if available (e.g., domains, tags)

            # Pass sense data to insert_definition
            definition_id = insert_definition(
                cur,
                word_id,
                definition_text,
                part_of_speech=pos_code_to_use,
                usage_notes=usage_notes_str if usage_notes_str else None,
                metadata=metadata_dict if metadata_dict else None,
                sources=source_identifier,  # Changed from source_identifier to sources
            )

            if not definition_id:
                # Error logged by insert_definition
                stats["errors"] += 1
                error_types["DefinitionInsertionFailed"] = error_types.get("DefinitionInsertionFailed", 0) + 1
                logger.error(f"Failed to insert definition for sense {sense_idx} of '{lemma}'. Skipping related data.")
                continue # Skip examples/references if definition failed

            stats["definitions_added"] += 1

            # --- Process Definition Examples ---
            # Examples structure: {"raw": "...", "examples": ["...", "..."]}
            examples_obj = sense.get("examples", {})
            if isinstance(examples_obj, dict):
                example_texts = []
                raw_example = examples_obj.get("raw")
                if raw_example and isinstance(raw_example, str) and raw_example.strip():
                    example_texts.append(raw_example.strip())

                # Append examples from the list if they exist
                example_list = examples_obj.get("examples", [])
                if isinstance(example_list, list):
                     for ex_item in example_list:
                          if ex_item and isinstance(ex_item, str) and ex_item.strip():
                               example_texts.append(ex_item.strip())

                for example_text in example_texts:
                    example_id = insert_definition_example(
                        cur,
                        definition_id,
                        {"text": example_text}, # Simple format for now
                        source_identifier,
                    )
                    if example_id:
                        stats["examples_added"] += 1

            # --- Process Definition References ---
            # Structure: ["reference1", "reference2"]
            references = sense.get("references", [])
            if isinstance(references, list):
                for ref_text in references:
                    if ref_text and isinstance(ref_text, str) and ref_text.strip():
                         # Add to relations batch (treat as related term?)
                        relations_batch.append(
                            {
                                "from_word": word_id,  # Changed from "word_id" to "from_word"
                                "to_word": ref_text.strip(),  # Changed from "related_word" to "to_word"
                                "relation_type": "related", # FIXED: Changed from rel_type to relation_type
                                "source": source_identifier, # FIXED: Changed from source_identifier to source
                                "metadata": {"context": "reference"},
                                "def_id": definition_id, # Link to this definition
                            }
                        )

            # --- Process Sense Domains ---
            # Add sense domains to definition tags or metadata if needed
            # sense_domains = sense.get("domains", [])
            # if sense_domains and isinstance(sense_domains, list):
            #     # Logic to add domains to definition record (e.g., update metadata JSONB)

    else:
        logger.warning(
            f"Expected list for 'senses' in '{lemma}', got {type(senses)}. Skipping definitions."
        )

    # --- Process Word Forms and Templates (if keys exist) ---
    # Add logic similar to Kaikki processor here if 'forms' or 'inflection_templates' keys are found
    # Example:
    # if "forms" in entry_data and isinstance(entry_data["forms"], list):
    #    for form_data in entry_data["forms"]:
    #        # Extract form_text, metadata
    #        # Call insert_word_form(cur, word_id, form_text, metadata, source_identifier)
    # if "inflection_templates" in entry_data and ...:
    #    # Call insert_word_template(...)

    # Note: No explicit return, function mutates stats, error_types, relations_batch
    # Exceptions are raised on critical errors to be caught by the caller


# Moved from dictionary_manager.py (originally around line 1311)
def process_tagalog_words(
    cur, filename: str, add_etymology: bool = True
) -> Dict:
    """
    Process a JSON file containing Tagalog words (tagalog-words.json format)
    using the provided cursor and iterative JSON parsing.
    Assumes transaction is managed by the caller.

    Args:
        cur: Active database cursor provided by the caller.
        filename: Path to the JSON file.
        add_etymology: Whether to add etymology data.

    Returns:
        Dictionary with processing statistics.
    """
    logger.info(f"Processing Tagalog words file: {filename} using provided cursor.")

    # --- Initialize Statistics ---
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "definitions_added": 0,
        "pronunciations_added": 0,
        "etymologies_added": 0,
        "examples_added": 0,
        "relations_processed": 0,
        "relations_inserted": 0,
        "relations_failed": 0,
    }
    error_types = {}
    word_id_cache = {} # Cache word IDs to reduce lookups during relation processing
    relations_batch = [] # Batch relations for later processing

    # --- Source Identifier ---
    raw_source_identifier = os.path.basename(filename)
    source_identifier = SourceStandardization.standardize_sources(raw_source_identifier)
    if not source_identifier: # Fallback
        source_identifier = "diksiyonaryo.ph" # Default name for this source
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    # --- Required Columns Check (Simplified for this processor) ---
    # Assuming standard schema, no need for column mapping/checking here unless customizing
    table_column_map = {} # Placeholder if needed by _process_single_tagalog_word_entry
    required_columns = {} # Placeholder if needed by _process_single_tagalog_word_entry

    # --- Load JSON Data ---
    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        stats["errors"] += 1
        error_types["FileNotFound"] = 1
        return stats
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        stats["errors"] += 1
        error_types["JSONDecodeError"] = 1
        return stats # Cannot proceed
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        stats["errors"] += 1
        error_types["FileReadError"] = 1
        return stats # Cannot proceed

    if not isinstance(data, dict):
        logger.error(f"File {filename} does not contain a top-level dictionary.")
        # Return stats indicating format error; caller should handle transaction
        stats["errors"] += 1
        error_types["InvalidFormat"] = 1
        return {**stats, "error_details": error_types}

    # --- Optional: Extract top-level metadata if needed ---
    dictionary_metadata = data.pop("__metadata__", {}) # Remove metadata if key exists
    entries_in_file = len(data) # Count actual word entries

    if entries_in_file == 0:
        logger.info(f"Found 0 word entries in {filename}. Skipping file.")
        # No commit needed, just return stats
        return stats # Return empty stats

    logger.info(f"Found {entries_in_file} word entries in {filename}")

    # --- Get connection for savepoints ---
    conn = cur.connection
    if conn.closed:
        logger.error("Connection is closed. Cannot proceed with processing.")
        stats["errors"] += 1
        error_types["ConnectionClosed"] = 1
        return {**stats, "error_details": error_types}

    # --- Process Entries ---
    iterator = data.items()
    pbar = None
    if tqdm:
        pbar = tqdm(iterator, total=entries_in_file, desc=f"Processing {source_identifier}", unit="entry", leave=False)

    try:
        # --- Iterate using data.items() directly, handle tqdm update manually ---
        for entry_index, (word_key, entry_data) in enumerate(data.items()):
            # --- Savepoint logic ---
            savepoint_name = f"tagalog_entry_{entry_index}"
            lemma_for_log = entry_data.get("word", word_key) # Get lemma for logging

            try:
                # Create savepoint for this entry
                cur.execute(f"SAVEPOINT {savepoint_name}")

                # Call the single entry processor
                _process_single_tagalog_word_entry(
                    cur,
                    entry_data,
                    word_key,
                    source_identifier,
                    "tl", # Hardcoded language code for this processor
                    add_etymology,
                    word_id_cache,
                    stats, # Pass stats dict for updates
                    error_types, # Pass error_types for updates
                    table_column_map,
                    relations_batch,
                    required_columns,
                )
                # If _process_single_tagalog_word_entry doesn't raise an exception, release savepoint
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                stats["processed"] += 1 # Increment processed count only on success

                # Commit every 100 entries to avoid transaction bloat
                if stats["processed"] % 100 == 0:
                    conn.commit()
                    logger.debug(f"Committed after processing {stats['processed']} entries")

            except Exception as entry_error:
                logger.error(f"Error processing entry \'{lemma_for_log}\' (key: {word_key}, index: {entry_index}): {entry_error}", exc_info=True)
                stats["errors"] += 1
                error_key = f"EntryProcessingError: {type(entry_error).__name__}"
                error_types[error_key] = error_types.get(error_key, 0) + 1

                # Rollback to the savepoint to discard changes for this entry
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                except Exception as rb_error:
                    logger.error(f"Failed to rollback/release savepoint {savepoint_name} for entry \'{lemma_for_log}\': {rb_error}")
                    # If we can't rollback to savepoint, try to rollback the whole transaction
                    try:
                        conn.rollback()
                        logger.info("Performed full transaction rollback after savepoint failure")
                    except Exception as full_rb_error:
                        logger.critical(f"CRITICAL: Failed both savepoint and full rollback: {full_rb_error}")
                        raise # Re-raise to stop processing

            finally:
                # --- Update tqdm manually inside the loop ---
                if pbar:
                    pbar.update(1)

        # Process the relations batch *after* iterating through all entries
        if relations_batch:
            try:
                process_relations_batch(cur, relations_batch, stats, word_id_cache)
                conn.commit()
                logger.info("Successfully processed and committed relations batch")
            except Exception as batch_error:
                logger.error(f"Error processing relations batch: {batch_error}", exc_info=True)
                error_types["RelationsBatchError"] = error_types.get("RelationsBatchError", 0) + 1
                try:
                    conn.rollback()
                    logger.info("Rolled back failed relations batch")
                except Exception as rb_error:
                    logger.critical(f"CRITICAL: Failed to rollback relations batch: {rb_error}")
                    raise # Re-raise to stop processing

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
            
    # --- Final Stats ---
    logger.info(f"Finished processing {source_identifier}. Processed: {stats['processed']}, Errors: {stats['errors']}, Skipped: {stats['skipped']}")
    if error_types:
        stats["error_details"] = error_types
    return stats