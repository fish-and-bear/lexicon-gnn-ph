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
    get_standardized_pos_id,
    with_transaction,
)
from backend.dictionary_manager.text_helpers import normalize_lemma, SourceStandardization

logger = logging.getLogger(__name__)

# Moved from dictionary_manager.py (originally around line 97 in old structure)
@with_transaction(commit=True)
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
        word_metadata=Json(word_creation_metadata) if word_creation_metadata else None,  # Wrap collected metadata
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
            cur, word_id, main_pron.strip(), "ipa", source_identifier=source_identifier
        )
        if pron_id:
            stats["pronunciations_added"] += 1

    if alt_pron and isinstance(alt_pron, str) and alt_pron.strip():
        # Treat alternate as a separate entry, maybe tag it?
        pron_id = insert_pronunciation(
            cur, word_id, alt_pron.strip(), "ipa", tags=["alternate"], source_identifier=source_identifier
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
                            "word_id": word_id,
                            "related_word": related_word.strip(),
                            "rel_type": "derived_from",  # Assume terms are sources
                            "source_identifier": source_identifier,
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
                "word_id": word_id,  # Current word
                "related_word": derivative.strip(),  # The derived word
                "rel_type": "root_of",  # Current word is root_of the derivative
                "source_identifier": source_identifier,
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

            # --- Insert Definition ---
            # Extract other sense data for definition insertion
            sense_notes = sense.get("notes") # Could be list or string
            usage_notes_str = None
            if isinstance(sense_notes, list):
                 usage_notes_str = "; ".join(str(n) for n in sense_notes if n)
            elif isinstance(sense_notes, str):
                 usage_notes_str = sense_notes.strip()

            # Pass sense data to insert_definition
            definition_id = insert_definition(
                cur,
                word_id,
                definition_text,
                original_pos=pos_code_to_use,
                standardized_pos_id=standardized_pos_id,
                usage_notes=usage_notes_str,
                source_identifier=source_identifier,
                # tags=None, # Can add sense-level tags if needed
                metadata=None, # Can add sense-level metadata if needed
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
                                 "word_id": word_id,
                                 "related_word": ref_text.strip(),
                                 "rel_type": "related", # Or maybe cross_reference?
                                 "source_identifier": source_identifier,
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
        if not isinstance(data, dict):
            logger.error(f"File {filename} does not contain a top-level dictionary.")
            stats["errors"] += 1
            error_types["InvalidFileFormat"] = 1
            return stats
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

    num_entries = len(data)
    logger.info(f"Found {num_entries} entries in {filename}")

    # --- Define Inline Relation Batch Processing Function ---
    def _process_relations_batch_inline(
        cur, batch, stats_dict, cache, error_types_dict
    ):
        """Processes a batch of relation data within the main loop's transaction."""
        inserted_count = 0
        failed_count = 0
        processed_count = len(batch)
        logger.info(f"Processing batch of {processed_count} potential relations...")

        # Pre-fetch IDs for related words if not already cached
        related_words_to_fetch = set()
        for rel in batch:
            related_lemma = rel.get("related_word")
            rel_lang = rel.get("metadata", {}).get("lang", "tl") # Default to 'tl' if lang missing
            normalized_rel_lemma = normalize_lemma(related_lemma)
            cache_key = f"{normalized_rel_lemma}|{rel_lang}"
            if cache_key not in cache:
                related_words_to_fetch.add((related_lemma, rel_lang))

        if related_words_to_fetch:
            logger.info(f"Fetching/Creating IDs for {len(related_words_to_fetch)} related words...")
            # This assumes get_or_create_word_id handles transactions correctly internally
            # or runs within the same transaction provided by `cur`
            for rel_lemma, rel_lang in related_words_to_fetch:
                 try:
                      # Get/create the related word ID using the same source identifier
                      rel_id = get_or_create_word_id(cur, rel_lemma, rel_lang, source_identifier=rel["source_identifier"])
                      normalized_rel_lemma_cache = normalize_lemma(rel_lemma)
                      cache_key = f"{normalized_rel_lemma_cache}|{rel_lang}"
                      cache[cache_key] = rel_id # Cache the result
                 except Exception as e:
                      logger.error(f"Failed to get/create ID for related word '{rel_lemma}' ({rel_lang}): {e}")
                      error_types_dict["RelatedWordIDError"] = error_types_dict.get("RelatedWordIDError", 0) + 1
                      # Add to cache with None to prevent repeated attempts for this word in this batch
                      normalized_rel_lemma_cache = normalize_lemma(rel_lemma)
                      cache_key = f"{normalized_rel_lemma_cache}|{rel_lang}"
                      cache[cache_key] = None


        # Process each relation in the batch
        for rel_data in batch:
            from_word_id = rel_data.get("word_id")
            related_lemma = rel_data.get("related_word")
            rel_type = rel_data.get("rel_type")
            rel_source = rel_data.get("source_identifier")
            rel_lang = rel_data.get("metadata", {}).get("lang", "tl") # Default to 'tl'

            if not all([from_word_id, related_lemma, rel_type, rel_source]):
                logger.warning(f"Skipping relation due to missing data: {rel_data}")
                failed_count += 1
                continue

            # Get cached target word ID
            normalized_rel_lemma = normalize_lemma(related_lemma)
            cache_key = f"{normalized_rel_lemma}|{rel_lang}"
            to_word_id = cache.get(cache_key)

            if to_word_id is None:
                # ID wasn't fetched or creation failed previously
                logger.warning(f"Skipping relation to '{related_lemma}' ({rel_lang}) - target word ID not found/created.")
                failed_count += 1
                continue

            # Insert the relation
            # Assuming insert_relation handles transactions appropriately (e.g., using @with_transaction(commit=True))
            # or runs within the provided cursor's transaction context.
            try:
                relation_id = insert_relation(
                    cur, from_word_id, to_word_id, rel_type, rel_source
                )
                if relation_id:
                    inserted_count += 1
                else:
                    # Error logged by insert_relation
                    failed_count += 1
                    error_types_dict["RelationInsertionFailed"] = error_types_dict.get("RelationInsertionFailed", 0) + 1
            except Exception as e:
                 logger.error(f"Unexpected error inserting relation {from_word_id}->{to_word_id} ({rel_type}): {e}")
                 failed_count += 1
                 error_types_dict["RelationInsertionError"] = error_types_dict.get("RelationInsertionError", 0) + 1

        logger.info(f"Relation batch processing finished: {inserted_count} inserted, {failed_count} failed.")
        stats_dict["relations_processed"] += processed_count
        stats_dict["relations_inserted"] += inserted_count
        stats_dict["relations_failed"] += failed_count
        return # Modify stats in place

    # --- Process Entries ---
    if tqdm:
        pbar = tqdm(data.items(), total=num_entries, desc=f"Processing {source_identifier}", unit="word")
        iterator = pbar
    else:
        iterator = data.items()

    # Main processing loop wrapped in try...finally for pbar cleanup
    try:
        for word_key, entry_data in iterator:
            if not isinstance(entry_data, dict):
                logger.warning(f"Skipping non-dictionary entry with key: {word_key}")
                stats["skipped"] += 1
                continue

            savepoint_name = f"tagalog_word_{stats['processed']}"
            try:
                # Create savepoint for the entry
                cur.execute(f"SAVEPOINT {savepoint_name}")

                # Process the single entry using the helper function
                _process_single_tagalog_word_entry(
                    cur,
                    entry_data,
                    word_key,
                    source_identifier,
                    "tl", # Hardcoded language code for this processor
                    add_etymology,
                    word_id_cache,
                    stats,
                    error_types,
                    table_column_map, # Pass placeholders
                    relations_batch, # Pass batch list
                    required_columns, # Pass placeholders
                )
                stats["processed"] += 1
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Commit entry changes

            except Exception as e:
                logger.error(
                    f"Error processing entry (key: {word_key}): {e}", exc_info=True
                )
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    logger.info(f"Rolled back changes for entry (key: {word_key})")
                except Exception as rb_err:
                    logger.error(f"Failed to rollback to savepoint {savepoint_name}: {rb_err}")
                    # If rollback fails, the main transaction might be compromised
                    # Returning stats reflects current state, but DB might be inconsistent
                    return stats
                stats["errors"] += 1
                # Log specific error type if possible
                error_type = type(e).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
            finally:
                 # Clean up released savepoint just in case (no harm if already released/rolled back)
                 try: cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                 except: pass

        # Process any remaining relations
        if relations_batch:
            _process_relations_batch_inline(cur, relations_batch, stats, word_id_cache, error_types)
            # print(f"Processed final relation batch {len(relations_batch)} items")

    finally: # Corresponds to the try block starting before the loop
        if pbar:
            pbar.close()

    # Calculate total issues
    total_issues = stats["errors"] + stats["skipped"]

    # --- Final Logging ---
    logger.info(f"Finished processing {filename}: {stats['processed']} processed, {stats['skipped']} skipped, {stats['errors']} errors.")
    if stats["errors"] > 0:
        logger.warning(f"Error summary for {filename}: {error_types}")
    logger.info(f"Stats for {filename}: {stats}")

    return stats 