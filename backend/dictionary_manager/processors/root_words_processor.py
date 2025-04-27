#!/usr/bin/env python3
"""
Processor for tagalog.com Root Words JSON files (cleaned version).
"""

import json
import logging
import os
from typing import Tuple, Dict, List, Any, Optional, Iterator
import psycopg2
import psycopg2.extensions
from psycopg2.extras import Json

# Third-party imports
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # Optional dependency

# Project-specific imports (using absolute paths)
from backend.dictionary_manager.db_helpers import (
    get_or_create_word_id,
    insert_definition,
    insert_relation,
)
from backend.dictionary_manager.text_helpers import (
    SourceStandardization, # Use the class directly
    standardize_source_identifier, # Or the helper function if preferred
)
from backend.dictionary_manager.enums import RelationshipType

logger = logging.getLogger(__name__)

# Moved from dictionary_manager.py (originally around line 949)

def process_root_words_cleaned(cur, filename: str):
    """
    Processes entries from the tagalog.com Root Words JSON file (cleaned version).
    Handles JSON input that is either a list of root word objects
    or a dictionary mapping root words (str) to their details (dict).
    Manages transactions manually using savepoints for individual entry resilience.

    Args:
        cur: Database cursor
        filename: Path to the root words cleaned JSON file

    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing Root Words (tagalog.com cleaned) file: {filename}")
    stats = {
        "roots_processed": 0,
        "definitions_added": 0,
        "relations_added": 0,
        "associated_processed": 0,
        "errors": 0,
        "skipped": 0,
    }
    error_types = {}  # Track error types

    # Define source identifier, standardizing the filename
    source_identifier = standardize_source_identifier(os.path.basename(filename))
    if not source_identifier:  # Fallback if standardization fails unexpectedly
        source_identifier = "tagalog.com-RootWords-Cleaned"
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    conn = cur.connection  # Get the connection for savepoint management

    try:
        # Add errors='replace' to handle potential invalid UTF-8 bytes
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return stats  # Return stats; outer transaction might continue or rollback
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        stats["errors"] += 1
        error_types["JSONDecodeError"] = error_types.get("JSONDecodeError", 0) + 1
        # Raise exception to make the outer migrate_data function rollback
        raise RuntimeError(f"Invalid JSON in file {filename}: {e}") from e
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        stats["errors"] += 1
        error_types[f"FileReadError: {type(e).__name__}"] = (
            error_types.get(f"FileReadError: {type(e).__name__}", 0) + 1
        )
        # Raise exception to make the outer migrate_data function rollback
        raise RuntimeError(f"Error reading file {filename}: {e}") from e

    # --- Determine format and prepare iterator ---
    entries_iterator = None
    total_roots = 0
    if isinstance(data, list):  # Original list format
        entries_iterator = enumerate(data)
        total_roots = len(data)
        logger.info(
            f"Found {total_roots} root word entries in list format in {filename}"
        )
    elif isinstance(data, dict):  # New dictionary format {root_word: details_dict}
        # Create an iterator that yields (index, root_word, root_details)
        def dict_iterator(d):
            for i, (key, value) in enumerate(d.items()):
                yield i, key, value

        entries_iterator = dict_iterator(data)
        total_roots = len(data)
        logger.info(
            f"Found {total_roots} root word entries in dictionary format in {filename}"
        )
    else:
        logger.error(
            f"File {filename} does not contain a list or dictionary of root word entries."
        )
        stats["errors"] += 1
        error_types["InvalidTopLevelFormat"] = (
            error_types.get("InvalidTopLevelFormat", 0) + 1
        )
        raise TypeError(f"Invalid top-level format in {filename}")  # Raise error

    if total_roots == 0:
        logger.info(f"No root word entries found in {filename}. Nothing to process.")
        return stats  # Return empty stats

    # --- Process Entries ---
    pbar = None # Initialize pbar
    if tqdm:
        # Use total_roots for tqdm total
        pbar = tqdm(total=total_roots, desc=f"Processing {source_identifier}")

    try: # ADDED TRY for pbar cleanup
        for entry_item in entries_iterator:
            # Handle different formats
            if isinstance(data, dict):
                entry_index, root_word, root_details = entry_item
            else:  # list format
                entry_index, root_word_entry = entry_item
                root_word = (
                    root_word_entry.get("root_word", "").strip()
                    if isinstance(root_word_entry, dict)
                    else ""
                )
                root_details = root_word_entry

            savepoint_name = f"root_entry_{entry_index}"  # Create a savepoint name

            try:
                # Check if the transaction is already aborted before trying to create a savepoint
                if hasattr(conn, 'info') and conn.info and conn.info.transaction_status == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
                    # Transaction is already aborted, need to rollback the entire transaction
                    logger.error(f"Transaction is already in aborted state before processing entry {entry_index} ('{root_word}')")
                    conn.rollback()  # Rollback the entire transaction
                    logger.info(f"Rolled back the entire transaction due to previous abort")
                    # Re-raise with a clear message for the outer transaction
                    raise RuntimeError("Transaction was already aborted before processing entry, entire transaction rolled back")
                
                # Start savepoint for this entry
                cur.execute(f"SAVEPOINT {savepoint_name}")

                # Skip if root_word is empty
                if not root_word:
                    logger.warning(
                        f"Skipping entry at index {entry_index} with empty root word."
                    )
                    stats["skipped"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    pbar.update(1)
                    continue

                language_code = "tl"

                # --- Root Word Creation ---
                word_id = get_or_create_word_id(
                    cur,
                    root_word,
                    language_code=language_code,
                    source_identifier=source_identifier,
                    is_root_word=True,
                )
                if not word_id or word_id <= 0:
                    logger.error(f"Failed to get/create root word ID for '{root_word}'")
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    stats["errors"] += 1
                    error_types["RootWordCreationFailed"] = error_types.get("RootWordCreationFailed", 0) + 1
                    pbar.update(1)
                    continue
                    
                logger.debug(f"Processing root word '{root_word}' (ID: {word_id})")

                # --- Process definition for the root word itself ---
                if isinstance(data, dict):
                    # In dict format, root word details are in the details[root_word] if it exists
                    root_word_details = root_details.get(root_word, {})
                    if isinstance(root_word_details, dict):
                        definition_text = root_word_details.get(
                            "definition", ""
                        ).strip()
                        part_of_speech = (
                            root_word_details.get("type", "").strip() or None
                        )

                        if definition_text:
                            if definition_text.endswith("..."):
                                definition_text = definition_text[:-3].strip()

                            def_id = insert_definition(
                                cur,
                                word_id,
                                definition_text,
                                part_of_speech=part_of_speech,
                            )
                            if def_id:
                                stats["definitions_added"] += 1
                                logger.debug(
                                    f"Added definition for root '{root_word}': {definition_text[:50]}..."
                                )
                else:
                    # Process list format definitions
                    definitions = root_details.get("definitions", [])
                    if isinstance(definitions, list):
                        for def_idx, definition_item in enumerate(definitions):
                            definition_text = None
                            part_of_speech = None
                            examples_json = None
                            tags_str = None
                            usage_notes = None

                            if isinstance(definition_item, str):
                                definition_text = definition_item.strip()
                            elif isinstance(definition_item, dict):
                                definition_text = (
                                    definition_item.get("text", "").strip()
                                    or definition_item.get("definition", "").strip()
                                )
                                part_of_speech = (
                                    definition_item.get("pos")
                                    or definition_item.get("part_of_speech")
                                    or definition_item.get("type")
                                )
                                if part_of_speech:
                                    part_of_speech = part_of_speech.strip()

                            if definition_text and definition_text.endswith("..."):
                                definition_text = definition_text[:-3].strip()

                            if definition_text:
                                def_id = insert_definition(
                                    cur,
                                    word_id,
                                    definition_text,
                                    part_of_speech=part_of_speech,
                                )
                                if def_id:
                                    stats["definitions_added"] += 1

                # --- Process associated words ---
                if isinstance(data, dict):
                    # In dictionary format, each key other than the root word is an associated word
                    associated_words = {
                        k: v for k, v in root_details.items() if k != root_word
                    }

                    for assoc_idx, (assoc_word, assoc_details) in enumerate(
                        associated_words.items()
                    ):
                        if assoc_word and assoc_word != root_word:
                            stats["associated_processed"] += 1

                            # Create the associated word
                            assoc_word_id = get_or_create_word_id(
                                cur,
                                assoc_word,
                                language_code=language_code,
                                root_word_id=word_id,
                                source_identifier=source_identifier,
                            )

                            if assoc_word_id:
                                # Add DERIVED_FROM relationship (associated -> root)
                                rel_metadata = {
                                    "source": source_identifier,
                                    "confidence": 95,
                                }
                                rel_id = insert_relation(
                                    cur,
                                    assoc_word_id,
                                    word_id,
                                    RelationshipType.DERIVED_FROM.value,  # Use .value
                                    source_identifier=source_identifier,
                                    metadata=rel_metadata
                                )
                                # Add inverse ROOT_OF relationship (root -> assoc)
                                inverse_rel_metadata = {
                                    "source": source_identifier,
                                    "confidence": 95,
                                }
                                inverse_rel_id = insert_relation(
                                    cur,
                                    word_id,
                                    assoc_word_id,
                                    RelationshipType.ROOT_OF.value,  # Use .value
                                    source_identifier=source_identifier,
                                    metadata=inverse_rel_metadata
                                )
                                if rel_id:
                                    stats["relations_added"] += 1
                                    logger.debug(
                                        f"Added relation: '{assoc_word}' DERIVED_FROM '{root_word}'"
                                    )

                                # Add definition for the associated word
                                if isinstance(assoc_details, dict):
                                    assoc_def_text = assoc_details.get(
                                        "definition", ""
                                    ).strip()
                                    if assoc_def_text and assoc_def_text.endswith(
                                        "..."
                                    ):
                                        assoc_def_text = assoc_def_text[:-3].strip()

                                    assoc_pos = (
                                        assoc_details.get("type", "").strip() or None
                                    )

                                    if assoc_def_text:
                                        assoc_def_id = insert_definition(
                                            cur,
                                            assoc_word_id,
                                            assoc_def_text,
                                            part_of_speech=assoc_pos,
                                        )
                                        if assoc_def_id:
                                            stats["definitions_added"] += 1
                                            logger.debug(
                                                f"Added definition for associated word '{assoc_word}': {assoc_def_text[:50]}..."
                                            )
                else:
                    # Process original list format associated words
                    associated_data = root_details.get("associated_words", {})
                    associated_items_iterator = None

                    if isinstance(associated_data, dict):

                        def assoc_dict_iterator(d):
                            for i, (key, value) in enumerate(d.items()):
                                if isinstance(value, dict):
                                    yield i, key.strip(), value
                                else:
                                    logger.warning(
                                        f"Skipping non-dict value for associated word key '{key}' in root '{root_word}'"
                                    )

                        associated_items_iterator = assoc_dict_iterator(associated_data)
                    elif isinstance(associated_data, list):

                        def assoc_list_iterator(l):
                            for i, item in enumerate(l):
                                word_str, details = None, {}
                                if isinstance(item, str):
                                    word_str = item.strip()
                                elif isinstance(item, dict):
                                    word_str = item.get("word", "").strip()
                                    details = item
                                if word_str:
                                    yield i, word_str, details
                                else:
                                    logger.warning(
                                        f"Skipping invalid associated word item at index {i} in root '{root_word}' (list format)"
                                    )

                        associated_items_iterator = assoc_list_iterator(associated_data)
                    else:
                        logger.warning(
                            f"Unexpected format for 'associated_words' for root '{root_word}': {type(associated_data)}. Skipping."
                        )
                        associated_items_iterator = iter([])

                    if associated_data:
                        processed_assoc_count_for_root = 0
                        for (
                            assoc_idx,
                            assoc_word,
                            assoc_details,
                        ) in associated_items_iterator:
                            if assoc_word and assoc_word != root_word:
                                stats["associated_processed"] += 1
                                processed_assoc_count_for_root += 1

                                assoc_word_id = get_or_create_word_id(
                                    cur,
                                    assoc_word,
                                    language_code=language_code,
                                    root_word_id=word_id,
                                    source_identifier=source_identifier,
                                )

                                if assoc_word_id:
                                    # Add DERIVED_FROM relationship
                                    rel_metadata = {
                                        "source": source_identifier,
                                        "index": assoc_idx,
                                        "confidence": 95,
                                    }
                                    rel_id = insert_relation(
                                        cur,
                                        assoc_word_id,
                                        word_id,
                                        RelationshipType.DERIVED_FROM.value, # Use Enum value
                                        source_identifier=source_identifier,
                                        metadata=rel_metadata
                                    )
                                    if rel_id:
                                        stats["relations_added"] += 1
                                        logger.debug(
                                            f"Added relation: '{assoc_word}' DERIVED_FROM '{root_word}'"
                                        )
                                    # FIX 2: Add inverse ROOT_OF relationship (root -> assoc)
                                    inverse_rel_metadata = { # Re-add metadata dictionary
                                        "source": source_identifier,
                                        "index": assoc_idx,
                                        "confidence": 95,
                                    }
                                    inverse_rel_id = insert_relation(
                                        cur,
                                        word_id,
                                        assoc_word_id,
                                        RelationshipType.ROOT_OF.value, # Use Enum value
                                        source_identifier=source_identifier,
                                        metadata=inverse_rel_metadata
                                    )
                                    # Process definition for associated word
                                    assoc_def_text = assoc_details.get(
                                        "definition", ""
                                    ).strip()
                                    if assoc_def_text and assoc_def_text.endswith(
                                        "..."
                                    ):
                                        assoc_def_text = assoc_def_text[:-3].strip()
                                    assoc_pos = (
                                        assoc_details.get("type", "").strip() or None
                                    )

                                    if assoc_def_text:
                                        assoc_def_id = insert_definition(
                                            cur,
                                            assoc_word_id,
                                            assoc_def_text,
                                            part_of_speech=assoc_pos,
                                        )
                                        if assoc_def_id:
                                            stats["definitions_added"] += 1
                                            logger.debug(
                                                f"Added definition for associated word '{assoc_word}': {assoc_def_text[:50]}..."
                                            )

                        if processed_assoc_count_for_root > 0:
                            logger.debug(
                                f"Root '{root_word}': Processed {processed_assoc_count_for_root} associated words."
                            )

                # --- Finish Entry Processing Successfully ---
                cur.execute(
                    f"RELEASE SAVEPOINT {savepoint_name}"
                )  # Commit the entry's changes
                stats["roots_processed"] += 1  # Increment successful count

            except Exception as e:
                logger.error(
                    f"Error processing entry {entry_index} ('{root_word}'): {e}", exc_info=True
                )
                stats["errors"] += 1
                error_key = f"EntryProcessingError: {type(e).__name__}"
                error_types[error_key] = error_types.get(error_key, 0) + 1
                
                # Check if the transaction is already aborted
                transaction_aborted = hasattr(conn, 'info') and conn.info and conn.info.transaction_status == psycopg2.extensions.TRANSACTION_STATUS_INERROR
                
                if transaction_aborted:
                    logger.error("Transaction already aborted, rolling back entire transaction")
                    try:
                        conn.rollback()  # Roll back the entire transaction
                        logger.info("Successfully rolled back entire transaction")
                        # Re-raise a clear error to signal the outer transaction handler
                        raise RuntimeError("Transaction aborted and rolled back") from e
                    except Exception as rb_error:
                        logger.critical(f"Failed to rollback transaction: {rb_error}")
                        raise RuntimeError(f"Critical database state - failed to rollback aborted transaction: {rb_error}") from e
                else:
                    # Transaction is not aborted, try rolling back to savepoint
                    try:
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        logger.info(f"Successfully rolled back to savepoint {savepoint_name}")
                        try:
                            cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                            logger.debug(f"Released savepoint {savepoint_name}")
                        except Exception as rel_error:
                            logger.warning(f"Error releasing savepoint {savepoint_name}: {rel_error}")
                            # Not fatal if we can't release the savepoint
                    except psycopg2.errors.InvalidSavepointSpecification as sv_error:
                        logger.error(f"Savepoint {savepoint_name} does not exist: {sv_error}")
                        # If the savepoint doesn't exist but transaction is not aborted, 
                        # we can still continue with the next entry
                    except Exception as rb_error:
                        logger.error(f"Error during savepoint rollback: {rb_error}")
                        # If we can't rollback to the savepoint but the transaction isn't
                        # aborted yet, we can try to continue with the next entry
            
            finally:
                if pbar: # Update pbar even if entry fails
                    pbar.update(1)
                
        # After the loop, check if the transaction is aborted
        if hasattr(conn, 'info') and conn.info and conn.info.transaction_status == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
            logger.error("Transaction is in aborted state after processing all entries")
            try:
                conn.rollback()
                logger.info("Successfully rolled back transaction after processing")
                raise RuntimeError("Transaction was aborted during processing, rolled back")
            except Exception as rb_error:
                logger.critical(f"Failed to rollback transaction after processing: {rb_error}")
                raise RuntimeError(f"Critical database state - failed to rollback aborted transaction: {rb_error}")

    finally: # ADDED FINALLY for pbar cleanup
        if pbar:
            pbar.close()

    # No final commit needed here - the outer migrate_data function handles the commit for the entire source file

    if error_types:
        logger.warning(
            f"Error summary for {filename}: {json.dumps(error_types, indent=2)}"
        )
    if stats["roots_processed"] == 0 and total_roots > 0:
        logger.warning(
            f"No root entries were successfully processed from {filename} despite finding {total_roots} entries."
        )

    # <<< Log the final statistics >>>
    logger.info(f"Statistics for {filename}: {json.dumps(stats, indent=2)}")

    # Return the statistics gathered during processing
    return stats

