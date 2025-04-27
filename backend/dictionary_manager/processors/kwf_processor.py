#!/usr/bin/env python3
"""
Processor for Komisyon sa Wikang Filipino (KWF) Dictionary JSON files.
"""

import json
import logging
import os
from typing import Tuple, Dict, Any, Optional

# Third-party imports
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
    insert_relation,
    get_standardized_pos_id, # Needed for mapping POS
    get_uncategorized_pos_id, # Fallback POS ID
)
from backend.dictionary_manager.text_helpers import (
    SourceStandardization,
    clean_html,
    get_standard_code, # Needed for standardizing POS
    preserve_acronyms_in_parentheses, # For handling lemma
    restore_preserved_acronyms, # For handling lemma
)
from backend.dictionary_manager.enums import RelationshipType
from psycopg2.extras import Json # For wrapping JSON data

logger = logging.getLogger(__name__)

# Moved from dictionary_manager.py (originally around line 87)
# Note: This function originally had @with_transaction(commit=False)
# If transaction management is handled by the caller (e.g., migrate_data),
# this decorator might not be needed here.
def process_kwf_dictionary(cur, filename: str) -> Tuple[int, int]:
    """
    Processes entries from the KWF Dictionary JSON file (dictionary format).
    Handles complex nested structure, HTML cleaning, and relations.
    Manages transactions manually using savepoints for individual entry resilience.

    Args:
        cur: Database cursor
        filename: Path to the kwf_dictionary.json file

    Returns:
        Tuple: (number_of_entries_processed_successfully, number_of_entries_with_issues)
    """
    # Standardize source identifier consistently
    raw_source_identifier = os.path.basename(filename)
    source_identifier = SourceStandardization.standardize_sources(raw_source_identifier)
    if not source_identifier:  # Fallback
        source_identifier = "KWF Diksiyonaryo ng Wikang Filipino"

    logger.info(f"Processing KWF Dictionary: {filename}")
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    conn = cur.connection  # Get the connection for savepoint management

    # Statistics tracking for this file
    stats = {
        "processed": 0,
        "definitions": 0,
        "relations": 0,
        "synonyms": 0,
        "antonyms": 0,
        "related_terms": 0,
        "cross_refs": 0,
        "etymologies": 0,
        "pronunciations": 0,
        "examples": 0,
        "idioms": 0,
        "skipped_invalid": 0,  # Entries skipped due to format issues
        "errors": 0,  # Entries skipped due to processing errors
    }
    error_types = {}  # Track specific error types encountered

    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return 0, 1  # 0 processed, 1 issue (file error)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        raise RuntimeError(f"Invalid JSON in file {filename}: {e}") from e
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        raise RuntimeError(f"Error reading file {filename}: {e}") from e

    if not isinstance(data, dict):
        logger.error(f"File {filename} does not contain a top-level dictionary.")
        return 0, 1

    # --- Optional: Extract top-level metadata if needed ---
    dictionary_metadata = data.pop("__metadata__", {})  # Remove metadata if key exists
    entries_in_file = len(data)  # Count actual word entries

    if entries_in_file == 0:
        logger.info(f"Found 0 word entries in {filename}. Skipping file.")
        return 0, 0

    logger.info(f"Found {entries_in_file} word entries in {filename}")

    # --- Process Entries ---
    iterator = data.items()
    pbar = None
    if tqdm:
        pbar = tqdm(iterator, total=entries_in_file, desc=f"Processing {source_identifier}", unit="entry", leave=False)
        iterator = pbar

    try:
        for entry_index, (original_key, entry) in enumerate(iterator):
            savepoint_name = f"kwf_entry_{entry_index}"
            lemma = ""  # Initialize for error logging
            word_id = None

            try:
                cur.execute(f"SAVEPOINT {savepoint_name}")

                if not isinstance(entry, dict):
                    logger.warning(f"Skipping non-dictionary value for key '{original_key}' at index {entry_index}")
                    stats["skipped_invalid"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    if pbar: pbar.update(1)
                    continue

                # Use formatted headword if available, otherwise the original key
                lemma = entry.get("formatted", original_key).strip()
                if not lemma:
                    logger.warning(f"Skipping entry at index {entry_index} (original key: '{original_key}') due to missing/empty 'formatted' or key.")
                    stats["skipped_invalid"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    if pbar: pbar.update(1)
                    continue

                original_lemma = lemma # Save for logging

                # Preserve acronyms in parentheses before any other processing
                lemma_with_markers = preserve_acronyms_in_parentheses(lemma)

                # Default language code for KWF is Tagalog
                language_code = "tl"

                # --- Extract word-level metadata --- (Copied from original)
                entry_metadata = entry.get("metadata", {})
                word_metadata_to_store = {}
                if isinstance(entry_metadata, dict):
                    source_langs = entry_metadata.get("source_language", [])
                    if isinstance(source_langs, list) and source_langs:
                        first_lang_info = source_langs[0]
                        if isinstance(first_lang_info, dict) and "value" in first_lang_info:
                            word_metadata_to_store["source_language"] = first_lang_info["value"]
                        elif isinstance(first_lang_info, str):
                            word_metadata_to_store["source_language"] = first_lang_info
                    elif isinstance(source_langs, str):
                        word_metadata_to_store["source_language"] = source_langs
                word_metadata_to_store["original_key"] = original_key
                word_metadata_to_store["kwf_original"] = entry.get("original", original_key)

                # --- Get or Create Word --- (Copied from original)
                try:
                    word_id = get_or_create_word_id(
                        cur,
                        lemma=lemma_with_markers, # Use the version with markers
                        language_code=language_code,
                        source_identifier=source_identifier,
                        word_metadata=(Json(word_metadata_to_store) if word_metadata_to_store else None),
                    )
                    if not word_id:
                        raise ValueError("get_or_create_word_id returned None")

                    restored_lemma = restore_preserved_acronyms(lemma_with_markers)
                    if restored_lemma != lemma_with_markers:
                        cur.execute("UPDATE words SET lemma = %s WHERE id = %s", (restored_lemma, word_id))
                        logger.debug(f"Restored acronyms in lemma: '{lemma}' -> '{restored_lemma}'")

                    logger.debug(f"Word '{lemma}' ({language_code}) -> ID: {word_id}")
                except Exception as word_err:
                    logger.error(f"CRITICAL FAILURE creating word '{lemma}' (original key: '{original_key}'): {word_err}")
                    raise word_err # Re-raise critical error

                # --- Process word-level Pronunciation, Etymology, Cross-refs from Metadata --- (Copied from original)
                if isinstance(entry_metadata, dict):
                    try:
                        pronunciations = entry_metadata.get("pronunciation", [])
                        if isinstance(pronunciations, list):
                            for pron_data in pronunciations:
                                pron_value = None; pron_type = "kwf_raw"; pron_tags = None; pron_meta = None
                                if isinstance(pron_data, str) and pron_data.strip(): pron_value = pron_data.strip()
                                elif isinstance(pron_data, dict) and pron_data.get("value"): pron_value = pron_data["value"].strip(); pron_type = pron_data.get("type", pron_type).strip().lower()
                                else: continue
                                if pron_value:
                                    pron_inserted_id = insert_pronunciation(cur, word_id=word_id, pronunciation_type=pron_type, value=pron_value, tags=pron_tags, metadata=pron_meta, source_identifier=source_identifier)
                                    if pron_inserted_id: stats["pronunciations"] += 1
                    except Exception as pron_err:
                        logger.warning(f"Error processing metadata pronunciations for '{lemma}' (ID: {word_id}): {pron_err}")
                        error_key = f"PronunciationInsertError: {type(pron_err).__name__}"; error_types[error_key] = error_types.get(error_key, 0) + 1
                    try:
                        etymologies = entry_metadata.get("etymology", [])
                        if isinstance(etymologies, list):
                            for ety_data in etymologies:
                                ety_text = None
                                if isinstance(ety_data, str) and ety_data.strip(): ety_text = ety_data.strip()
                                elif isinstance(ety_data, dict) and ety_data.get("value"): ety_text = ety_data["value"].strip()
                                else: continue
                                if ety_text:
                                    ety_id = insert_etymology(cur, word_id, ety_text, source_identifier)
                                    if ety_id: stats["etymologies"] += 1
                    except Exception as ety_err:
                        logger.warning(f"Error processing metadata etymology for '{lemma}' (ID: {word_id}): {ety_err}")
                        error_key = f"EtymologyInsertError: {type(ety_err).__name__}"; error_types[error_key] = error_types.get(error_key, 0) + 1
                    try:
                        cross_refs = entry_metadata.get("cross_references", [])
                        if isinstance(cross_refs, list):
                            for ref_item in cross_refs:
                                ref_word = None
                                if isinstance(ref_item, dict) and ref_item.get("term"): ref_word = ref_item["term"].strip()
                                elif isinstance(ref_item, str): ref_word = ref_item.strip()
                                if ref_word and ref_word.lower() != lemma.lower():
                                    ref_id = get_or_create_word_id(cur, ref_word, language_code, source_identifier)
                                    if ref_id and ref_id != word_id:
                                        rel_id_1 = insert_relation(cur, word_id, ref_id, RelationshipType.SEE_ALSO.value, source_identifier)
                                        if rel_id_1: stats["relations"] += 1; stats["cross_refs"] += 1
                    except Exception as ref_err:
                        logger.warning(f"Error processing metadata cross-references for '{lemma}' (ID: {word_id}): {ref_err}")
                        error_key = f"RelationInsertError: {type(ref_err).__name__}"; error_types[error_key] = error_types.get(error_key, 0) + 1

                # --- Process Definitions by POS --- (Copied from original)
                definitions_by_pos = entry.get("definitions", {})
                if isinstance(definitions_by_pos, dict):
                    for raw_pos, definitions_list in definitions_by_pos.items():
                        if not isinstance(definitions_list, list): logger.debug(f"Skipping invalid definitions list for POS '{raw_pos}' in word '{lemma}'"); continue
                        standardized_pos_code = get_standard_code(raw_pos); pos_id = None
                        try:
                            # Use get_standardized_pos_id directly to get the ID
                            pos_id = get_standardized_pos_id(cur, raw_pos)
                            if not pos_id:
                                 logger.warning(f"Failed to get standardized POS ID for '{raw_pos}' (Word: '{lemma}'). Using 'unc'.")
                                 pos_id = get_uncategorized_pos_id(cur)
                        except Exception as pos_err:
                            logger.warning(f"Error getting POS ID for '{raw_pos}' (Word: '{lemma}'): {pos_err}. Using 'unc'.")
                            try:
                                pos_id = get_uncategorized_pos_id(cur)
                            except Exception as unc_err:
                                logger.error(f"CRITICAL: Failed to get even 'unc' POS ID: {unc_err}. Skipping definitions for '{raw_pos}' of '{lemma}'.")
                                continue # Skip definitions for this POS

                        if pos_id is None:
                            logger.error(f"Critical error: pos_id is None after fallbacks for POS '{raw_pos}' of '{lemma}'. Skipping definitions.")
                            continue

                        for def_idx, def_item in enumerate(definitions_list):
                            if not isinstance(def_item, dict): logger.debug(f"Skipping invalid definition item at index {def_idx} for POS '{raw_pos}', word '{lemma}'"); continue
                            definition_text_raw = def_item.get("meaning", ""); definition_text = clean_html(definition_text_raw)
                            if not definition_text: continue
                            categories = def_item.get("categories", []); usage_note_text = clean_html(def_item.get("note"))
                            def_tags_list = [c for c in categories if isinstance(c, str) and c.strip()] # Store as list
                            # Process examples
                            examples_processed = []
                            example_sets = def_item.get("example_sets", [])
                            if isinstance(example_sets, list):
                                 for ex_set in example_sets:
                                     if isinstance(ex_set, dict) and "examples" in ex_set and isinstance(ex_set["examples"], list):
                                         for ex_data in ex_set["examples"]:
                                             if isinstance(ex_data, dict) and "text" in ex_data:
                                                 ex_text_clean = clean_html(ex_data["text"])
                                                 if ex_text_clean:
                                                     ex_obj = {"text": ex_text_clean}
                                                     if ex_set.get("label"): ex_obj["label"] = ex_set["label"]
                                                     examples_processed.append(ex_obj)
                            examples_json_str = json.dumps(examples_processed) if examples_processed else None
                            if examples_processed: stats["examples"] += len(examples_processed)

                            definition_id = None
                            # Pre-serialize metadata to string to avoid TypeError in db_helpers.py
                            metadata_dict = {"examples": examples_processed} if examples_processed else None
                            metadata_arg = json.dumps(metadata_dict) if metadata_dict else None
                            try:
                                # Updated to match current insert_definition signature
                                def_id = insert_definition(
                                    cur,
                                    word_id=word_id,
                                    definition_text=definition_text,
                                    part_of_speech=raw_pos,  # Changed from original_pos
                                    usage_notes=usage_note_text,
                                    tags=def_tags_list,
                                    metadata=metadata_arg, # Pass pre-serialized string or None
                                    sources=source_identifier
                                )
                                if def_id:
                                    stats["definitions"] += 1; definition_id = def_id
                                else:
                                    logger.warning(f"insert_definition failed for '{lemma}', POS '{raw_pos}', Def {def_idx+1}")
                                    error_key = f"DefinitionInsertFailure"; error_types[error_key] = error_types.get(error_key, 0) + 1
                            except Exception as def_err:
                                logger.error(f"Error inserting definition for '{lemma}', POS '{raw_pos}', Def {def_idx+1}: {def_err}", exc_info=True)
                                error_key = f"DefinitionInsertError: {type(def_err).__name__}"; error_types[error_key] = error_types.get(error_key, 0) + 1
                                continue # Skip relations if definition failed

                            if definition_id:
                                synonyms_list = def_item.get("synonyms", []); antonyms_list = def_item.get("antonyms", []); def_cross_refs = def_item.get("cross_references", [])
                                # Process Synonyms
                                if isinstance(synonyms_list, list):
                                    for syn_item in synonyms_list:
                                        syn_word = None; rel_type = RelationshipType.SYNONYM
                                        if isinstance(syn_item, str): syn_word = syn_item
                                        elif isinstance(syn_item, dict) and "term" in syn_item: syn_word = syn_item["term"]
                                        syn_word_clean = clean_html(syn_word)
                                        if syn_word_clean and syn_word_clean.lower() != lemma.lower():
                                            try:
                                                target_id = get_or_create_word_id(cur, syn_word_clean, language_code, source_identifier)
                                                if target_id and target_id != word_id:
                                                    r1 = insert_relation(cur, word_id, target_id, rel_type.value, source_identifier, metadata={"definition_id": definition_id})
                                                    r2 = insert_relation(cur, target_id, word_id, rel_type.value, source_identifier, metadata={"definition_id": definition_id})
                                                    if r1: stats["relations"] += 1; stats["synonyms"] += 1
                                            except Exception as e: logger.warning(f"Error creating synonym relation for '{lemma}': {e}")
                                # Process Antonyms
                                if isinstance(antonyms_list, list):
                                    for ant_item in antonyms_list:
                                        ant_word = None; rel_type = RelationshipType.ANTONYM
                                        if isinstance(ant_item, str): ant_word = ant_item
                                        elif isinstance(ant_item, dict) and "term" in ant_item: ant_word = ant_item["term"]
                                        ant_word_clean = clean_html(ant_word)
                                        if ant_word_clean and ant_word_clean.lower() != lemma.lower():
                                            try:
                                                target_id = get_or_create_word_id(cur, ant_word_clean, language_code, source_identifier)
                                                if target_id and target_id != word_id:
                                                    r1 = insert_relation(cur, word_id, target_id, rel_type.value, source_identifier, metadata={"definition_id": definition_id})
                                                    r2 = insert_relation(cur, target_id, word_id, rel_type.value, source_identifier, metadata={"definition_id": definition_id})
                                                    if r1: stats["relations"] += 1; stats["antonyms"] += 1
                                            except Exception as e: logger.warning(f"Error creating antonym relation for '{lemma}': {e}")
                                # Process Cross-references
                                if isinstance(def_cross_refs, list):
                                    for ref_item in def_cross_refs:
                                        ref_word = None; rel_type = RelationshipType.SEE_ALSO
                                        if isinstance(ref_item, dict) and ref_item.get("term"): ref_word = ref_item["term"]
                                        elif isinstance(ref_item, str): ref_word = ref_item
                                        ref_word_clean = clean_html(ref_word)
                                        if ref_word_clean and ref_word_clean.lower() != lemma.lower():
                                            try:
                                                target_id = get_or_create_word_id(cur, ref_word_clean, language_code, source_identifier)
                                                if target_id and target_id != word_id:
                                                    r1 = insert_relation(cur, word_id, target_id, rel_type.value, source_identifier, metadata={"definition_id": definition_id})
                                                    if r1: stats["relations"] += 1; stats["cross_refs"] += 1
                                            except Exception as e: logger.warning(f"Error creating cross-reference relation for '{lemma}': {e}")

                # --- Process Top-Level Related Terms --- (Copied from original)
                related_block = entry.get("related", {})
                if isinstance(related_block, dict):
                    related_terms = related_block.get("related_terms", []); top_antonyms = related_block.get("antonyms", [])
                    # Process Related Terms
                    if isinstance(related_terms, list):
                        for rel_item in related_terms:
                            rel_word = None; rel_type = RelationshipType.RELATED
                            if isinstance(rel_item, dict) and "term" in rel_item: rel_word = rel_item["term"]
                            elif isinstance(rel_item, str): rel_word = rel_item
                            rel_word_clean = clean_html(rel_word)
                            if rel_word_clean and rel_word_clean.lower() != lemma.lower():
                                try:
                                    target_id = get_or_create_word_id(cur, rel_word_clean, language_code, source_identifier)
                                    if target_id and target_id != word_id:
                                        r1 = insert_relation(cur, word_id, target_id, rel_type.value, source_identifier, metadata={"from_related_block": True})
                                        if r1: stats["relations"] += 1; stats["related_terms"] += 1
                                except Exception as e: logger.warning(f"Error creating related_term relation for '{lemma}': {e}")
                    # Process Top-Level Antonyms
                    if isinstance(top_antonyms, list):
                        for ant_item in top_antonyms:
                            ant_word = None; rel_type = RelationshipType.ANTONYM
                            if isinstance(ant_item, str): ant_word = ant_item
                            elif isinstance(ant_item, dict) and "term" in ant_item: ant_word = ant_item["term"]
                            ant_word_clean = clean_html(ant_word)
                            if ant_word_clean and ant_word_clean.lower() != lemma.lower():
                                try:
                                    target_id = get_or_create_word_id(cur, ant_word_clean, language_code, source_identifier)
                                    if target_id and target_id != word_id:
                                        r1 = insert_relation(cur, word_id, target_id, rel_type.value, source_identifier, metadata={"from_related_block": True})
                                        r2 = insert_relation(cur, target_id, word_id, rel_type.value, source_identifier, metadata={"from_related_block": True})
                                        if r1: stats["relations"] += 1; stats["antonyms"] += 1
                                except Exception as e: logger.warning(f"Error creating top-level antonym relation for '{lemma}': {e}")

                # --- Process Idioms --- (Copied from original)
                idioms_list = entry.get("idioms", [])
                if isinstance(idioms_list, list) and idioms_list:
                    try:
                        cleaned_idioms = [clean_html(i) for i in idioms_list if isinstance(i, str) and clean_html(i)]
                        if cleaned_idioms:
                            idioms_json = Json(cleaned_idioms)
                            cur.execute("UPDATE words SET idioms = %s WHERE id = %s", (idioms_json, word_id))
                            stats["idioms"] += len(cleaned_idioms)
                    except Exception as idiom_err: logger.warning(f"Error storing idioms for '{lemma}' (ID: {word_id}): {idiom_err}")

                # --- Finish Entry --- (Copied from original)
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                stats["processed"] += 1

                # --- Periodic Commit --- (Copied from original)
                if stats["processed"] % 500 == 0:
                    try:
                        conn.commit()
                        logger.info(f"Committed batch after {stats['processed']} KWF entries.")
                    except Exception as commit_err:
                        logger.error(f"Error during batch commit for KWF: {commit_err}. Rolling back...", exc_info=True)
                        conn.rollback()

            except Exception as entry_err:
                logger.error(f"Failed processing KWF entry #{entry_index} ('{original_key}' -> '{lemma}'): {entry_err}", exc_info=True)
                stats["errors"] += 1
                error_key = f"KwfEntryError: {type(entry_err).__name__}"; error_types[error_key] = error_types.get(error_key, 0) + 1
                try: cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                except Exception as rb_err:
                    logger.critical(f"CRITICAL: Failed rollback to savepoint {savepoint_name} after KWF entry error: {rb_err}. Aborting file.", exc_info=True)
                    raise entry_err from rb_err # Abort if rollback fails
            finally:
                if pbar:
                    pbar.update(1)

    except Exception as pbar_err:
        logger.error(f"Error processing KWF entries: {pbar_err}", exc_info=True)
        stats["errors"] += 1
        error_key = f"KwfEntryError: {type(pbar_err).__name__}"; error_types[error_key] = error_types.get(error_key, 0) + 1
        try: cur.execute("ROLLBACK")
        except Exception as rb_err:
            logger.critical(f"CRITICAL: Failed rollback after KWF entry error: {rb_err}. Aborting file.", exc_info=True)
            raise pbar_err from rb_err # Abort if rollback fails
    finally:
        if pbar:
            pbar.close()

    # --- Final Commit handled by migrate_data --- (Comment copied)
    logger.info(f"Finished processing {filename}. Stats: {stats}")
    if error_types:
        logger.warning(f"Error summary for {filename}: {error_types}")

    total_issues = stats["skipped_invalid"] + stats["errors"]
    return stats["processed"], total_issues 