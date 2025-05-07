#!/usr/bin/env python3
"""
marayum_processor.py

Processes dictionary entries from Project Marayum JSON files.
"""

import json
import logging
import os
import glob
import traceback
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json
from tqdm import tqdm

# Absolute imports from dictionary_manager
from backend.dictionary_manager.db_helpers import (
    get_or_create_word_id,
    insert_definition,
    insert_pronunciation,
    insert_relation,
    insert_etymology,
    insert_credit,
    insert_definition_example,
    with_transaction,
)
from backend.dictionary_manager.text_helpers import (
    standardize_source_identifier,
    SourceStandardization,
    normalize_lemma, # Needed by get_or_create_word_id implicitly?
    clean_html, # Likely needed for definition text
    get_language_code, # Added missing import
    get_standard_code, # Added get_standard_code import
)
# Import enums directly
from backend.dictionary_manager.enums import RelationshipType, DEFAULT_LANGUAGE_CODE

logger = logging.getLogger(__name__)

# --- ADDED: POS Mapping (similar to other processors, adapt if Marayum POS terms differ) ---
MARAYUM_POS_TO_STANDARD_MAP = {
    # Core POS - Assuming Marayum might use Tagalog terms or English ones
    "pangngalan": "pangngalan",
    "noun": "pangngalan",
    "pandiwa": "pandiwa",
    "verb": "pandiwa",
    "pang-uri": "pang-uri",
    "adjective": "pang-uri",
    "pang-abay": "pang-abay",
    "adverb": "pang-abay",
    "panghalip": "panghalip",
    "pronoun": "panghalip",
    "pang-ukol": "pang-ukol",
    "preposition": "pang-ukol",
    "pangatnig": "pangatnig",
    "conjunction": "pangatnig",
    "pandamdam": "pandamdam",
    "interjection": "pandamdam",
    "pantukoy": "pantukoy",
    "determiner": "pantukoy",
    "article": "pantukoy",
    "pamilang": "pamilang",
    "numeral": "pamilang",
    "partikula": "partikula",
    "particle": "partikula",
    "pang-angkop": "pang-angkop",
    "linker": "pang-angkop",

    # Descriptive / Functional types
    "parirala": "parirala",
    "expression": "parirala",
    "phrase": "parirala",
    "idyoma": "idyoma",
    "idiom": "idyoma",
    "daglat": "daglat",
    "abbreviation": "daglat",
    "akronim": "akronim",
    "acronym": "akronim",
    "initialism": "daglat", 
    "pagpapaikli": "pagpapaikli",
    "contraction": "pagpapaikli",
    "pangngalang pantangi": "pangngalang pantangi",
    "proper noun": "pangngalang pantangi",
    "pangngalang pambalana": "pangngalang pambalana",
    "common noun": "pangngalang pambalana",
    "pagbati": "pagbati",
    "greeting": "pagbati",
    "pananong": "pananong",
    "question word": "pananong",
    "interrogative": "pananong",
    
    # Other specific terms Marayum might use for POS/grammatical info
    "png.": "pangngalan", # Abbreviation for pangngalan
    "pnd.": "pandiwa",   # Abbreviation for pandiwa
    "pu.": "pang-uri",    # Abbreviation for pang-uri
    "pa.": "pang-abay",   # Abbreviation for pang-abay
    "ph.": "panghalip",  # Abbreviation for panghalip
    "pal.": "parirala",  # Common for phrase/expression
    "stm.": "salitang-ugat", # Salitang-ugat / Root word, if used in POS context
    "adj.": "pang-uri", # Common English abbreviation
    "adv.": "pang-abay", # Common English abbreviation
    "prep.": "pang-ukol", # Common English abbreviation
    "conj.": "pangatnig", # Common English abbreviation
    "int.": "pandamdam", # Common English abbreviation
    "pron.": "panghalip", # Common English abbreviation

    # Add more based on actual POS tags found in Marayum JSON files
}
# --- END ADDED: POS Mapping ---

# -------------------------------------------------------------------
# Marayum Processing Logic (Moved from dictionary_manager.py)
# -------------------------------------------------------------------


def process_marayum_json(
    cur, filename: str, source_identifier: Optional[str] = None
) -> Tuple[int, int]:
    """Processes a single Marayum JSON dictionary file, storing comprehensive metadata."""
    # Determine the effective source identifier
    if source_identifier:
        effective_source_identifier = source_identifier
    else:
        # Fallback to standardizing from filename if not provided
        effective_source_identifier = standardize_source_identifier(
            os.path.basename(filename)
        )
    logger.info(
        f"Processing Marayum file: {filename} with source: {effective_source_identifier}"
    )

    conn = cur.connection
    if conn.closed:
        logger.error("Connection is closed. Cannot proceed with processing.")
        return 0, 1  # Indicate 0 processed, 1 issue (connection error)

    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading or parsing Marayum file {filename}: {e}")
        return 0, 1  # Indicate 0 processed, 1 issue (file error)
    except Exception as e:
        logger.error(
            f"Unexpected error reading Marayum file {filename}: {e}", exc_info=True
        )
        return 0, 1  # Indicate 0 processed, 1 issue (file error)

    # --- Adjust for Dictionary Structure ---
    if not isinstance(data, dict):  # Expect a dictionary now
        logger.error(
            f"Marayum file {filename} does not contain a top-level dictionary."
        )
        return 0, 1  # Indicate 0 processed, 1 issue (format error)

    # Extract the list of word entries from the 'words' key
    word_entries_list = data.get("words", [])
    if not isinstance(word_entries_list, list):
        logger.error(
            f"Marayum file {filename} dictionary does not contain a 'words' list."
        )
        return 0, 1  # Indicate 0 processed, 1 issue (format error)

    entries_in_file = len(word_entries_list)  # Get length from the 'words' list
    if entries_in_file == 0:
        logger.info(f"Found 0 word entries in {filename}. Skipping file.")
        return 0, 0  # 0 processed, 0 issues

    logger.info(f"Found {entries_in_file} word entries in {filename}")

    # --- Determine Language Code for the entire file ---
    dict_info = data.get("dictionary_info", {})
    base_language_name = dict_info.get("base_language", "")
    
    # MODIFIED: Unpack tuple from get_language_code and set initial default for language_code
    standardized_language_code = DEFAULT_LANGUAGE_CODE 
    raw_input_language_for_file = base_language_name.lower().strip() if base_language_name else ""

    if base_language_name:
        normalized_lang_name = base_language_name.lower().strip() # This is the raw input
        std_code, raw_input = get_language_code(normalized_lang_name)
        standardized_language_code = std_code
        raw_input_language_for_file = raw_input # This will be normalized_lang_name or original if mapping changed it

        if not standardized_language_code or standardized_language_code == DEFAULT_LANGUAGE_CODE and normalized_lang_name != DEFAULT_LANGUAGE_CODE:
            logger.warning(
                f"Could not map base_language '{base_language_name}' (raw_input: '{raw_input_language_for_file}') for {filename}. Defaulting to '{DEFAULT_LANGUAGE_CODE}'."
            )
            standardized_language_code = DEFAULT_LANGUAGE_CODE
            # raw_input_language_for_file remains the initial attempt from base_language_name
        else:
            logger.info(
                f"Determined language code '{standardized_language_code}' (from raw: '{raw_input_language_for_file}') for {filename} from base language '{base_language_name}'."
            )
    else: # No base_language_name provided in dictionary_info
        logger.warning(f"No base_language specified in dictionary_info for {filename}. Defaulting to '{DEFAULT_LANGUAGE_CODE}'. Raw input will be empty.")
        standardized_language_code = DEFAULT_LANGUAGE_CODE
        raw_input_language_for_file = "" # No raw input to store

    # Initialize counters for this file's processing run
    stats = {
        "processed": 0,
        "definitions": 0,
        "relations": 0,
        "pronunciations": 0,
        "etymologies": 0,
        "credits": 0,
        "skipped": 0,
        "errors": 0,
        "examples": 0,
    }
    error_types = {}

    # Iterate over the extracted list of word entries
    with tqdm(
        total=entries_in_file,
        desc=f"Processing {effective_source_identifier}",
        unit="entry",
        leave=False,
    ) as pbar:
        try:
            for entry_index, entry in enumerate(word_entries_list):
                savepoint_name = f"marayum_{entry_index}_{abs(hash(str(entry)) % 1000000)}"
                lemma = ""
                word_id = None

                try:
                    cur.execute(f"SAVEPOINT {savepoint_name}")

                    if not isinstance(entry, dict):
                        logger.warning(
                            f"Skipping non-dictionary item at index {entry_index} in {filename}"
                        )
                        stats["skipped"] += 1
                        cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        pbar.update(1)
                        continue

                    lemma = entry.get("word", "").strip()
                    if not lemma:
                        logger.warning(
                            f"Skipping entry at index {entry_index} due to missing or empty 'word' field"
                        )
                        stats["skipped"] += 1
                        cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        pbar.update(1)
                        continue

                    # --- Language code determined before the loop, use the 'language_code' variable ---

                    # --- Extract Comprehensive Word Metadata ---
                    # Store the entire entry, excluding fields handled elsewhere, as JSON
                    word_metadata = {
                        k: v
                        for k, v in entry.items()
                        if k
                        not in [
                            "word",  # Handled as lemma
                            "definitions",  # Handled separately
                            "pronunciation",  # Handled separately
                            "etymology",  # Handled separately
                            "see_also",  # Handled separately
                            "credits",  # Handled separately
                            "examples",  # Often duplicated in definitions, handled there
                            "pos", # Word-level POS is handled below if not overridden by definition-level POS
                        ]
                    }

                    # --- Get or Create Word ---
                    try:
                        # Pass the language_code determined for the file
                        word_id = get_or_create_word_id(
                            cur,
                            lemma,
                            language_code=standardized_language_code,  # Use file-level standardized language code
                            source_identifier=effective_source_identifier,
                            word_metadata=word_metadata,
                            raw_input_language=raw_input_language_for_file # ADDED: Pass raw language for file
                        )
                        if not word_id:
                            # This case should ideally be handled within get_or_create_word_id by raising an error
                            raise ValueError(
                                f"get_or_create_word_id returned None unexpectedly for lemma '{lemma}'"
                            )
                        # Use debug level for successful creation log
                        logger.debug(
                            f"Word '{lemma}' ({standardized_language_code}) created/found (ID: {word_id}) from source '{effective_source_identifier}'."
                        )

                    except Exception as word_err:
                        logger.error(
                            f"CRITICAL: Failed to get/create word ID for lemma '{lemma}' (Index: {entry_index}) in {filename}: {word_err}"
                        )
                        stats["errors"] += 1
                        error_key = f"WordCreationError: {type(word_err).__name__}"
                        error_types[error_key] = error_types.get(error_key, 0) + 1
                        cur.execute(
                            f"ROLLBACK TO SAVEPOINT {savepoint_name}"
                        )  # Rollback this entry
                        pbar.update(1)
                        continue  # Skip to next entry - cannot proceed without word_id

                    # --- Process Credits using insert_credit (AFTER word_id is obtained) ---
                    credits_raw = entry.get("credits")
                    if credits_raw:
                        # insert_credit handles string or dict and logs internally
                        credit_inserted_id = insert_credit(
                            cur, word_id, credits_raw, effective_source_identifier
                        )
                        if credit_inserted_id:
                            stats["credits"] += 1
                        else:
                            # Log failure here as well, as insert_credit might only log warning/error
                            logger.warning(
                                f"Failed to insert credit for word ID {word_id} ('{lemma}') from source '{effective_source_identifier}'. Raw data: {credits_raw}"
                            )
                            error_key = f"CreditInsertFailure"
                            error_types[error_key] = error_types.get(error_key, 0) + 1

                    # --- Process Definitions and Examples ---
                    definitions = entry.get("definitions", [])
                    if isinstance(definitions, list):
                        for def_idx, def_item in enumerate(definitions):
                            if (
                                not isinstance(def_item, dict)
                                or "definition" not in def_item
                            ):
                                logger.debug(
                                    f"Skipping invalid definition item at index {def_idx} for word '{lemma}' (ID: {word_id})"
                                )
                                continue

                            definition_text = def_item.get("definition", "")
                            # Allow definition text to be None or empty string initially
                            if isinstance(definition_text, str):
                                definition_text = definition_text.strip()
                            elif definition_text is None:
                                definition_text = ""  # Handle None case explicitly
                            else:
                                logger.warning(
                                    f"Non-string definition text found for word '{lemma}' (ID: {word_id}), Def {def_idx+1}: {type(definition_text)}. Skipping definition."
                                )
                                continue  # Skip this definition if type is wrong

                            if not definition_text:
                                logger.debug(
                                    f"Skipping empty definition for word '{lemma}' (ID: {word_id}), Def {def_idx+1}"
                                )
                                continue

                            # --- Prepare Comprehensive Definition Metadata ---
                            # Store the entire definition item, excluding fields stored directly, as JSON
                            def_metadata = {
                                k: v
                                for k, v in def_item.items()
                                if k
                                not in [
                                    "definition",  # Stored directly
                                    "examples",  # Stored directly via 'examples' arg
                                    "gram" # Handled for POS below
                                ]
                            }

                            # Process examples associated with this definition
                            examples_processed = []
                            examples_raw = def_item.get("examples", [])
                            if isinstance(examples_raw, list):
                                examples_processed = process_examples(
                                    examples_raw
                                )  # Use helper
                            
                            # --- Determine POS for this definition ---
                            definition_pos_raw = def_item.get("gram") # Check definition-level POS (e.g., from 'gram' key)
                            word_level_pos_raw = entry.get("pos")     # Fallback to word-level POS from the main entry
                            
                            final_pos_to_map = None
                            if definition_pos_raw and isinstance(definition_pos_raw, str) and definition_pos_raw.strip():
                                final_pos_to_map = definition_pos_raw.strip()
                            elif word_level_pos_raw and isinstance(word_level_pos_raw, str) and word_level_pos_raw.strip():
                                final_pos_to_map = word_level_pos_raw.strip()
                            
                            # --- Use get_standard_code for POS mapping --- 
                            part_of_speech_for_db = get_standard_code(final_pos_to_map)
                            if final_pos_to_map and part_of_speech_for_db == "unc" and final_pos_to_map.lower() != "unc":
                                logger.debug(f"POS '{final_pos_to_map}' for '{lemma}' mapped to 'unc' by get_standard_code.")
                            # --- End POS Determination and Mapping ---

                            # Insert definition
                            try:
                                usage_notes = None # Marayum definitions might have notes in def_metadata
                                def_id = insert_definition(
                                    cur,
                                    word_id,
                                    definition_text,
                                    part_of_speech=part_of_speech_for_db, # Use mapped POS
                                    usage_notes=usage_notes,
                                    tags=None, # Marayum tags are often in def_metadata via 'domain' or 'region'
                                    sources=effective_source_identifier,
                                    metadata=def_metadata,
                                )
                                if def_id:
                                    stats["definitions"] += 1
                                    # --- INSERT EXAMPLES SEPARATELY --- (Moved from insert_definition call)
                                    if examples_processed:
                                        for example_item in examples_processed:
                                            if isinstance(example_item, dict) and example_item.get('text'):
                                                try:
                                                    # Check if function exists before calling
                                                    if 'insert_definition_example' in globals() and callable(globals()['insert_definition_example']):
                                                        example_inserted_id = insert_definition_example(
                                                            cur,
                                                            def_id, # Use the returned definition ID
                                                            example_item, # Pass the example dictionary
                                                            effective_source_identifier
                                                        )
                                                        if example_inserted_id:
                                                            stats["examples"] += 1 # Increment example stats here
                                                    else:
                                                        logger.warning("insert_definition_example function not found, cannot store examples.")
                                                        # Optionally break if function missing to avoid repeated logs
                                                        break
                                                except Exception as ex_err:
                                                    logger.error(f"Error inserting example for def ID {def_id}: {ex_err}", exc_info=True)
                                                    error_key = f"ExampleInsertError: {type(ex_err).__name__}"
                                                    error_types[error_key] = error_types.get(error_key, 0) + 1
                                    # --- END EXAMPLE INSERTION ---
                                else:  # insert_definition returned None or raised error handled internally
                                    logger.warning(
                                        f"insert_definition failed for '{lemma}' (ID: {word_id}), Def {def_idx+1}. Check internal logs."
                                    )
                                    error_key = f"DefinitionInsertFailure"
                                    error_types[error_key] = (
                                        error_types.get(error_key, 0) + 1
                                    )

                            except Exception as def_err:
                                # Catch errors not handled inside insert_definition
                                logger.error(
                                    f"Error during definition insertion for '{lemma}' (ID: {word_id}), Def {def_idx+1}: {def_err}",
                                    exc_info=True,
                                )
                                error_key = (
                                    f"DefinitionInsertError: {type(def_err).__name__}"
                                )
                                error_types[error_key] = error_types.get(error_key, 0) + 1
                                # Continue processing other parts of the entry, but log the error

                    # --- Process Pronunciation (if available) ---
                    pronunciation = entry.get("pronunciation")
                    if pronunciation:
                        # Marayum pronunciation is often just a string, format it
                        pron_obj = {}
                        if isinstance(pronunciation, str) and pronunciation.strip():
                            pron_obj = {
                                "value": pronunciation.strip(),
                                "type": "ipa",
                            }  # Assume IPA if not empty
                        elif isinstance(pronunciation, dict):
                            pron_obj = pronunciation  # Use as is if already a dict
                        else:
                            logger.debug(
                                f"Skipping invalid pronunciation data type for '{lemma}' (ID: {word_id}): {type(pronunciation)}"
                            )

                        if pron_obj and pron_obj.get("value"):  # Ensure value exists
                            try:
                                # --- Refined Extraction for pron_type ---
                                pron_type_raw = pron_obj.get("type")
                                pron_type = "ipa" # Default
                                if isinstance(pron_type_raw, str) and pron_type_raw.strip():
                                    pron_type = pron_type_raw.strip().lower()
                                elif pron_type_raw is not None:
                                    # Log if type was provided but not a usable string
                                    logger.warning(f"Pronunciation 'type' for word ID {word_id} is not a simple string: {pron_type_raw}. Defaulting to '{pron_type}'.")
                                # --- End Refinement ---

                                pron_value = pron_obj.get("value")
                                # Extract tags and metadata if they exist in pron_obj
                                pron_tags = pron_obj.get("tags")
                                pron_metadata = pron_obj.get("metadata")

                                pron_inserted_id = insert_pronunciation(
                                    cur,
                                    word_id,
                                    pron_type,          # Pass validated type string
                                    pron_value,         # Pass value string
                                    tags=pron_tags,     # Pass optional tags
                                    metadata=pron_metadata, # Pass optional metadata
                                    source_identifier=effective_source_identifier
                                )
                                if pron_inserted_id:
                                    stats["pronunciations"] += 1
                                else:
                                    logger.warning(
                                        f"insert_pronunciation failed for '{lemma}' (ID: {word_id}). Check internal logs."
                                    )
                                    error_key = f"PronunciationInsertFailure"
                                    error_types[error_key] = (
                                        error_types.get(error_key, 0) + 1
                                    )
                            except Exception as pron_err:
                                logger.error(
                                    f"Error during pronunciation insertion for '{lemma}' (ID: {word_id}): {pron_err}",
                                    exc_info=True,
                                )
                                error_key = (
                                    f"PronunciationInsertError: {type(pron_err).__name__}"
                                )
                                error_types[error_key] = error_types.get(error_key, 0) + 1

                    # --- Process Etymology (if available) ---
                    etymology_data = entry.get("etymology") # Changed variable name
                    if etymology_data:
                        etymology_text_to_insert = None
                        etymology_lang_codes = None

                        if isinstance(etymology_data, str) and etymology_data.strip():
                            etymology_text_to_insert = etymology_data.strip()
                        elif isinstance(etymology_data, dict):
                            etymology_text_to_insert = etymology_data.get("text", "") or etymology_data.get("notes", "") # Prefer text, fallback to notes
                            if isinstance(etymology_text_to_insert, str):
                                etymology_text_to_insert = etymology_text_to_insert.strip()
                            else:
                                etymology_text_to_insert = None # Ensure it's None if not a string
                            
                            source_lang = etymology_data.get("source_language")
                            if isinstance(source_lang, str) and source_lang.strip():
                                etymology_lang_codes = source_lang.strip()
                            elif isinstance(source_lang, list) and source_lang:
                                etymology_lang_codes = ",".join(filter(None, [str(l).strip() for l in source_lang]))

                        if etymology_text_to_insert:
                            try:
                                ety_id = insert_etymology(
                                    cur, 
                                    word_id, 
                                    etymology_text_to_insert, 
                                    effective_source_identifier,
                                    language_codes=etymology_lang_codes
                                )
                                if ety_id:
                                    stats["etymologies"] += 1
                                else:
                                    logger.warning(
                                        f"insert_etymology failed for '{lemma}' (ID: {word_id}). Text: '{etymology_text_to_insert[:50]}...'. Check internal logs."
                                    )
                                    error_key = f"EtymologyInsertFailure"
                                    error_types[error_key] = error_types.get(error_key, 0) + 1
                            except Exception as ety_err:
                                logger.error(
                                    f"Error during etymology insertion for '{lemma}' (ID: {word_id}): {ety_err}",
                                    exc_info=True,
                                )
                                error_key = f"EtymologyInsertError: {type(ety_err).__name__}"
                                error_types[error_key] = error_types.get(error_key, 0) + 1
                        elif isinstance(etymology_data, dict):
                             logger.debug(f"Skipping etymology for '{lemma}' (ID: {word_id}) due to missing text in object: {etymology_data}")


                    # --- Process See Also (as Relations) ---
                    see_also = entry.get("see_also", [])
                    if isinstance(see_also, list):
                        # Assuming process_see_also returns list of related word strings
                        see_also_words = process_see_also(see_also, standardized_language_code)
                        for related_word_str in see_also_words:
                            if (
                                related_word_str.lower() != lemma.lower()
                            ):  # Avoid self-relation (case-insensitive)
                                related_word_id = None
                                try:
                                    # Get ID for the related word, creating if necessary
                                    # IMPORTANT: We pass None for metadata here, otherwise we might overwrite
                                    # the full metadata if this 'see_also' word is processed later directly.
                                    related_word_id = get_or_create_word_id(
                                        cur,
                                        related_word_str,
                                        language_code=standardized_language_code,  # Use same standardized language code
                                        source_identifier=effective_source_identifier,  
                                        word_metadata=None,  
                                        raw_input_language=raw_input_language_for_file # ADDED: Pass raw language of the file
                                    )
                                    if (
                                        related_word_id and related_word_id != word_id
                                    ):  # Check IDs aren't same
                                        rel_id = insert_relation(
                                            cur,
                                            word_id,
                                            related_word_id,
                                            RelationshipType.SEE_ALSO.value,
                                            source_identifier=effective_source_identifier,
                                        )
                                        if rel_id:
                                            stats["relations"] += 1
                                            # Optionally insert bidirectional relation
                                            # insert_relation(cur, related_word_id, word_id, RelationshipType.SEE_ALSO, effective_source_identifier)
                                        else:
                                            # Log if insert_relation fails (might be due to constraint/conflict handled internally)
                                            logger.debug(
                                                f"Failed to insert SEE_ALSO relation {word_id} -> {related_word_id}. Might already exist."
                                            )
                                            # error_key = f"SeeAlsoRelationInsertFailure" # Only if failure is unexpected
                                            # error_types[error_key] = error_types.get(error_key, 0) + 1

                                except Exception as rel_err:
                                    logger.error(
                                        f"Error processing 'see_also' relation for '{lemma}' -> '{related_word_str}': {rel_err}",
                                        exc_info=True,
                                    )
                                    error_key = (
                                        f"SeeAlsoRelationError: {type(rel_err).__name__}"
                                    )
                                    error_types[error_key] = (
                                        error_types.get(error_key, 0) + 1
                                    )

                    # --- Finish Entry Processing ---
                    # If we reach here, the main parts were processed (or errors handled non-critically)
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

    # Log final stats
    logger.info(
        f"Finished processing {filename}. Processed: {stats['processed']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}"
    )
    logger.info(
        f"  Stats => Defs: {stats['definitions']}, Examples: {stats['examples']}, Etys: {stats['etymologies']}, Relations: {stats['relations']}"
    )
    if error_types:
        logger.warning(f"Error summary for {filename}: {error_types}")

    total_issues = stats["skipped"] + stats["errors"]
    return stats["processed"], total_issues


# Helper function to process examples list
def process_examples(examples_list):
    processed = []
    if not isinstance(examples_list, list):
        return processed # Return empty list if input is not a list
    for ex_item in examples_list:
        cleaned_text = None
        processed_item = {}

        if isinstance(ex_item, str) and ex_item.strip():
            cleaned_text = clean_html(ex_item.strip())
            # if cleaned_text: # This check is redundant if we only append if cleaned_text later
            #     processed_item = {"text": cleaned_text} # This line was causing issues if cleaned_text was empty
        elif isinstance(ex_item, dict):
            # Handle common keys like 'text' or 'example'
            raw_text = None
            if "text" in ex_item and isinstance(ex_item["text"], str):
                raw_text = ex_item["text"]
            elif "example" in ex_item and isinstance(ex_item["example"], str):
                raw_text = ex_item["example"]
            
            if raw_text and raw_text.strip():
                cleaned_text = clean_html(raw_text.strip())
                if cleaned_text: # Only proceed if cleaning results in non-empty text
                    # Keep other fields from the original dict
                    processed_item = ex_item.copy()
                    processed_item["text"] = cleaned_text # Standardize key to 'text'
                    # Remove original key if it wasn't 'text' and was 'example'
                    if "example" in processed_item and "example" != "text": 
                        del processed_item["example"] 

        # Only append if we have a valid, cleaned text and a corresponding item structure
        if cleaned_text: # This check is crucial
            if processed_item and "text" in processed_item: # If item came from dict
                processed.append(processed_item)
            elif not processed_item: # If item came from string, create new dict
                processed.append({"text": cleaned_text})

    return processed


# Helper function to process "see also" entries
def process_see_also(see_also_list, language_code: str) -> List[str]:
    """
    Processes the 'see_also' list from a Marayum entry.
    Handles list containing strings or simple dictionaries with a 'word' key.
    Returns a list of cleaned, valid related word strings.
    """
    processed_words = []
    if not isinstance(see_also_list, list):
        logger.warning(f"Invalid 'see_also' format: expected list, got {type(see_also_list)}")
        return processed_words # Return empty list if input is not a list

    for item in see_also_list:
        related_word = None
        if isinstance(item, str):
            related_word = item.strip()
        elif isinstance(item, dict) and "word" in item and isinstance(item["word"], str):
            related_word = item["word"].strip()
        # Add more checks here if the structure is more complex (e.g., nested dicts)

        if related_word:
            # Basic cleaning - could be enhanced if needed
            cleaned_word = clean_html(related_word) # Use clean_html if available
            if cleaned_word:
                processed_words.append(cleaned_word)
        else:
             logger.debug(f"Skipping invalid item in 'see_also' list: {item}")

    return processed_words


@with_transaction(commit=True)
def process_marayum_directory(cur, directory_path: str) -> None:
    """Process all Project Marayum dictionary files in the specified directory."""

    # Normalize directory path
    directory_path = os.path.normpath(directory_path)
    logger.info(f"Processing Marayum dictionaries from directory: {directory_path}")

    # Define the standard source identifier for ALL files in this directory
    marayum_source_id = SourceStandardization.standardize_sources("marayum")
    # Check if the directory exists
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return

    # Find only processed JSON files in the directory
    json_pattern = os.path.join(directory_path, "*_processed.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        logger.warning(f"No processed JSON files found in {directory_path}")
        logger.info(f"Looking for files with the pattern: {json_pattern}")
        # Try to list what's actually in the directory
        if os.path.isdir(directory_path):
            files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
            if files:
                processed_files = [f for f in files if f.endswith("_processed.json")]
                unprocessed_files = [
                    f for f in files if not f.endswith("_processed.json")
                ]
                if processed_files:
                    logger.info(f"Processed files found: {', '.join(processed_files)}")
                if unprocessed_files:
                    logger.info(
                        f"Unprocessed files found (skipping): {', '.join(unprocessed_files)}"
                    )
            else:
                logger.info(f"No JSON files found in directory: {directory_path}")
        return

    total_processed = 0
    total_errors = 0
    total_files_processed = 0
    total_files_skipped = 0

    # Sort files by size for efficient processing (process smaller files first)
    json_files.sort(key=lambda x: os.path.getsize(x))

    # Process each processed JSON file found
    for json_file in json_files:
        try:
            # Normalize path
            json_file = os.path.normpath(json_file)

            # Check file size and readability
            if not os.path.isfile(json_file):
                logger.error(f"Not a file: {json_file}")
                total_files_skipped += 1
                continue

            file_size = os.path.getsize(json_file)
            if file_size == 0:
                logger.warning(f"Skipping empty file: {json_file}")
                total_files_skipped += 1
                continue

            if not os.access(json_file, os.R_OK):
                logger.error(f"File not readable: {json_file}")
                total_files_skipped += 1
                continue

            # Process the dictionary
            processed, errors = process_marayum_json(
                cur, json_file, source_identifier=marayum_source_id
            )  # <-- MODIFIED CALL
            total_processed += processed
            total_errors += errors
            if processed > 0:
                total_files_processed += 1
            else:
                total_files_skipped += 1
                logger.warning(
                    f"No entries processed from {os.path.basename(json_file)}"
                )

        except Exception as e:
            logger.error(
                f"Error processing Marayum dictionary file {json_file}: {str(e)}"
            )
            total_errors += 1
            total_files_skipped += 1
            continue

    # Log final statistics
    logger.info(f"Completed processing Marayum dictionaries:")
    logger.info(f"  Files processed: {total_files_processed}")
    logger.info(f"  Files skipped: {total_files_skipped}")
    logger.info(f"  Total entries processed: {total_processed}")
    logger.info(f"  Total errors encountered: {total_errors}")


