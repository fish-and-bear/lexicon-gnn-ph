#!/usr/bin/env python3
"""
Process Wiktionary dump from Kaikki.org into the dictionary database format.
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Optional dependency

# Third-party imports
import psycopg2
from psycopg2.extras import Json

# Project-specific imports
from backend.dictionary_manager.db_helpers import (
    DBConnection,
    get_or_create_word_id,
    batch_get_or_create_word_ids,
    insert_definition,
    insert_etymology,
    insert_pronunciation,
    insert_relation,
    insert_word_form,
    insert_word_template,
    add_linguistic_note,
    get_standardized_pos_id,
    with_transaction,
    get_connection,
    insert_definition_link,
    insert_definition_category,
    insert_definition_example,
)
from backend.dictionary_manager.text_helpers import (
    clean_html,
    extract_parenthesized_text,
    normalize_lemma,
    standardize_source_identifier,
    BaybayinRomanizer
)
from backend.dictionary_manager.enums import RelationshipType

logger = logging.getLogger(__name__)

# Default language code for Tagalog entries
DEFAULT_LANGUAGE_CODE = "tl"

# For simple Baybayin validation
VALID_BAYBAYIN_REGEX = re.compile(r'^[\u1700-\u171F\s\u1735]+$')

# List of strings to consider as "non-words" that shouldn't be stored as words
NON_WORD_STRINGS = [
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
    "spanish-based orthography",
]

# --- NEW HELPER FUNCTION ---
# Replaced with adapted logic from dictionary_manager_orig.py.backup
def extract_script_info(
    entry: Dict[str, Any], script_name_in_template: str, script_tag: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts specific script form and explicit romanization if available,
    primarily by checking the 'forms' array for a matching script tag.
    NOTE: The head_templates fallback logic was removed due to unreliability based on data analysis.

    Args:
        entry: The dictionary entry dictionary.
        script_name_in_template: (Currently unused after removing head_template logic) The name for potential future use.
        script_tag: The tag used to identify the script in the 'forms' array (e.g., "Baybayin", "Badlit").

    Returns:
        A tuple containing the script form (str or None) and its romanization (str or None).
    """
    script_form: Optional[str] = None
    romanized: Optional[str] = None # Use 'romanized' as in backup logic

    # Add warning if trying to extract Badlit, as data format seems inconsistent/problematic
    if script_tag == "Badlit":
        logger.debug("Attempting to extract Badlit script. Note: Data format for Badlit in Kaikki dump appears inconsistent.")

    # Try 'forms' array first (Logic from backup)
    if "forms" in entry and isinstance(entry["forms"], list):
        for form_data in entry["forms"]:
            if (
                isinstance(form_data, dict)
                and "tags" in form_data
                and script_tag in form_data.get("tags", [])
            ):
                form_text = form_data.get("form", "").strip()
                if form_text:
                    # Remove common prefixes sometimes added by Wiktionary processing
                    prefixes = ["spelling ", "script ", script_tag.lower() + " "]
                    cleaned_form = form_text
                    for prefix in prefixes:
                        if cleaned_form.lower().startswith(prefix):
                            cleaned_form = cleaned_form[len(prefix) :].strip()

                    # Basic validation: Check if the result likely contains non-Latin script characters
                    # This helps filter out accidental matches like "Baybayin script" itself.
                    if cleaned_form and any(
                        ord(char) > 127 for char in cleaned_form # Simple check for non-ASCII
                        # A more robust check might involve specific Unicode ranges if needed
                        # e.g., not ("a" <= char.lower() <= "z")
                    ):
                        script_form = cleaned_form
                        # Get explicit romanization if provided in the same form entry
                        romanized = form_data.get("roman") or form_data.get("romanization")
                        # Return immediately if found in forms
                        return script_form, romanized

    # If not found in forms, return None, None (head_templates fallback removed)
    return None, None
# --- END NEW HELPER FUNCTION ---

# Helper function to normalize relation types
def normalize_relation_type(rel_type_str):
    """
    Normalize a relationship type string into proper enum value and properties.
    
    Args:
        rel_type_str: String representation of relationship type
        
    Returns:
        Tuple of (relation_type, bidirectional, inverse_type)
    """
    rel_enum = RelationshipType.from_string(rel_type_str)
    bidirectional = rel_enum.bidirectional
    inverse_rel = rel_enum.inverse
    
    return rel_enum.value, bidirectional, inverse_rel.value if inverse_rel else None

# New helper function to batch collect words for relation processing
def collect_related_words_batch(entry, language_code, word_cache):
    """
    Collects all related words from an entry in a single pass for batch lookup.
    This significantly reduces the number of individual get_or_create_word_id calls.
    
    Args:
        entry: The dictionary entry to process
        language_code: The language code of the main entry
        word_cache: Existing word cache dictionary to check first
        
    Returns:
        set of (lemma, language_code) tuples for batch lookup
    """
    # Set to collect unique (lemma, lang_code) tuples
    words_to_lookup = set()
    
    # Skip if entry doesn't exist or isn't a dict
    if not entry or not isinstance(entry, dict):
        return words_to_lookup
        
    # Get the main word from entry (for skipping self-references)
    main_word = entry.get('word', '').lower()
    
    # 1. Check etymology templates
    if 'etymology_templates' in entry and isinstance(entry['etymology_templates'], list):
        for template in entry['etymology_templates']:
            if not isinstance(template, dict) or 'name' not in template:
                continue
                
            template_name = template['name'].lower()
            args = template.get('args', {})
            
            # Handle different template types
            if template_name == 'blend':
                # Extract components (usually args 2, 3, etc.)
                for i in range(2, 6):  # Check args 2, 3, 4, 5
                    comp = args.get(str(i))
                    if comp and isinstance(comp, str):
                        comp_clean = clean_html(comp)
                        if comp_clean and comp_clean.lower() != main_word and comp_clean.lower() not in NON_WORD_STRINGS:
                            # For blends, same language
                            words_to_lookup.add((comp_clean, language_code))
                            
            elif template_name == 'inh' or template_name == 'inherited':
                proto_lang = args.get('2', '')
                proto_word = args.get('3', '')
                if proto_lang and proto_word and isinstance(proto_word, str):
                    proto_word_clean = clean_html(proto_word).lstrip('*')
                    if proto_word_clean and proto_word_clean.lower() != main_word and proto_word_clean.lower() not in NON_WORD_STRINGS:
                        words_to_lookup.add((proto_word_clean, proto_lang))
                        
            elif template_name == 'cog' or template_name == 'cognate':
                cog_lang = args.get('1', '')
                cog_word = args.get('2', '')
                if cog_lang and cog_word and isinstance(cog_word, str):
                    cog_clean = clean_html(cog_word)
                    if cog_clean and cog_clean.lower() != main_word and cog_clean.lower() not in NON_WORD_STRINGS:
                        words_to_lookup.add((cog_clean, cog_lang))
                        
            elif template_name == 'doublet':
                doublet_lang = args.get('1', '') or language_code
                doublet_word = args.get('2', '')
                if doublet_word and isinstance(doublet_word, str):
                    doublet_clean = clean_html(doublet_word)
                    if doublet_clean and doublet_clean.lower() != main_word and doublet_clean.lower() not in NON_WORD_STRINGS:
                        words_to_lookup.add((doublet_clean, doublet_lang))
                        
            elif template_name in ['derived', 'borrowing', 'derived from', 'borrowed from', 'bor', 'der', 'bor+']:
                # Extract source language and word
                source_lang = ''
                source_word = ''

                if template_name in ['bor', 'bor+', 'borrowing', 'borrowed from']:
                    source_lang = args.get('2', '') or args.get('1', '')
                    source_word = args.get('3', '')
                elif template_name in ['der', 'derived', 'derived from']:
                    source_lang = args.get('1', '')
                    source_word = args.get('2', '')

                if not source_lang:
                    source_lang = language_code

                if source_word and isinstance(source_word, str):
                    source_clean = clean_html(source_word)
                    if source_clean and source_clean.lower() != main_word and source_clean.lower() not in NON_WORD_STRINGS:
                        words_to_lookup.add((source_clean, source_lang))
                        
    # 2. Process senses for semantic relations
    if 'senses' in entry and isinstance(entry['senses'], list):
        for sense_idx, sense in enumerate(entry['senses']):
            if not isinstance(sense, dict):
                continue
                
            # Extract related terms from sense
            for rel_key in ['synonyms', 'antonyms', 'coordinate_terms', 'hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
                rel_items = sense.get(rel_key, [])
                
                if isinstance(rel_items, list):
                    for rel_item in rel_items:
                        rel_word = None
                        rel_lang = language_code  # Default same language
                        
                        if isinstance(rel_item, str):
                            rel_word = rel_item
                        elif isinstance(rel_item, dict) and 'word' in rel_item:
                            rel_word = rel_item.get('word')
                            # Check if language is specified
                            if 'lang' in rel_item:
                                rel_lang = rel_item.get('lang') or language_code
                                
                        if rel_word:
                            rel_word_clean = clean_html(rel_word)
                            if rel_word_clean and rel_word_clean.lower() != main_word and rel_word_clean.lower() not in NON_WORD_STRINGS:
                                words_to_lookup.add((rel_word_clean, rel_lang))
    
    # Return the set of collected words for batch processing
    return words_to_lookup

# -------------------------------------------------------------------
# Main Kaikki Processing Function (Moved from dictionary_manager.py)
# -------------------------------------------------------------------

def process_kaikki_jsonl(cur, filename: str):
    """Process Kaikki.org dictionary entries with optimized transaction handling."""
    # Configurable parameters for performance tuning
    CHUNK_SIZE = 20  # Reduce from 50 to 20 to prevent database overload
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
    def process_entry_chunk(entries_chunk, chunk_idx):
        """Process a chunk of entries with transaction handling and retries."""
        # --- REVISED CHUNK STATS INITIALIZATION ---
        chunk_stats = {
            "processed_ok": 0,
            "processed_with_errors": 0, # Count entries processed but had minor issues logged within process_single_entry
            "failed_entries": 0, # Count entries that caused a rollback within the chunk
            "error_details": defaultdict(int), # Track specific error types within the chunk
            "stats": defaultdict(int) # Aggregate detailed stats (definitions, relations, etc.)
        }
        
        # Reduce number of entries per transaction to avoid overloading PostgreSQL
        SUB_CHUNK_SIZE = 5  # Process 5 entries per database transaction
        
        # Split entries_chunk into smaller transaction batches
        for sub_chunk_idx in range(0, len(entries_chunk), SUB_CHUNK_SIZE):
            sub_chunk = entries_chunk[sub_chunk_idx:sub_chunk_idx + SUB_CHUNK_SIZE]
            sub_stats, error = process_sub_chunk(sub_chunk, chunk_idx, sub_chunk_idx)
            
            # Aggregate statistics
            if sub_stats:
                chunk_stats["processed_ok"] += sub_stats.get("processed_ok", 0)
                chunk_stats["processed_with_errors"] += sub_stats.get("processed_with_errors", 0)
                chunk_stats["failed_entries"] += sub_stats.get("failed_entries", 0)
                
                # Merge error details
                for error_key, count in sub_stats.get("error_details", {}).items():
                    chunk_stats["error_details"][error_key] += count
                
                # Merge detailed stats
                for stat_key, count in sub_stats.get("stats", {}).items():
                    chunk_stats["stats"][stat_key] += count
            
            if error:
                # Non-fatal error occurred but we'll continue with next sub-chunk
                chunk_stats["error_details"][f"SubChunkError-{sub_chunk_idx}"] += 1
        
        return chunk_stats, None  # No critical error message
        
    def process_sub_chunk(entries_sub_chunk, chunk_idx, sub_chunk_idx):
        """Process a small batch of entries within a single transaction."""
        sub_chunk_stats = {
            "processed_ok": 0,
            "processed_with_errors": 0,
            "failed_entries": 0,
            "error_details": defaultdict(int),
            "stats": defaultdict(int)
        }
        
        conn = None
        transaction_aborted = False # Flag to track transaction state
        # --- GET CONNECTION OUTSIDE CURSOR CONTEXT MANAGER ---
        try:
            # Implement connection retry logic
            conn = get_connection_with_retry(max_retries=3, delay=1.0)
            if not conn:
                logger.error(f"Failed to get DB connection for sub-chunk {chunk_idx}-{sub_chunk_idx}. Skipping sub-chunk.")
                sub_chunk_stats["failed_entries"] = len(entries_sub_chunk)
                sub_chunk_stats["error_details"]["DBConnectionFailed"] += 1
                return sub_chunk_stats, f"DB connection failed for sub-chunk {chunk_idx}-{sub_chunk_idx}"

            # --- PROCESS ENTRIES WITH PER-ENTRY SAVEPOINTS ---
            with conn.cursor() as cur:
                for entry_index, entry in enumerate(entries_sub_chunk):
                    # If the transaction is already known to be aborted, skip processing remaining entries
                    if transaction_aborted:
                        logger.warning(f"Skipping entry {entry_index} ('{entry.get('word', 'N/A')}') in sub-chunk {chunk_idx}-{sub_chunk_idx} because transaction is aborted.")
                        sub_chunk_stats["failed_entries"] += 1
                        sub_chunk_stats["error_details"]["SkippedDueToAbortedTxn"] += 1
                        continue # Move to the next entry
                        
                    entry_savepoint_name = f"entry_{chunk_idx}_{sub_chunk_idx}_{entry_index}_{int(time.time())}"
                    try:
                        cur.execute(f"SAVEPOINT {entry_savepoint_name}")

                        # Call the single entry processor
                        # Ensure process_single_entry now accepts and uses the cursor
                        result, error_msg = process_single_entry(cur, entry) # Pass cursor

                        if result and result.get("success"):
                            # Entry processed successfully (potentially with minor logged errors)
                            cur.execute(f"RELEASE SAVEPOINT {entry_savepoint_name}")
                            sub_chunk_stats["processed_ok"] += 1
                            # Aggregate detailed stats
                            for stat_key, stat_value in result.get('stats', {}).items():
                                sub_chunk_stats["stats"][stat_key] += stat_value
                            if error_msg: # If there were minor errors logged
                                sub_chunk_stats["processed_with_errors"] += 1
                                # Optionally log/track minor error messages here if needed
                                sub_chunk_stats["error_details"]["MinorEntryError"] += 1
                        else:
                            # Entry processing failed critically, rollback this entry
                            logger.warning(f"Rolling back entry {entry_index} in sub-chunk {chunk_idx}-{sub_chunk_idx} due to error: {error_msg or 'Unknown critical failure'}. Word: '{entry.get('word', 'N/A')}'")
                            try:
                                cur.execute(f"ROLLBACK TO SAVEPOINT {entry_savepoint_name}")
                                # RELEASE only if rollback succeeded
                                cur.execute(f"RELEASE SAVEPOINT {entry_savepoint_name}") 
                            except Exception as rb_err:
                                logger.error(f"Error during savepoint rollback/release for entry {entry_index}: {rb_err}. Transaction might be aborted.")
                                transaction_aborted = True # Mark transaction as aborted
                            
                            sub_chunk_stats["failed_entries"] += 1
                            error_type_key = f"EntryProcessingFailure: {error_msg or 'Unknown'}"
                            sub_chunk_stats["error_details"][error_type_key[:100]] += 1 # Limit key length

                    except Exception as single_entry_err:
                        # Catch unexpected errors from process_single_entry or this loop structure
                        logger.error(f"Unexpected error processing entry {entry_index} ('{entry.get('word', 'N/A')}') in sub-chunk {chunk_idx}-{sub_chunk_idx}: {single_entry_err}", exc_info=True)
                        # Don't try to rollback to savepoint if transaction is already aborted
                        if not transaction_aborted:
                           try:
                               cur.execute(f"ROLLBACK TO SAVEPOINT {entry_savepoint_name}")
                               # RELEASE only if rollback succeeded 
                               cur.execute(f"RELEASE SAVEPOINT {entry_savepoint_name}")
                           except Exception as rb_err:
                               logger.error(f"Error during savepoint rollback/release after unexpected error for entry {entry_index}: {rb_err}. Transaction likely aborted.")
                               # Mark transaction as aborted if rollback fails
                               transaction_aborted = True
                        else:
                            logger.warning(f"Skipping savepoint rollback for entry {entry_index} as transaction was already marked aborted.")
                            
                        sub_chunk_stats["failed_entries"] += 1
                        error_type_key = f"UnexpectedEntryError: {type(single_entry_err).__name__}"
                        sub_chunk_stats["error_details"][error_type_key] += 1
                        transaction_aborted = True # Ensure transaction is marked aborted after any exception

                # --- COMMIT or ROLLBACK THE SUB-CHUNK ---
                if transaction_aborted:
                    # If any entry caused the transaction to abort, rollback the whole sub-chunk
                    logger.warning(f"Rolling back sub-chunk {chunk_idx}-{sub_chunk_idx} due to aborted transaction state.")
                    try:
                        conn.rollback()
                    except Exception as final_rb_err:
                        logger.error(f"Error during final rollback for aborted sub-chunk {chunk_idx}-{sub_chunk_idx}: {final_rb_err}")
                        # Update stats to reflect all entries failed if rollback itself fails
                        sub_chunk_stats["failed_entries"] = len(entries_sub_chunk)
                        sub_chunk_stats["processed_ok"] = 0
                        sub_chunk_stats["processed_with_errors"] = 0
                        sub_chunk_stats["error_details"]["FinalRollbackError"] += 1
                        return sub_chunk_stats, f"Final rollback error in sub-chunk {chunk_idx}-{sub_chunk_idx}"
                else:
                    # If transaction wasn't aborted, commit the successful entries
                    try:
                        conn.commit()
                        processed_count = sub_chunk_stats['processed_ok'] # Count only fully successful ones
                        logger.info(f"Sub-chunk {chunk_idx}-{sub_chunk_idx} committed ({processed_count} OK, {sub_chunk_stats['processed_with_errors']} with minor errors, {sub_chunk_stats['failed_entries']} failed).")
                    except Exception as commit_err:
                        logger.error(f"Error committing sub-chunk {chunk_idx}-{sub_chunk_idx}: {commit_err}", exc_info=True)
                        try:
                            conn.rollback()
                        except Exception as rb_err:
                             logger.error(f"Error during rollback after commit error: {rb_err}")
                             pass # Ignore errors during rollback after commit failure
                        # Update stats to reflect all entries failed if commit fails
                        sub_chunk_stats["failed_entries"] = len(entries_sub_chunk)
                        sub_chunk_stats["processed_ok"] = 0
                        sub_chunk_stats["processed_with_errors"] = 0
                        sub_chunk_stats["error_details"]["CommitError"] += 1
                        return sub_chunk_stats, f"Commit error in sub-chunk {chunk_idx}-{sub_chunk_idx}"
                
                return sub_chunk_stats, None # Return stats and no critical error message

        except Exception as e: # Handle other unexpected errors during sub-chunk setup/teardown
            logger.error(f"Unexpected error processing sub-chunk {chunk_idx}-{sub_chunk_idx}: {e}", exc_info=True)
            sub_chunk_stats["failed_entries"] = len(entries_sub_chunk)
            sub_chunk_stats["error_details"][f"UnexpectedChunkError: {type(e).__name__}"] += 1
            
            # Attempt to rollback if connection exists
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except Exception as rb_err:
                     logger.error(f"Failed to rollback connection during sub-chunk error handling: {rb_err}")
            
            return sub_chunk_stats, f"Unexpected error processing sub-chunk {chunk_idx}-{sub_chunk_idx}: {e}"

        finally:
            # Ensure connection is closed if it was opened (important for connection pooling)
            if conn:
                 conn.close()
                 logger.debug(f"Closed DB connection for sub-chunk {chunk_idx}-{sub_chunk_idx}")

    def get_connection_with_retry(max_retries=3, delay=1.0):
        """Get database connection with retry logic"""
        attempt = 0
        while attempt < max_retries:
            try:
                conn = get_connection()
                if conn:
                    return conn
            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
                
            # Exponential backoff for retry
            time.sleep(delay * (2 ** attempt))
            attempt += 1
        
        logger.error(f"Failed to get database connection after {max_retries} attempts")
        return None

    # --- Single entry processor ---
    def process_single_entry(cur, entry):
        """Process a single dictionary entry with optimized data handling."""
        word = entry.get("word") # Use get to avoid KeyError if 'word' is missing
        if not word:
            logger.warning("Skipping entry due to missing 'word' field.")
            # Return structure indicating failure, matching the original logic
            return {"word_id": None, "stats": {}, "success": False}, "No word field"

        # --- Log entry start ---
        logger.debug(f"--- Starting processing for entry: '{word}' ---")

        # Apply special Kaikki lemma processing to preserve parenthesized variants and handle numbers
        # processed_word = process_kaikki_lemma(word)
        processed_word = word
        word_id = None
        error_messages = []
        local_stats = {k: 0 for k in entry_stats.keys()}

        try:
            # Get language code and validate
            logger.debug(f"[{word}] Getting language code...")
            pos = entry.get("pos", "unc")
            language_code = entry.get("lang_code", DEFAULT_LANGUAGE_CODE)
            if not language_code or len(language_code) > 10:
                language_code = DEFAULT_LANGUAGE_CODE
            logger.debug(f"[{word}] Language code set to: {language_code}")

            # Extract Baybayin and other script info
            logger.debug(f"[{word}] Extracting script info...")
            baybayin_form, baybayin_romanized = extract_script_info(
                entry, "Baybayin", "Baybayin"
            )
            badlit_form, badlit_romanized = extract_script_info(
                entry, "Badlit", "Badlit"
            )
            logger.debug(f"[{word}] Script info extracted. Baybayin: '{baybayin_form}', Badlit: '{badlit_form}'")

            # --- Baybayin Validation ---
            logger.debug(f"[{word}] Starting Baybayin validation...")
            validated_baybayin_form = None
            has_baybayin = False
            # --- BEGIN IMPROVED BAYBAYIN EXTRACTION ---
            original_baybayin_for_log = baybayin_form  # Store original for logging if needed
            if baybayin_form and isinstance(baybayin_form, str):
                # Use BaybayinRomanizer's validation method directly
                # --- FIX: Apply cleaning directly here --- 
                cleaned_baybayin_form = baybayin_form
                # Remove punctuation
                cleaned_baybayin_form = cleaned_baybayin_form.replace('᜵', '').replace('᜶', '')
                # Remove extra spaces (leading/trailing/multiple internal)
                cleaned_baybayin_form = ' '.join(cleaned_baybayin_form.split())

                # Validate the *cleaned* form using the romanizer's character knowledge
                if romanizer and romanizer.validate_text(cleaned_baybayin_form):
                    validated_baybayin_form = cleaned_baybayin_form
                    has_baybayin = True
                    logger.debug(f"[{word}] Baybayin validated: '{validated_baybayin_form}'")
                else:
                    # Log if the original form was non-empty but became invalid/empty after cleaning
                    if baybayin_form:
                        logger.warning(f"Invalid or problematic Baybayin form found for '{word}': '{baybayin_form}' (cleaned: '{cleaned_baybayin_form}'). Ignoring Baybayin for this entry.")
                        error_messages.append(f"Invalid Baybayin form '{baybayin_form}' ignored")
                    # Ensure validated form is None if cleaning/validation failed
                    validated_baybayin_form = None 
                    has_baybayin = False
            else:
                # Ensure validated form is None if original was missing or not string
                validated_baybayin_form = None 
                has_baybayin = False
            # --- END IMPROVED BAYBAYIN EXTRACTION ---
            if baybayin_form:
                logger.debug(f"[{word}] Baybayin form present, cleaning and validating...")
                # Check if the form contains only valid characters
                # Stripping the problematic dotted circle U+25CC proactively
                cleaned_baybayin_form = baybayin_form.replace("\u25cc", "").strip()
                if cleaned_baybayin_form and VALID_BAYBAYIN_REGEX.match(cleaned_baybayin_form):
                    validated_baybayin_form = cleaned_baybayin_form
                    has_baybayin = True
                    logger.debug(f"[{word}] Baybayin validated: '{validated_baybayin_form}'")
                else:
                    # Log if the original form was non-empty but became invalid/empty after cleaning
                    if baybayin_form:
                        logger.warning(f"Invalid or problematic Baybayin form found for '{word}': '{baybayin_form}'. Ignoring Baybayin for this entry.")
                        error_messages.append(f"Invalid Baybayin form '{baybayin_form}' ignored")

            # Generate romanization if needed, using the validated form
            logger.debug(f"[{word}] Starting romanization check...")
            romanized_form = None
            if (
                validated_baybayin_form
                and not baybayin_romanized
                and romanizer
                and romanizer.validate_text(validated_baybayin_form)
            ):
                try:
                    logger.debug(f"[{word}] Attempting to romanize: '{validated_baybayin_form}'")
                    romanized_form = romanizer.romanize(validated_baybayin_form)
                    logger.debug(f"[{word}] Romanization successful: '{romanized_form}'")
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
            logger.debug(f"[{word}] Checking word cache...")
            cache_key = f"{processed_word.lower()}|{language_code}"
            if cache_key in word_cache:
                word_id = word_cache[cache_key]
                logger.debug(f"[{word}] Found in cache. Word ID: {word_id}")
            else:
                logger.debug(f"[{word}] Not in cache. Preparing to get/create word ID...")
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
                logger.debug(f"[{word}] Calling get_or_create_word_id...")
                # --- DEBUG: Log arguments before calling get_or_create_word_id ---
                log_get_or_create_args = {
                    "lemma": processed_word,
                    "language_code": language_code,
                    "source_identifier": source_identifier,
                    "preserve_numbers": True,
                    "has_baybayin": has_baybayin,
                    "baybayin_form": validated_baybayin_form,
                    "romanized_form": romanized_form,
                    "badlit_form": badlit_form,
                    "hyphenation": hyphenation, # Pass the original list/None
                    "is_proper_noun": is_proper_noun,
                    "is_abbreviation": is_abbreviation,
                    "is_initialism": is_initialism,
                    "tags": word_tags_str,
                    "word_metadata": word_metadata,
                }
                # Truncate long strings for logging clarity
                truncated_log_args = {k: (str(v)[:100] + '...' if isinstance(v, str) and len(v) > 100 else v) 
                                      for k, v in log_get_or_create_args.items()}
                logger.debug(f"[{word}] Args for get_or_create_word_id (truncated): {truncated_log_args}")
                # --- END DEBUG LOG ---
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
                logger.debug(f"[{word}] get_or_create_word_id returned ID: {word_id}")

                # Add to cache for potential reuse
                if word_id:
                    word_cache[cache_key] = word_id

            if not word_id:
                return None, f"Failed to get/create word ID for '{word}'"

            # --- BATCH PROCESSING FOR RELATED WORDS ---
            # Collect all related words from this entry for batch processing
            logger.debug(f"[{word}] Collecting related words for batch lookup...")
            
            try:
                related_words_batch = collect_related_words_batch(entry, language_code, word_cache)
                if related_words_batch is None:
                    logger.warning(f"[{word}] collect_related_words_batch returned None, using empty set instead")
                    related_words_batch = set()
                
                batch_size = len(related_words_batch)
                
                if batch_size > 0:
                    logger.debug(f"[{word}] Found {batch_size} related words for batch lookup")
                    
                    # Convert set to list for batch_get_or_create_word_ids
                    related_words_list = list(related_words_batch)
                    
                    try:
                        # Use batch lookup function
                        batch_result = batch_get_or_create_word_ids(
                            cur, 
                            related_words_list, 
                            source=source_identifier,
                            batch_size=1000  # Keep default or adjust as needed
                        )
                        
                        # Update word cache with batch results
                        for (lemma, lang_code), word_id in batch_result.items():
                            if word_id:  # Only cache valid IDs
                                word_cache[f"{lemma.lower()}|{lang_code}"] = word_id
                        
                        logger.debug(f"[{word}] Successfully looked up {len(batch_result)} related words in batch")
                    except Exception as batch_err:
                        logger.error(f"[{word}] Error in batch word lookup: {batch_err}")
                        error_messages.append(f"Batch word lookup error: {batch_err}")
                        # Continue processing without related words
                else:
                    logger.debug(f"[{word}] No additional related words to batch lookup")
            except Exception as batch_collect_err:
                logger.error(f"[{word}] Error collecting related words: {batch_collect_err}")
                error_messages.append(f"Related word collection error: {batch_collect_err}")
                # Continue processing without related words
            # --- END BATCH PROCESSING ---

            # Process related data in properly grouped operations

            # --- Log before pronunciations ---
            logger.debug(f"[{word}] Processing pronunciations...")
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

            logger.debug(f"[{word}] Finished pronunciations.")

            # --- Log before forms ---
            logger.debug(f"[{word}] Processing forms...")
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

            logger.debug(f"[{word}] Finished forms.")

            # --- Log before inflection templates ---
            logger.debug(f"[{word}] Processing inflection templates...")
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
            logger.debug(f"[{word}] Finished inflection templates.")

            # --- Log before etymology ---
            logger.debug(f"[{word}] Processing etymology...")
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
                                            inverse_rel = rel_type.inverse # Corrected: Use property instead of method
                                            if inverse_rel:
                                                insert_relation(
                                                    cur,
                                                    related_word_id,
                                                    word_id,
                                                    inverse_rel.value, # <-- Use .value of the enum
                                                    source_identifier,
                                                    metadata,
                                                )
                except Exception as e:
                    error_messages.append(f"Etymology error: {str(e)}")
                    logger.error(f"[{word}] Error processing etymology: {e}", exc_info=True) # Added word context
            logger.debug(f"[{word}] Finished etymology.")

            # --- Log before top-level relations ---
            logger.debug(f"[{word}] Processing top-level relations...")
            # Process Top-Level Relation Fields
            top_level_relations_map = {
                "derived": RelationshipType.ROOT_OF,  # Word is ROOT_OF the derived term
                "related": RelationshipType.RELATED,
                "synonyms": RelationshipType.SYNONYM,
                "antonyms": RelationshipType.ANTONYM,
            }

            for field, rel_type in top_level_relations_map.items():
                if field in entry and isinstance(entry[field], list):
                    for related_item_idx, related_item in enumerate(entry[field]): # Added index for logging
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

                                logger.debug(f"[{word}] Processing top-level relation {field} item {related_item_idx}: {related_word_clean if 'related_word_clean' in locals() else related_item}")

            logger.debug(f"[{word}] Finished top-level relations.")

            # --- Log before head templates ---
            logger.debug(f"[{word}] Processing head templates...")
            # 4. Process templates
            if "head_templates" in entry and entry["head_templates"]:
                try:
                    for template in entry["head_templates"]:
                        if not isinstance(template, dict) or "name" not in template:
                            continue

                        template_name = template["name"]
                        args = template.get("args")
                        expansion = template.get("expansion")

                        # --- MODIFICATION: Use insert_word_template helper --- 
                        try:
                            if "insert_word_template" in globals() and callable(globals()["insert_word_template"]):
                                template_id = insert_word_template(
                                    cur,
                                    word_id,
                                    template_name,
                                    args=args if isinstance(args, dict) else {},
                                    expansion=expansion,
                                    source_identifier=source_identifier,
                                )
                                if template_id:
                                    local_stats["templates"] += 1
                                else:
                                     logger.warning(
                                        f"insert_word_template returned non-ID for head template '{template_name}' (Word ID: {word_id}). Args: {args}"
                                    )
                                     error_messages.append(
                                        f"Failed to store head template '{template_name}'"
                                    )
                            else:
                                if "insert_word_template_missing_logged" not in locals():
                                    logger.error(
                                        f"Function 'insert_word_template' not defined. Cannot store head templates for word ID {word_id}."
                                    )
                                    error_messages.append(
                                        f"Missing function: insert_word_template"
                                    )
                                    insert_word_template_missing_logged = True
                        except Exception as template_e:
                            error_msg = f"Error storing head template '{template_name}' for word ID {word_id}: {str(template_e)}. Args: {args}"
                            logger.error(error_msg, exc_info=True)
                            error_messages.append(error_msg)
                        # --- END MODIFICATION ---

                except Exception as e:
                    error_messages.append(f"Head template processing error: {str(e)}")
                    logger.error(f"[{word}] Error processing head templates block: {e}", exc_info=True)

            logger.debug(f"[{word}] Finished head templates.")

            # --- Log before senses ---
            logger.debug(f"[{word}] Processing senses...")
            # 5. Process senses/definitions
            if "senses" in entry and isinstance(entry["senses"], list):
                processed_sense_count = 0
                num_senses = len(entry["senses"])
                logger.debug(f"[{word}] Found {num_senses} senses.")

                for sense_idx, sense in enumerate(entry["senses"]):
                    # --- Log sense start ---
                    logger.debug(f"[{word}] Processing sense {sense_idx + 1}/{num_senses}...")
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
                                    for rel_item_idx, related_item in enumerate(sense[field]): # Added index
                                        logger.debug(f"[{word}] Processing sense relation {field} item {rel_item_idx}: {related_item}")
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

            logger.debug(f"[{word}] Finished senses.")

            # Update global statistics
            for key, value in local_stats.items():
                entry_stats[key] += value

            # Log warning if no senses were processed for an entry that had senses
            if len(entry["senses"]) > 0 and processed_sense_count == 0:
                logger.warning(f"Failed to process any senses for entry '{word}'")
                error_messages.append(f"No senses processed for '{word}'")

            # --- Log entry end ---
            logger.debug(f"--- Finished processing for entry: '{word}' ---")

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

            try:
                for line_num, line in enumerate(f): # Use enumerate for better error context
                    if not line.strip():
                        continue  # Skip empty lines
                    
                    entry_count += 1
                    # stats["total_entries"] = entry_count # This might be inaccurate if file size was used
                    
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
                        # --- Corrected JSON parsing loop ---
                        for json_line in current_chunk: # Use different variable name to avoid confusion
                            try:
                                entry = json.loads(json_line)
                                parsed_entries.append(entry)
                            except json.JSONDecodeError:
                                # Handle error for this specific line
                                logger.warning(f"Invalid JSON in chunk {chunk_index}, line approx {line_num}: {json_line[:100]}...")
                                stats["skipped_json_errors"] += 1
                        # --- End corrected loop ---
                        
                        if parsed_entries:
                            chunk_stats, chunk_error = process_entry_chunk(parsed_entries, chunk_index)

                            if chunk_stats: # Check if stats dict is not None
                                stats["processed_ok"] += chunk_stats.get("processed_ok", 0)
                                stats["processed_with_errors"] += chunk_stats.get("processed_with_errors", 0)
                                stats["failed_entries"] += chunk_stats.get("failed_entries", 0)

                                chunk_error_details = chunk_stats.get("error_details", {})
                                for error_key, error_count in chunk_error_details.items():
                                    error_summary[error_key] = error_summary.get(error_key, 0) + error_count

                                for stat_key in entry_stats.keys():
                                    entry_stats[stat_key] += chunk_stats.get(stat_key, 0)
                            elif chunk_error:
                                 logger.error(f"Chunk {chunk_index} failed entirely: {chunk_error}")
                                 stats["failed_entries"] += len(parsed_entries)
                                 error_summary["ChunkProcessingFailed"] = error_summary.get("ChunkProcessingFailed", 0) + 1

                            # Reset for next chunk (only inside the loop)
                            current_chunk = []
                            chunk_index += 1

                if progress_bar:
                    progress_bar.close()

                # Process any remaining entries in the final chunk
                if current_chunk:
                    # Parse JSON entries since we're receiving text lines
                    parsed_entries = []
                    # --- Corrected final JSON parsing loop ---
                    for json_line in current_chunk: # Use different variable name
                        try:
                            entry = json.loads(json_line)
                            parsed_entries.append(entry)
                        except json.JSONDecodeError:
                            # Handle error for this specific line
                            logger.warning(f"Invalid JSON in final chunk: {json_line[:100]}...")
                            stats["skipped_json_errors"] += 1
                    # --- End corrected loop ---
                    
                    if parsed_entries:
                        chunk_stats, chunk_error = process_entry_chunk(parsed_entries, chunk_index)

                        if chunk_stats: # Check if stats dict is not None
                            stats["processed_ok"] += chunk_stats.get("processed_ok", 0)
                            stats["processed_with_errors"] += chunk_stats.get("processed_with_errors", 0)
                            stats["failed_entries"] += chunk_stats.get("failed_entries", 0)

                            chunk_error_details = chunk_stats.get("error_details", {})
                            for error_key, error_count in chunk_error_details.items():
                                error_summary[error_key] = error_summary.get(error_key, 0) + error_count

                            for stat_key in entry_stats.keys():
                                entry_stats[stat_key] += chunk_stats.get(stat_key, 0)
                        elif chunk_error:
                                logger.error(f"Final chunk failed entirely: {chunk_error}")
                                stats["failed_entries"] += len(parsed_entries)
                                error_summary["ChunkProcessingFailed"] = error_summary.get("ChunkProcessingFailed", 0) + 1

            finally:
                if progress_bar:
                    progress_bar.close()

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


