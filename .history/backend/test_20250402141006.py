# Replace the existing insert_definition function (starts around line 2925)
@with_transaction(commit=True)
def insert_definition(cur, word_id: int, definition_text: str,
                      source_identifier: str, # MANDATORY
                      part_of_speech: Optional[str] = None, # Changed default to None
                      examples: Optional[str] = None, usage_notes: Optional[str] = None,
                      tags: Optional[str] = None, # Added optional tags parameter back
                      standardized_pos_id: Optional[int] = None, # Added optional standardized_pos_id
                      original_pos: Optional[str] = None # Added optional original_pos
                      ) -> Optional[int]:
    """
    Insert definition data for a word. Ensures source is updated on conflict.
    Handles potential unique constraint violations gracefully.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        definition_text: The definition text.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.
        part_of_speech: Original part of speech string (optional, used if standardized_pos_id not provided).
        examples: Example sentences (optional, should be JSON string or None).
        usage_notes: Notes on usage (optional).
        tags: Comma-separated tags string (optional).
        standardized_pos_id: Pre-calculated standardized POS ID (optional).
        original_pos: Original POS string (optional, overrides part_of_speech if provided).

    Returns:
        The ID of the inserted/updated definition record, or None if failed or duplicate ignored.
    """
    if not source_identifier:
        logger.error(f"CRITICAL: Skipping definition insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    if not definition_text:
        logger.warning(f"Empty definition text for word ID {word_id} from source '{source_identifier}'. Skipping.")
        return None

    # Use provided original_pos if available, otherwise use part_of_speech
    final_original_pos = original_pos if original_pos is not None else part_of_speech

    try:
        # Get standardized part of speech ID if not provided
        if standardized_pos_id is None:
            standardized_pos_id = get_standardized_pos_id(cur, final_original_pos)

        # --- Truncation (optional, uncomment if needed) ---
        # max_def_length = 4096
        # if len(definition_text) > max_def_length:
        #     logger.warning(f"Definition text for word ID {word_id} truncated from {len(definition_text)} to {max_def_length} characters.")
        #     definition_text = definition_text[:max_def_length]
        # max_examples_length = 4096
        # if examples and len(examples) > max_examples_length:
        #     logger.warning(f"Examples for word ID {word_id} truncated from {len(examples)} to {max_examples_length} characters.")
        #     examples = examples[:max_examples_length]
        # --- End Truncation ---

        # Insert or update definition
        cur.execute("""
            INSERT INTO definitions (word_id, definition_text, standardized_pos_id, examples, usage_notes, tags, sources, original_pos)
            VALUES (%(word_id)s, %(definition_text)s, %(pos_id)s, %(examples)s, %(usage_notes)s, %(tags)s, %(sources)s, %(original_pos)s)
            ON CONFLICT (word_id, definition_text, standardized_pos_id)
            DO UPDATE SET
                -- Update fields, preferring EXCLUDED (new) values but keeping old if new is NULL
                examples = COALESCE(EXCLUDED.examples, definitions.examples),
                usage_notes = COALESCE(EXCLUDED.usage_notes, definitions.usage_notes),
                tags = COALESCE(EXCLUDED.tags, definitions.tags),
                sources = EXCLUDED.sources, -- <<<<< KEY FIX: Update sources on conflict
                original_pos = COALESCE(EXCLUDED.original_pos, definitions.original_pos),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, {
            'word_id': word_id,
            'definition_text': definition_text,
            'pos_id': standardized_pos_id,
            'examples': examples, # Ensure this is None or a valid JSON string if applicable
            'usage_notes': usage_notes,
            'tags': tags,
            'sources': source_identifier, # Use the mandatory source_identifier
            'original_pos': final_original_pos # Store original POS string
        })
        result = cur.fetchone()
        if result:
            def_id = result[0]
            logger.debug(f"Inserted/Updated definition (ID: {def_id}) for word ID {word_id} from source '{source_identifier}'.")
            return def_id
        else:
            # This case might happen if RETURNING clause doesn't return anything (highly unlikely for INSERT/UPDATE)
             logger.warning(f"Definition insert/update for word ID {word_id} did not return an ID.")
             return None

    except psycopg2.errors.UniqueViolation:
        # This specific error might occur if ON CONFLICT doesn't cover all unique constraints
        # or if there's a race condition (though less likely with SERIALIZABLE isolation).
        logger.debug(f"Definition already exists (UniqueViolation caught) for word ID {word_id} and text '{definition_text[:50]}...'.")
        # Optionally, query the existing definition ID if needed
        # cur.execute("SELECT id FROM definitions WHERE word_id = %s AND definition_text = %s AND standardized_pos_id = %s",
        #             (word_id, definition_text, standardized_pos_id))
        # existing_def = cur.fetchone()
        # return existing_def[0] if existing_def else None
        return None # Or return existing ID if queried
    except psycopg2.Error as e:
        logger.error(f"Database error inserting definition for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=False) # Reduce log noise
        # Transaction decorator handles rollback
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting definition for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        # Transaction decorator handles rollback
        return None


# Replace the existing process_kwf_dictionary function (starts around line 3426)
@with_transaction(commit=False)  # Manage transactions manually
def process_kwf_dictionary(cur, filename: str):
    """Processes entries from the KWF Diksiyonaryo ng Wikang Filipino JSON file."""
    logger.info(f"Processing KWF dictionary file: {filename}")
    stats = {"entries": 0, "definitions": 0, "pronunciations": 0, "etymologies": 0, "relations": 0, "errors": 0}
    # Define source identifier for this function
    source_identifier = standardize_source_identifier(filename) # Or a more specific name like "KWF-Diksiyonaryo"
    conn = cur.connection # Get connection for manual commit/rollback

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"KWF dictionary file not found: {filename}")
        return stats # Return empty stats if file not found
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in KWF dictionary file {filename}: {e}")
        stats["errors"] += 1 # Count file loading as one error
        return stats
    except Exception as e:
        logger.error(f"Error reading KWF file {filename}: {e}", exc_info=True)
        stats["errors"] += 1
        return stats


    total_entries = len(data)
    logger.info(f"Found {total_entries} entries in KWF dictionary.")

    for entry_index, entry in enumerate(tqdm(data, total=total_entries, desc="Processing KWF")):
        # Create a savepoint for each entry
        savepoint_name = f"kwf_entry_{entry_index}" # Use index for safety
        try:
            cur.execute(f"SAVEPOINT {savepoint_name}")
        except Exception as e:
            logger.error(f"Failed to create savepoint {savepoint_name} for KWF entry index {entry_index}: {e}. Skipping.")
            stats["errors"] += 1
            continue # Skip this entry

        try:
            lemma = entry.get('word', '').strip()
            if not lemma:
                stats["errors"] += 1
                logger.warning(f"Skipping KWF entry at index {entry_index} with empty word.")
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release savepoint before continuing
                continue

            language_code = 'tl' # Assuming KWF is primarily Tagalog
            # Extract potential POS from the first definition for context (might be None)
            first_def_pos = None
            if entry.get('definitions') and isinstance(entry['definitions'], list):
                first_def_pos = entry['definitions'][0].get('part_of_speech')

            # Determine if proper noun (example based on possible KWF tag)
            is_proper_noun = False
            if first_def_pos and 'pangngalang pantangi' in first_def_pos.lower():
                 is_proper_noun = True

            # --- Word Creation ---
            # Pass source_identifier and any other relevant kwargs
            word_id = get_or_create_word_id(
                cur,
                lemma,
                language_code=language_code,
                source_identifier=source_identifier, # Pass source_identifier
                is_proper_noun=is_proper_noun
                # Add other kwargs like check_exists=False if needed
            )
            if not word_id:
                raise ValueError(f"Failed to get/create word ID for KWF entry: {lemma}")

            stats["entries"] += 1

            # --- Process definitions ---
            if 'definitions' in entry and isinstance(entry['definitions'], list):
                for definition_entry in entry['definitions']:
                    if not isinstance(definition_entry, dict): continue # Skip invalid definition formats
                    definition_text = definition_entry.get('definition', '').strip()
                    if not definition_text: continue

                    part_of_speech = definition_entry.get('part_of_speech', None)
                    examples_list = definition_entry.get('examples', [])
                    examples_str = None
                    # Handle list of strings or potentially list of dicts for examples
                    if examples_list:
                         if all(isinstance(ex, str) for ex in examples_list):
                             examples_str = '\\n'.join(ex.strip() for ex in examples_list if ex.strip())
                         # Add elif for dicts if KWF uses that structure

                    try:
                        def_id = insert_definition(
                            cur,
                            word_id,
                            definition_text,
                            source_identifier=source_identifier, # Pass source_identifier
                            part_of_speech=part_of_speech, # Pass original POS
                            examples=examples_str
                            # Pass tags, usage_notes if available in KWF data
                        )
                        if def_id:
                            stats["definitions"] += 1
                        else:
                            # insert_definition logs errors, maybe increment a warning counter?
                            pass
                    except Exception as def_e:
                        logger.warning(f"Failed to insert KWF definition for '{lemma}': {definition_text[:50]}... Error: {def_e}")
                        # Don't increment main error count here, let insert_definition handle logging

            # --- Process pronunciations ---
            pron_data = entry.get('pronunciation') # Could be string or dict based on KWF format
            if pron_data:
                 try:
                      pron_id = insert_pronunciation(cur, word_id, pron_data, source_identifier=source_identifier)
                      if pron_id:
                          stats["pronunciations"] += 1
                 except Exception as pron_e:
                       logger.warning(f"Failed to insert KWF pronunciation for '{lemma}': {pron_data}. Error: {pron_e}")

            # --- Process etymology ---
            ety_text = entry.get('etymology')
            if isinstance(ety_text, str) and ety_text.strip():
                try:
                    ety_id = insert_etymology(cur, word_id, ety_text.strip(), source_identifier=source_identifier)
                    if ety_id:
                        stats["etymologies"] += 1
                except Exception as ety_e:
                     logger.warning(f"Failed to insert KWF etymology for '{lemma}': {ety_text[:50]}... Error: {ety_e}")

            # --- Process relations (synonyms, antonyms, etc.) ---
            relation_map = {
                'synonyms': RelationshipType.SYNONYM,
                'antonyms': RelationshipType.ANTONYM
                # Add other relation types if present in KWF data
            }
            for rel_key, rel_type in relation_map.items():
                related_words = entry.get(rel_key)
                if isinstance(related_words, list):
                    for rel_word_str in related_words:
                        rel_word_clean = rel_word_str.strip() if isinstance(rel_word_str, str) else None
                        if rel_word_clean and rel_word_clean != lemma: # Avoid self-relation
                             try:
                                  # Get/create related word ID, ensuring source is passed
                                  rel_id = get_or_create_word_id(cur, rel_word_clean, language_code, source_identifier=source_identifier)
                                  if rel_id:
                                      # Insert the relation, passing the source
                                      relation_record_id = insert_relation(cur, word_id, rel_id, rel_type, source_identifier=source_identifier)
                                      if relation_record_id:
                                           stats["relations"] += 1
                                      # If bidirectional (like SYNONYM, ANTONYM), insert the inverse
                                      if rel_type.bidirectional:
                                          insert_relation(cur, rel_id, word_id, rel_type, source_identifier=source_identifier)
                                          # Don't double-count bidirectional in stats
                             except Exception as rel_e:
                                  logger.warning(f"Failed to insert KWF {rel_key} relation for '{lemma}'->'{rel_word_clean}': {rel_e}")


            # Release the savepoint if processing was successful
            cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")

        except Exception as entry_e:
            logger.error(f"Error processing KWF entry index {entry_index} ('{entry.get('word', 'N/A')}') : {entry_e}", exc_info=False) # Less verbose logging
            stats["errors"] += 1
            try:
                # Rollback to the savepoint created at the start of this entry's processing
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            except Exception as rb_err:
                # If rollback fails, the transaction state is uncertain, likely needs full rollback
                logger.critical(f"CRITICAL: Failed to rollback to savepoint {savepoint_name} after error: {rb_err}. Attempting full transaction rollback.")
                conn.rollback() # Rollback the entire transaction
                logger.warning("Performed full transaction rollback due to savepoint rollback failure.")
                # Re-raise the original error to potentially stop the whole process if critical
                raise entry_e from rb_err

    # Final commit for the file after processing all entries
    try:
        conn.commit()
        logger.info(f"Finished processing KWF file {filename}. Stats: {stats}")
    except Exception as commit_err:
         logger.error(f"Error during final commit for {filename}: {commit_err}. Rolling back any uncommitted changes...")
         stats["errors"] += 1 # Count commit failure as an error
         conn.rollback()

    return stats


# Replace the existing process_tagalog_words function (starts around line 4059)
@with_transaction(commit=True) # Keep commit=True for simplicity unless batching is needed
def process_tagalog_words(cur, filename: str):
    """Processes entries from the diksiyonaryo.ph JSON file."""
    logger.info(f"Processing Tagalog Words (diksiyonaryo.ph) file: {filename}")
    stats = {"processed": 0, "skipped": 0, "errors": 0, "definitions_added": 0}
    # Define source identifier
    source_identifier = standardize_source_identifier(filename) # Or "diksiyonaryo.ph"

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return stats
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        stats["errors"] += 1
        return stats
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        stats["errors"] += 1
        return stats


    total_entries = len(data)
    logger.info(f"Found {total_entries} entries in {filename}")

    for entry in tqdm(data, total=total_entries, desc="Processing diksiyonaryo.ph"):
        # Note: Since commit=True is used on the function, each entry effectively runs
        # in its own transaction. An error in one entry rolls back only that entry.
        try:
            lemma = entry.get('word', '').strip()
            if not lemma:
                stats["skipped"] += 1
                continue

            language_code = 'tl' # Assuming Tagalog

            # --- Word Creation ---
            word_id = get_or_create_word_id(
                cur,
                lemma,
                language_code=language_code,
                source_identifier=source_identifier # Pass source_identifier
                # Add other kwargs if derivable from diksiyonaryo.ph data
            )
            if not word_id:
                # Log error inside get_or_create_word_id is preferred
                # logger.error(f"Failed to get/create word ID for diksiyonaryo.ph entry: {lemma}")
                stats["errors"] += 1
                continue # Skip if word creation failed

            # --- Process definition(s) ---
            definition_text = entry.get('definition', '').strip()
            if definition_text:
                # Try to extract POS if available (e.g., often in parenthesis)
                pos_in_def = None
                clean_definition = definition_text
                # Example regex (adjust based on actual format):
                # pos_match = re.search(r'^\((.*?)\)\s*:', definition_text)
                # if pos_match:
                #     pos_in_def = pos_match.group(1).strip()
                #     clean_definition = re.sub(r'^\(.*?:\)\s*', '', definition_text).strip()

                try:
                    def_id = insert_definition(
                        cur,
                        word_id,
                        clean_definition,
                        source_identifier=source_identifier, # Pass source_identifier
                        part_of_speech=pos_in_def # Pass original extracted POS if any
                        # Pass examples, tags etc. if available in entry dict
                    )
                    if def_id:
                         stats["definitions_added"] += 1
                    # else: # insert_definition handles logging warnings/errors
                    #      stats["errors"] += 1 # Or maybe a different counter for insert issues
                except Exception as def_e:
                     # This catch might be redundant if insert_definition handles its errors well
                     logger.warning(f"Unexpected error inserting definition for '{lemma}' from {source_identifier}: {def_e}")
                     stats["errors"] += 1
            else:
                # Increment skipped only if 'definition' key is missing or empty
                if 'definition' not in entry or not entry['definition']:
                     logger.debug(f"Skipping entry '{lemma}' from {source_identifier} due to missing definition.")
                     stats["skipped"] += 1

            stats["processed"] += 1

        except Exception as entry_e:
            # Catch errors during word creation or other unexpected issues per entry
            logger.error(f"Error processing entry '{entry.get('word', 'UNKNOWN')}' from {filename}: {entry_e}", exc_info=False) # Less verbose
            stats["errors"] += 1
            # The transaction decorator handles rollback for this specific entry

    logger.info(f"Finished processing {filename}: {stats}")
    return stats


# Replace the existing process_root_words_cleaned function (starts around line 4377)
@with_transaction(commit=False)  # Manage transactions manually
def process_root_words_cleaned(cur, filename: str):
    """Processes entries from the tagalog.com Root Words JSON file."""
    logger.info(f"Processing Root Words (tagalog.com) file: {filename}")
    stats = {"roots_processed": 0, "definitions_added": 0, "relations_added": 0, "associated_processed": 0, "errors": 0}
    # Define source identifier
    source_identifier = standardize_source_identifier(filename) # Or "tagalog.com-RootWords"
    conn = cur.connection

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return stats
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        stats["errors"] += 1
        return stats
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        stats["errors"] += 1
        return stats


    total_roots = len(data)
    logger.info(f"Found {total_roots} root word entries in {filename}")

    for entry_index, root_word_entry in enumerate(tqdm(data, total=total_roots, desc="Processing tagalog.com")):
        savepoint_name = f"tagalogcom_root_{entry_index}"
        try:
            cur.execute(f"SAVEPOINT {savepoint_name}")
        except Exception as e:
            logger.error(f"Failed to create savepoint {savepoint_name} for tagalog.com root index {entry_index}: {e}. Skipping.")
            stats["errors"] += 1
            continue

        try:
            root_word = root_word_entry.get('root_word', '').strip()
            if not root_word:
                stats["errors"] += 1 # Or skipped?
                logger.warning(f"Skipping tagalog.com entry at index {entry_index} with empty root word.")
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                continue

            language_code = 'tl'

            # --- Root Word Creation ---
            root_word_id = get_or_create_word_id(
                cur,
                root_word,
                language_code=language_code,
                source_identifier=source_identifier, # Pass source_identifier
                is_root_word=True # Add flag if schema supports it
            )
            if not root_word_id:
                raise ValueError(f"Failed to get/create root word ID for tagalog.com entry: {root_word}")

            stats["roots_processed"] += 1

            # --- Process definitions for the root word ---
            definitions = root_word_entry.get('definitions', [])
            if isinstance(definitions, list):
                 for definition_item in definitions:
                     # Handling definitions that might be strings or dicts
                     definition_text: Optional[str] = None
                     part_of_speech: Optional[str] = None
                     examples: Optional[str] = None # Should be JSON string if examples are structured

                     if isinstance(definition_item, str):
                         definition_text = definition_item.strip()
                         # Potentially extract POS from string definition if pattern exists
                     elif isinstance(definition_item, dict):
                         # Check common keys for definition text
                         definition_text = definition_item.get('text', '').strip() or \
                                           definition_item.get('definition', '').strip()
                         # Check common keys for part of speech
                         part_of_speech = definition_item.get('pos') or \
                                          definition_item.get('part_of_speech')
                         # Extract examples if present, format as needed (e.g., JSON list of strings)
                         raw_examples = definition_item.get('examples')
                         if isinstance(raw_examples, list) and all(isinstance(ex, str) for ex in raw_examples):
                             # Simple list of strings example
                             examples_list = [ex.strip() for ex in raw_examples if ex.strip()]
                             if examples_list:
                                try:
                                    examples = json.dumps(examples_list)
                                except TypeError: # Handle potential non-serializable data if format is complex
                                    logger.warning(f"Could not serialize examples for root '{root_word}': {examples_list}")
                                    examples = None
                         elif isinstance(raw_examples, str): # If examples is just a single string
                             examples = json.dumps([raw_examples.strip()]) if raw_examples.strip() else None


                     if definition_text:
                         try:
                             def_id = insert_definition(
                                 cur,
                                 root_word_id,
                                 definition_text,
                                 source_identifier=source_identifier, # Pass source_identifier
                                 part_of_speech=part_of_speech, # Pass original POS
                                 examples=examples # Pass examples (JSON string or None)
                             )
                             if def_id:
                                 stats["definitions_added"] += 1
                             # else: insert_definition logs issues
                         except Exception as def_e:
                             logger.warning(f"Failed insert definition for root '{root_word}': {definition_text[:50]}... Error: {def_e}")


            # --- Process associated words (assuming they are derived) ---
            associated_words = root_word_entry.get('associated_words', [])
            if isinstance(associated_words, list):
                for assoc_word_entry in associated_words:
                    assoc_word: Optional[str] = None
                    # Check if entry is string or dict
                    if isinstance(assoc_word_entry, str):
                        assoc_word = assoc_word_entry.strip()
                    elif isinstance(assoc_word_entry, dict):
                         assoc_word = assoc_word_entry.get('word', '').strip()
                         # Could potentially add definitions for associated words here if provided

                    if assoc_word and assoc_word != root_word:
                        stats["associated_processed"] += 1
                        try:
                            assoc_word_id = get_or_create_word_id(
                                cur,
                                assoc_word,
                                language_code=language_code,
                                root_word_id=root_word_id, # Link derived word to root in word table if possible
                                source_identifier=source_identifier # Pass source_identifier
                            )
                            if assoc_word_id:
                                 # Add DERIVED_FROM relationship
                                 rel_id = insert_relation(
                                     cur, assoc_word_id, root_word_id,
                                     RelationshipType.DERIVED_FROM,
                                     source_identifier=source_identifier # Pass source
                                 )
                                 if rel_id:
                                     stats["relations_added"] += 1
                                 # Optionally add ROOT_OF relation from root to derived
                                 # rel_id_inv = insert_relation(cur, root_word_id, assoc_word_id, RelationshipType.ROOT_OF, source_identifier=source_identifier)
                                 # if rel_id_inv: stats["relations_added"] += 1 # Or just one count per pair

                        except Exception as assoc_e:
                            logger.warning(f"Failed to process/relate associated word '{assoc_word}' for root '{root_word}': {assoc_e}")
                            # Don't increment main error count, maybe a specific counter?

            # Release savepoint for successful entry
            cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")

        except Exception as entry_e:
            logger.error(f"Error processing tagalog.com root index {entry_index} ('{root_word_entry.get('root_word', 'N/A')}') : {entry_e}", exc_info=False)
            stats["errors"] += 1
            try:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            except Exception as rb_err:
                logger.critical(f"CRITICAL: Failed rollback to savepoint {savepoint_name}: {rb_err}. Attempting full transaction rollback.")
                conn.rollback()
                logger.warning("Performed full transaction rollback.")
                raise entry_e from rb_err

    # Final commit
    try:
        conn.commit()
        logger.info(f"Finished processing {filename}. Stats: {stats}")
    except Exception as commit_err:
         logger.error(f"Error during final commit for {filename}: {commit_err}. Rolling back...")
         stats["errors"] += 1
         conn.rollback()

    return stats


# Replace the existing process_marayum_json function (starts around line 6704)
@with_transaction(commit=False)  # Manage transactions manually within the loop
def process_marayum_json(cur, filename: str) -> Tuple[int, int]:
    """Process a single Marayum JSON file, ensuring source propagation."""
    # Use the standardized source from the original function logic
    source_identifier = SourceStandardization.standardize_sources(os.path.basename(filename))
    logger.info(f"Processing Marayum file: {filename} with source: {source_identifier}")

    processed_count = 0
    error_count = 0
    skipped_count = 0
    definitions_added = 0 # Add counter for definitions
    relations_added = 0   # Add counter for relations
    pronunciations_added = 0 # Add counter for pronunciations
    etymologies_added = 0 # Add counter for etymologies
    entries_in_file = 0
    conn = cur.connection # Get connection for manual commit/rollback

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading or parsing Marayum file {filename}: {e}")
        return 0, 1 # Return 0 processed, 1 error (representing the file)
    except Exception as e:
        logger.error(f"Unexpected error reading Marayum file {filename}: {e}", exc_info=True)
        return 0, 1

    if not isinstance(data, list):
        logger.error(f"Marayum file {filename} does not contain a list of entries.")
        return 0, 1

    entries_in_file = len(data)
    logger.info(f"Found {entries_in_file} entries in {filename}")

    # Assuming get_language_mapping() is defined elsewhere and returns a dict
    # language_mapping = get_language_mapping() # Load this once if needed by get_language_code

    with tqdm(total=entries_in_file, desc=f"Processing {source_identifier}", unit="entry") as pbar:
        for entry_index, entry in enumerate(data):
            # --- Savepoint ---
            # Marayum uses entry_index, which is safe
            savepoint_name = f"marayum_entry_{entry_index}"
            try:
                cur.execute(f"SAVEPOINT {savepoint_name}")
            except Exception as e:
                logger.error(f"Failed to create savepoint {savepoint_name} for Marayum entry index {entry_index}: {e}. Skipping.")
                error_count += 1
                pbar.update(1)
                continue # Skip this entry

            try:
                if not isinstance(entry, dict):
                     logger.warning(f"Skipping invalid entry at index {entry_index} in {filename} (not a dict)")
                     skipped_count += 1
                     cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release if skipping early
                     pbar.update(1)
                     continue

                headword = entry.get('headword', '').strip()
                language_name = entry.get('language', 'Unknown') # Keep original language name
                language_code = get_language_code(language_name) # Use helper

                if not headword:
                     logger.warning(f"Skipping entry at index {entry_index} in {filename} (no headword)")
                     skipped_count += 1
                     cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                     pbar.update(1)
                     continue

                # --- Word Creation ---
                # Preserve original metadata logic
                word_metadata = {'marayum_source': source_identifier, 'language_name': language_name, 'processed_timestamp': datetime.now().isoformat()}
                # Add other Marayum fields to metadata if needed: e.g., entry.get('id')
                try:
                    word_id = get_or_create_word_id(
                        cur, headword, language_code=language_code,
                        source_identifier=source_identifier, # Pass MANDATORY source_identifier
                        word_metadata=json.dumps(word_metadata) # Pass metadata as before
                    )
                except TypeError as te: # Catch if word_metadata isn't expected by get_or_create_word_id
                    logger.warning(f"TypeError calling get_or_create_word_id for '{headword}', possibly due to word_metadata. Retrying without it. Error: {te}")
                    word_id = get_or_create_word_id(
                         cur, headword, language_code=language_code,
                         source_identifier=source_identifier # Pass MANDATORY source_identifier
                     )

                if not word_id: raise ValueError(f"Failed to get/create word ID for '{headword}' ({language_code}) from {source_identifier}")

                # --- Process Pronunciations ---
                pronunciations = entry.get('pronunciations')
                if isinstance(pronunciations, list):
                     for pron in pronunciations:
                         if isinstance(pron, str) and pron.strip():
                             try:
                                 # Pass mandatory source_identifier argument correctly
                                 pron_id = insert_pronunciation(cur, word_id, pron.strip(), source_identifier=source_identifier)
                                 if pron_id: pronunciations_added += 1
                             except Exception as e: logger.warning(f"Failed pronunciation insert '{pron}' for '{headword}' from {source_identifier}: {e}")

                # --- Process Etymology ---
                etymology_text_raw = entry.get('etymology')
                if isinstance(etymology_text_raw, str) and etymology_text_raw.strip():
                     etymology_text = etymology_text_raw.strip()
                     try:
                         # Pass mandatory source_identifier argument correctly
                         ety_id = insert_etymology(cur, word_id, etymology_text, source_identifier=source_identifier)
                         if ety_id: etymologies_added += 1
                     except Exception as e: logger.warning(f"Failed etymology insert for '{headword}' from {source_identifier}: {e}")

                # --- Process Senses ---
                senses = entry.get('senses')
                if isinstance(senses, list):
                     for sense_idx, sense in enumerate(senses):
                         if not isinstance(sense, dict): continue
                         definition = sense.get('definition', '').strip()
                         pos = sense.get('partOfSpeech', '') # Marayum uses 'partOfSpeech'
                         examples_raw = sense.get('examples', [])

                         if definition:
                             # Process examples into JSON list of strings/dicts
                             examples_list = []
                             if isinstance(examples_raw, list):
                                  examples_list = [ex.strip() for ex in examples_raw if isinstance(ex, str) and ex.strip()]
                                  # If Marayum examples are dicts, adapt here:
                                  # examples_list = [{"text": ex.get("text"), ...} for ex in examples_raw if isinstance(ex, dict)]

                             # Convert examples list to JSON string or None
                             examples_json = None
                             if examples_list:
                                 try:
                                     examples_json = json.dumps(examples_list)
                                 except TypeError:
                                     logger.warning(f"Could not serialize examples for sense in '{headword}': {examples_list}")

                             # Get standardized POS ID using the original POS string
                             # Use standardize_entry_pos helper if defined and needed
                             std_pos = standardize_entry_pos(pos) if pos else None
                             standardized_pos_id = get_standardized_pos_id(cur, std_pos)

                             try:
                                 # Pass mandatory source_identifier
                                 # Pass standardized_pos_id and original_pos directly
                                 definition_id = insert_definition(
                                     cur, word_id, definition,
                                     source_identifier=source_identifier, # Pass mandatory source
                                     standardized_pos_id=standardized_pos_id, # Pass pre-calculated ID
                                     original_pos=pos, # Pass original string
                                     examples=examples_json
                                     # Add tags/usage_notes if Marayum provides them here in sense
                                 )
                                 if definition_id:
                                     definitions_added += 1
                                     # Process sense-level relations if Marayum format includes them
                                     sense_relation_map = {
                                         'synonyms': RelationshipType.SYNONYM,
                                         'antonyms': RelationshipType.ANTONYM
                                         # Add other sense relations if needed
                                     }
                                     for sense_rel_key, sense_rel_type in sense_relation_map.items():
                                         if sense_rel_key in sense and isinstance(sense[sense_rel_key], list):
                                             for sense_rel_word in sense[sense_rel_key]:
                                                 if isinstance(sense_rel_word, str) and sense_rel_word.strip():
                                                     sense_rel_word_clean = sense_rel_word.strip()
                                                     if sense_rel_word_clean != headword:
                                                         try:
                                                             sense_rel_id = get_or_create_word_id(cur, sense_rel_word_clean, language_code, source_identifier=source_identifier)
                                                             if sense_rel_id:
                                                                 # Add metadata specific to sense relation if needed
                                                                 sense_rel_metadata = {'source': source_identifier, 'definition_id': definition_id, 'confidence': 70} # Example confidence
                                                                 # Pass mandatory source_identifier argument correctly
                                                                 rel_rec_id = insert_relation(
                                                                     cur, word_id, sense_rel_id, sense_rel_type,
                                                                     source_identifier=source_identifier, # Pass source
                                                                     metadata=sense_rel_metadata
                                                                 )
                                                                 if rel_rec_id: relations_added += 1
                                                                 # Handle bidirectional for sense relations if needed
                                                                 # if sense_rel_type.bidirectional:
                                                                 #    insert_relation(cur, sense_rel_id, word_id, sense_rel_type, source_identifier=source_identifier, metadata=sense_rel_metadata)

                                                         except Exception as rel_e:
                                                             logger.warning(f"Error adding sense {sense_rel_key} '{sense_rel_word_clean}' for '{headword}': {rel_e}")

                             except psycopg2.errors.UniqueViolation:
                                logger.debug(f"Marayum def exists for '{headword}', sense idx {sense_idx}: {definition[:30]}...")
                             except Exception as def_e: logger.error(f"Failed def insert for '{headword}', sense idx {sense_idx}: {definition[:30]}... : {def_e}", exc_info=False) # Keep log cleaner

                # --- Process Top-Level Relations ---
                # Using the structure from the provided snippet
                top_level_relation_map = {
                     'synonyms': RelationshipType.SYNONYM,
                     'antonyms': RelationshipType.ANTONYM,
                     'relatedWords': RelationshipType.RELATED # Assuming 'relatedWords' maps to RELATED
                     # Add 'hypernyms', 'hyponyms' etc. if Marayum provides them
                }
                for rel_key, rel_type_enum in top_level_relation_map.items():
                     related_words_list = entry.get(rel_key)
                     if isinstance(related_words_list, list):
                         for related_word_str in related_words_list:
                             if isinstance(related_word_str, str) and related_word_str.strip():
                                 related_word_clean = related_word_str.strip()
                                 if related_word_clean != headword: # Avoid self-relation
                                     try:
                                         related_id = get_or_create_word_id(cur, related_word_clean, language_code, source_identifier=source_identifier)
                                         if related_id:
                                             # Add metadata for top-level relations if needed
                                             top_rel_metadata = {'source': source_identifier, 'confidence': 70} # Example confidence
                                             # Pass mandatory source_identifier argument correctly
                                             top_rel_rec_id = insert_relation(
                                                 cur, word_id, related_id, rel_type_enum,
                                                 source_identifier=source_identifier, # Pass source
                                                 metadata=top_rel_metadata
                                             )
                                             if top_rel_rec_id: relations_added += 1
                                             # Handle bidirectional for top-level relations
                                             # if rel_type_enum.bidirectional:
                                             #    insert_relation(cur, related_id, word_id, rel_type_enum, source_identifier=source_identifier, metadata=top_rel_metadata)

                                     except Exception as top_rel_e: logger.warning(f"Error adding top-level {rel_key} '{related_word_clean}' for '{headword}': {top_rel_e}")

                # --- Finish Entry ---
                processed_count += 1
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")

            except Exception as entry_err:
                logger.error(f"Error processing Marayum entry index {entry_index} ('{entry.get('headword', 'N/A')}') in {filename}: {entry_err}", exc_info=False) # Keep log cleaner
                error_count += 1
                try:
                    # Rollback to the savepoint for this specific entry
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                except Exception as rb_err:
                    logger.critical(f"CRITICAL: Failed to rollback savepoint {savepoint_name} for {filename}: {rb_err}. Attempting full transaction rollback.")
                    conn.rollback() # Rollback the whole transaction for safety
                    logger.warning("Performed full transaction rollback due to savepoint rollback failure.")
                    # Consider re-raising the error to stop processing if this happens
                    # raise entry_err from rb_err
            finally:
                pbar.update(1) # Update progress bar regardless of success or failure

            # --- Periodic Commit (Optional but recommended for large files) ---
            entries_processed_so_far = processed_count + error_count + skipped_count
            if entries_processed_so_far > 0 and entries_processed_so_far % 500 == 0: # Commit every 500 entries
                 try:
                     conn.commit()
                     logger.debug(f"Committed progress for {filename} at entry {entry_index}")
                 except Exception as commit_err:
                     logger.error(f"Error committing progress for {filename} at entry {entry_index}: {commit_err}. Rolling back...")
                     conn.rollback()
                     # Log the error but continue processing remaining entries if possible
                     # The uncommitted batch will be retried or lost depending on subsequent commits/errors

    # Final commit for the file
    try:
        conn.commit()
        logger.info(f"Finished processing {filename}. Processed: {processed_count}, Definitions: {definitions_added}, Relations: {relations_added}, Pronunciations: {pronunciations_added}, Etymologies: {etymologies_added}, Skipped: {skipped_count}, Errors: {error_count}")
    except Exception as final_commit_err:
        logger.error(f"Error during final commit for {filename}: {final_commit_err}. Rolling back any remaining changes...")
        conn.rollback()
        error_count += 1 # Mark an error for the file due to final commit failure

    # Return counts: (successful, total issues)
    total_issues = error_count + skipped_count
    return processed_count, total_issues