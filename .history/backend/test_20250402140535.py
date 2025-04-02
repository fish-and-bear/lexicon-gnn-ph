# Replace the existing insert_definition function (starts around line 2924)
@with_transaction(commit=True)
def insert_definition(cur, word_id: int, definition_text: str,
                      source_identifier: str, # MANDATORY
                      part_of_speech: Optional[str] = None, # Changed default to None
                      examples: Optional[str] = None, usage_notes: Optional[str] = None,
                      tags: Optional[str] = None) -> Optional[int]:
    """
    Insert definition data for a word. Ensures source is updated on conflict.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        definition_text: The definition text.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.
        part_of_speech: Original part of speech string (optional).
        examples: Example sentences (optional).
        usage_notes: Notes on usage (optional).
        tags: Comma-separated tags string (optional).

    Returns:
        The ID of the inserted/updated definition record, or None if failed.
    """
    if not source_identifier:
        logger.error(f"CRITICAL: Skipping definition insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    if not definition_text:
        logger.warning(f"Empty definition text for word ID {word_id} from source '{source_identifier}'. Skipping.")
        return None

    try:
        # Get standardized part of speech ID
        standardized_pos_id = get_standardized_pos_id(cur, part_of_speech)

        # Truncate definition if too long for the database field (adjust length as needed)
        # Assuming definition_text max length is handled by DB or is sufficient. Add check if needed.
        # max_def_length = 4096
        # if len(definition_text) > max_def_length:
        #      logger.warning(f"Definition text for word ID {word_id} truncated from {len(definition_text)} to {max_def_length} characters.")
        #      definition_text = definition_text[:max_def_length]

        # Truncate examples if too long
        # Assuming examples max length is handled by DB or is sufficient. Add check if needed.
        # max_examples_length = 4096
        # if examples and len(examples) > max_examples_length:
        #      logger.warning(f"Examples for word ID {word_id} truncated from {len(examples)} to {max_examples_length} characters.")
        #      examples = examples[:max_examples_length]

        # Insert or update definition
        # Corrected ON CONFLICT clause to update sources
        cur.execute("""
            INSERT INTO definitions (word_id, definition_text, standardized_pos_id, examples, usage_notes, tags, sources, original_pos)
            VALUES (%(word_id)s, %(definition_text)s, %(pos_id)s, %(examples)s, %(usage_notes)s, %(tags)s, %(sources)s, %(original_pos)s)
            ON CONFLICT (word_id, definition_text, standardized_pos_id)
            DO UPDATE SET
                examples = COALESCE(EXCLUDED.examples, definitions.examples), -- Keep existing if new is NULL
                usage_notes = COALESCE(EXCLUDED.usage_notes, definitions.usage_notes), -- Keep existing if new is NULL
                tags = COALESCE(EXCLUDED.tags, definitions.tags), -- Keep existing if new is NULL
                sources = EXCLUDED.sources, -- Update sources field with the new source (last write wins)
                original_pos = COALESCE(EXCLUDED.original_pos, definitions.original_pos), -- Keep existing if new is NULL
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, {
            'word_id': word_id,
            'definition_text': definition_text,
            'pos_id': standardized_pos_id,
            'examples': examples,
            'usage_notes': usage_notes,
            'tags': tags,
            'sources': source_identifier, # Use the mandatory source_identifier
            'original_pos': part_of_speech # Store original POS string
        })
        def_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated definition (ID: {def_id}) for word ID {word_id} from source '{source_identifier}'.")
        return def_id

    except psycopg2.Error as e:
        logger.error(f"Database error inserting definition for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
        # No rollback needed here as @with_transaction handles it
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting definition for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        # No rollback needed here as @with_transaction handles it
        return None


# Replace the existing process_kwf_dictionary function (starts around line 3430)
@with_transaction(commit=False)  # Manage transactions manually
def process_kwf_dictionary(cur, filename: str):
    logger.info(f"Processing KWF dictionary file: {filename}")
    stats = {"entries": 0, "definitions": 0, "pronunciations": 0, "etymologies": 0, "relations": 0, "errors": 0}
    # Define source identifier for this function
    source_identifier = standardize_source_identifier(filename) # Or "KWF Diksiyonaryo ng Wikang Filipino"
    conn = cur.connection # Get connection for manual commit/rollback

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_entries = len(data)
        logger.info(f"Found {total_entries} entries in KWF dictionary.")

        for entry in tqdm(data, total=total_entries, desc="Processing KWF"):
            conn.rollback() # Start fresh transaction scope for entry using rollback

            try:
                lemma = entry.get('word', '').strip()
                if not lemma:
                    stats["errors"] += 1
                    logger.warning(f"Skipping KWF entry with empty word.")
                    continue

                language_code = 'tl' # Assuming KWF is Tagalog
                # Basic POS extraction, refine if needed
                pos = entry.get('definitions', [{}])[0].get('part_of_speech', '') if entry.get('definitions') else ''
                # Example check for proper noun - refine based on actual KWF POS tags
                is_proper_noun = 'pangngalang pantangi' in pos.lower() if pos else False

                # Get or create word ID
                word_id = get_or_create_word_id(
                    cur,
                    lemma,
                    language_code=language_code,
                    source_identifier=source_identifier, # Pass source_identifier
                    is_proper_noun=is_proper_noun,
                    # Pass other kwargs if available and needed by get_or_create_word_id
                    check_exists=False # Assuming default behavior is desired
                )
                if not word_id: # Handle case where word creation fails
                    stats["errors"] += 1
                    logger.error(f"Failed to get/create word ID for KWF entry: {lemma}")
                    continue

                stats["entries"] += 1

                # Process definitions
                if 'definitions' in entry and isinstance(entry['definitions'], list):
                    for definition_entry in entry['definitions']:
                        definition_text = definition_entry.get('definition', '').strip()
                        if not definition_text: continue
                        part_of_speech = definition_entry.get('part_of_speech', None)
                        examples_list = definition_entry.get('examples', [])
                        # KWF examples might be plain strings or dicts, adjust accordingly
                        examples = None
                        if examples_list:
                             if all(isinstance(ex, str) for ex in examples_list):
                                 examples = '\\n'.join(ex.strip() for ex in examples_list if ex.strip())
                             else: # Handle if examples are dicts e.g. {'text': '...', 'translation': '...'}
                                 example_texts = [ex.get('text', '').strip() for ex in examples_list if isinstance(ex, dict) and ex.get('text', '').strip()]
                                 examples = '\\n'.join(example_texts) if example_texts else None


                        try:
                            def_id = insert_definition(
                                cur,
                                word_id,
                                definition_text,
                                part_of_speech=part_of_speech,
                                examples=examples,
                                source_identifier=source_identifier # Pass source_identifier
                            )
                            if def_id:
                                stats["definitions"] += 1
                            else:
                                stats["errors"] += 1
                        except Exception as def_e:
                            logger.warning(f"Failed to insert KWF definition for '{lemma}': {def_e}")
                            stats["errors"] += 1


                # Process pronunciations (if KWF data includes it)
                # Assuming entry might have a 'pronunciation' key
                pron_data = entry.get('pronunciation') # Could be string or dict
                if pron_data:
                     try:
                          pron_id = insert_pronunciation(cur, word_id, pron_data, source_identifier=source_identifier)
                          if pron_id:
                              stats["pronunciations"] += 1
                     except Exception as pron_e:
                           logger.warning(f"Failed to insert KWF pronunciation for '{lemma}': {pron_e}")
                           stats["errors"] += 1


                # Process etymology (if KWF data includes it)
                # Assuming entry might have an 'etymology' key
                ety_text = entry.get('etymology')
                if ety_text:
                    try:
                        ety_id = insert_etymology(cur, word_id, ety_text, source_identifier=source_identifier)
                        if ety_id:
                            stats["etymologies"] += 1
                    except Exception as ety_e:
                         logger.warning(f"Failed to insert KWF etymology for '{lemma}': {ety_e}")
                         stats["errors"] += 1


                # Process relations (e.g., synonyms, if KWF data includes it)
                # Assuming entry might have a 'synonyms' key with a list of strings
                synonyms = entry.get('synonyms')
                if isinstance(synonyms, list):
                    for syn_word in synonyms:
                        syn_word_clean = syn_word.strip()
                        if syn_word_clean and syn_word_clean != lemma: # Avoid self-relation
                             try:
                                  # Get/create synonym word ID, ensuring source is passed
                                  syn_id = get_or_create_word_id(cur, syn_word_clean, language_code, source_identifier=source_identifier)
                                  if syn_id:
                                      rel_id = insert_relation(cur, word_id, syn_id, RelationshipType.SYNONYM, source_identifier)
                                      if rel_id:
                                           stats["relations"] += 1
                                      # Optionally insert inverse relation if needed
                                      # insert_relation(cur, syn_id, word_id, RelationshipType.SYNONYM, source_identifier)
                             except Exception as rel_e:
                                  logger.warning(f"Failed to insert KWF synonym relation for '{lemma}'->'{syn_word_clean}': {rel_e}")
                                  stats["errors"] += 1
                # Add similar logic for antonyms, related words etc. if present in KWF data

                conn.commit() # Commit successful entry processing

            except Exception as entry_e:
                logger.error(f"Error processing KWF entry '{entry.get('word', 'UNKNOWN')}': {entry_e}", exc_info=True)
                stats["errors"] += 1
                conn.rollback() # Rollback failed entry

        logger.info(f"Finished processing KWF: Processed {stats['entries']} entries, {stats['definitions']} definitions, {stats['pronunciations']} pronunciations, {stats['etymologies']} etymologies, {stats['relations']} relations, with {stats['errors']} errors.")
        return stats

    except FileNotFoundError:
        logger.error(f"KWF dictionary file not found: {filename}")
        return stats
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in KWF dictionary file: {filename}")
        return stats
    except Exception as e:
        logger.error(f"General error processing KWF dictionary: {e}", exc_info=True)
        try: cur.connection.rollback() # Ensure rollback on unexpected exit
        except: pass
        return stats


# Replace the existing process_tagalog_words function (starts around line 4063)
@with_transaction(commit=True) # Assuming commit=True is the desired behavior now
def process_tagalog_words(cur, filename: str):
    logger.info(f"Processing Tagalog Words (diksiyonaryo.ph) file: {filename}")
    stats = {"processed": 0, "skipped": 0, "errors": 0, "definitions_added": 0}
    # Define source identifier
    source_identifier = standardize_source_identifier(filename) # Or "diksiyonaryo.ph"

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_entries = len(data)
        logger.info(f"Found {total_entries} entries in {filename}")

        for entry in tqdm(data, total=total_entries, desc="Processing diksiyonaryo.ph"):
            try:
                lemma = entry.get('word', '').strip()
                if not lemma:
                    stats["skipped"] += 1
                    continue

                language_code = 'tl' # Assuming Tagalog

                # Get or create word ID
                word_id = get_or_create_word_id(
                    cur,
                    lemma,
                    language_code=language_code,
                    source_identifier=source_identifier # Pass source_identifier
                    # Add other kwargs if needed based on diksiyonaryo.ph data
                )
                if not word_id:
                    stats["errors"] += 1
                    logger.error(f"Failed to get/create word ID for diksiyonaryo.ph entry: {lemma}")
                    continue # Skip if word creation failed

                # Process definition(s)
                definition_text = entry.get('definition', '').strip()
                if definition_text:
                    # Try to extract POS if available, often in definition itself
                    # Example: pos_match = re.search(r'\((.*?)\)', definition_text)
                    # pos_in_def = pos_match.group(1).strip() if pos_match else None
                    # clean_definition = re.sub(r'\(.*?\)\s*', '', definition_text).strip() if pos_in_def else definition_text
                    pos_in_def = None # Placeholder - refine based on actual data format
                    clean_definition = definition_text # Assume definition is clean for now

                    try:
                        def_id = insert_definition(
                            cur,
                            word_id,
                            clean_definition,
                            part_of_speech=pos_in_def, # Pass extracted POS if any
                            source_identifier=source_identifier # Pass source_identifier
                            # Pass examples, tags etc. if available in entry
                        )
                        if def_id:
                             stats["definitions_added"] += 1
                        else:
                             stats["errors"] += 1
                    except Exception as def_e:
                         logger.warning(f"Failed to insert definition for '{lemma}' from {source_identifier}: {def_e}")
                         stats["errors"] += 1
                else:
                    # Only increment skipped if there was no definition *at all*
                    if 'definition' not in entry or not entry['definition']:
                        stats["skipped"] += 1

                stats["processed"] += 1

            except Exception as entry_e:
                logger.error(f"Error processing entry '{entry.get('word', 'UNKNOWN')}' from {filename}: {entry_e}", exc_info=True)
                stats["errors"] += 1
                # Since @with_transaction(commit=True) is used, error will cause rollback for this entry automatically.

        logger.info(f"Finished processing {filename}: {stats}")
        return stats

    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return stats
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filename}")
        return stats
    except Exception as e:
        logger.error(f"General error processing {filename}: {e}", exc_info=True)
        # The transaction decorator should handle rollback on general exceptions
        return stats


# Replace the existing process_root_words_cleaned function (starts around line 4381)
@with_transaction(commit=False)  # Manage transactions manually
def process_root_words_cleaned(cur, filename: str):
    logger.info(f"Processing Root Words (tagalog.com) file: {filename}")
    stats = {"roots_processed": 0, "definitions_added": 0, "relations_added": 0, "associated_processed": 0, "errors": 0}
    # Define source identifier
    source_identifier = standardize_source_identifier(filename) # Or "tagalog.com Root Words"
    conn = cur.connection

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_roots = len(data)
        logger.info(f"Found {total_roots} root word entries in {filename}")

        for root_word_entry in tqdm(data, total=total_roots, desc="Processing tagalog.com"):
            conn.rollback() # Start fresh transaction scope

            try:
                root_word = root_word_entry.get('root_word', '').strip()
                if not root_word:
                    stats["errors"] += 1
                    continue

                language_code = 'tl'

                # Get or create root word ID
                root_word_id = get_or_create_word_id(
                    cur,
                    root_word,
                    language_code=language_code,
                    source_identifier=source_identifier, # Pass source_identifier
                    is_root_word=True # Example: add a flag if your schema supports it
                )
                if not root_word_id:
                    stats["errors"] += 1
                    logger.error(f"Failed to get/create root word ID for tagalog.com entry: {root_word}")
                    continue
                stats["roots_processed"] += 1

                # Process definitions for the root word
                definitions = root_word_entry.get('definitions', [])
                if isinstance(definitions, list):
                     for definition_item in definitions:
                         # Handling definitions that might be strings or dicts
                         definition_text = None
                         part_of_speech = None
                         examples = None
                         if isinstance(definition_item, str):
                             definition_text = definition_item.strip()
                             # Potentially extract POS from string definition if pattern exists
                         elif isinstance(definition_item, dict):
                             definition_text = definition_item.get('text', '').strip() or definition_item.get('definition', '').strip()
                             part_of_speech = definition_item.get('pos') or definition_item.get('part_of_speech')
                             # Extract examples if present, format as needed
                             raw_examples = definition_item.get('examples')
                             if isinstance(raw_examples, list):
                                 examples = '\n'.join(ex.strip() for ex in raw_examples if ex.strip())
                             elif isinstance(raw_examples, str):
                                 examples = raw_examples.strip()


                         if definition_text:
                             try:
                                 def_id = insert_definition(
                                     cur,
                                     root_word_id,
                                     definition_text,
                                     part_of_speech=part_of_speech,
                                     examples=examples,
                                     source_identifier=source_identifier # Pass source_identifier
                                 )
                                 if def_id:
                                     stats["definitions_added"] += 1
                                 else:
                                     stats["errors"] += 1
                             except Exception as def_e:
                                 logger.warning(f"Failed to insert tagalog.com definition for root '{root_word}': {def_e}")
                                 stats["errors"] += 1

                # Process associated words (assuming they are derived)
                associated_words = root_word_entry.get('associated_words', [])
                if isinstance(associated_words, list):
                    for assoc_word_entry in associated_words:
                        assoc_word = None
                        # Check if entry is string or dict
                        if isinstance(assoc_word_entry, str):
                            assoc_word = assoc_word_entry.strip()
                        elif isinstance(assoc_word_entry, dict):
                             assoc_word = assoc_word_entry.get('word', '').strip()
                             # Could potentially add definitions for associated words too

                        if assoc_word and assoc_word != root_word:
                            stats["associated_processed"] += 1
                            try:
                                assoc_word_id = get_or_create_word_id(
                                    cur,
                                    assoc_word,
                                    language_code=language_code,
                                    root_word_id=root_word_id, # Link derived word to root
                                    source_identifier=source_identifier # Pass source_identifier
                                )
                                if assoc_word_id:
                                     # Add DERIVED_FROM relationship
                                     rel_id = insert_relation(cur, assoc_word_id, root_word_id, RelationshipType.DERIVED_FROM, source_identifier)
                                     if rel_id:
                                         stats["relations_added"] += 1
                                     # Optionally add ROOT_OF relation from root to derived
                                     # insert_relation(cur, root_word_id, assoc_word_id, RelationshipType.ROOT_OF, source_identifier)

                            except Exception as assoc_e:
                                logger.warning(f"Failed to process/relate associated word '{assoc_word}' for root '{root_word}': {assoc_e}")
                                stats["errors"] += 1

                conn.commit() # Commit successful root word processing

            except Exception as entry_e:
                logger.error(f"Error processing root word entry '{root_word_entry.get('root_word', 'UNKNOWN')}': {entry_e}", exc_info=True)
                stats["errors"] += 1
                conn.rollback() # Rollback failed entry

        logger.info(f"Finished processing {filename}: {stats}")
        return stats

    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return stats
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filename}")
        return stats
    except Exception as e:
        logger.error(f"General error processing {filename}: {e}", exc_info=True)
        try: cur.connection.rollback()
        except: pass
        return stats


# Replace the existing process_marayum_json function (starts around line 6708)
@with_transaction(commit=False)  # Manage transactions manually within the loop
def process_marayum_json(cur, filename: str) -> Tuple[int, int]:
    logger.info(f"Processing Marayum JSON file: {filename}")
    processed_count = 0
    error_count = 0
    definitions_added = 0
    # Define source identifier
    source_identifier = standardize_source_identifier(filename) # Or "Project Marayum"
    conn = cur.connection

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error(f"Marayum file {filename} does not contain a list of entries.")
            return 0, 0

        total_entries = len(data)
        logger.info(f"Found {total_entries} entries in {filename}")

        for entry in tqdm(data, total=total_entries, desc=f"Processing {os.path.basename(filename)}"):
            conn.rollback() # Start fresh transaction scope

            try:
                headword = entry.get('headword', '').strip()
                if not headword:
                    error_count += 1
                    logger.warning("Skipping Marayum entry with no headword.")
                    continue

                # Determine language code (using get_language_code helper)
                language_name = entry.get('language', 'Tagalog') # Default to Tagalog if missing
                language_code = get_language_code(language_name) # Use helper function

                # Get or create word ID
                word_id = get_or_create_word_id(
                    cur,
                    headword,
                    language_code=language_code,
                    source_identifier=source_identifier # Pass source_identifier
                )
                if not word_id:
                    error_count += 1
                    logger.error(f"Failed to get/create word ID for Marayum entry: {headword} ({language_name})")
                    continue

                # Process definitions (assuming 'definitions' is a list of strings or dicts)
                definitions = entry.get('definitions', [])
                if isinstance(definitions, list):
                    for definition_item in definitions:
                         definition_text = None
                         part_of_speech = None
                         examples = None
                         usage_notes = None

                         if isinstance(definition_item, str):
                             definition_text = definition_item.strip()
                             # Try to extract POS like [noun], [verb] etc.
                             pos_match = re.match(r"^\s*\[([^\]]+)\]", definition_text)
                             if pos_match:
                                 part_of_speech = pos_match.group(1).strip()
                                 definition_text = re.sub(r"^\s*\[[^\]]+\]\s*", "", definition_text)
                         elif isinstance(definition_item, dict):
                             # Adapt if Marayum definitions are structured dicts
                             definition_text = definition_item.get('text', '').strip()
                             part_of_speech = definition_item.get('pos')
                             # Extract examples, usage_notes if they exist in the dict

                         if definition_text:
                              try:
                                   def_id = insert_definition(
                                       cur,
                                       word_id,
                                       definition_text,
                                       part_of_speech=part_of_speech,
                                       examples=examples, # Pass extracted examples
                                       usage_notes=usage_notes, # Pass extracted notes
                                       source_identifier=source_identifier # Pass source_identifier
                                   )
                                   if def_id:
                                       definitions_added += 1
                                   else:
                                       error_count += 1
                              except Exception as def_e:
                                   logger.warning(f"Failed to insert Marayum definition for '{headword}': {def_e}")
                                   error_count += 1

                # Process other fields if they exist (pronunciation, examples, relations etc.)
                # Example: Pronunciation
                pronunciation = entry.get('pronunciation')
                if pronunciation:
                    try:
                         # Assuming insert_pronunciation exists and takes source_identifier
                         insert_pronunciation(cur, word_id, pronunciation, source_identifier)
                    except Exception as pron_e:
                         logger.warning(f"Failed to insert Marayum pronunciation for '{headword}': {pron_e}")
                         error_count += 1

                # Example: Relations (like synonyms)
                synonyms = entry.get('synonyms')
                if isinstance(synonyms, list):
                     for syn_word in synonyms:
                         syn_word = syn_word.strip()
                         if syn_word and syn_word != headword:
                              try:
                                   syn_id = get_or_create_word_id(cur, syn_word, language_code, source_identifier=source_identifier)
                                   if syn_id:
                                        # Assuming insert_relation exists and takes source_identifier
                                        insert_relation(cur, word_id, syn_id, RelationshipType.SYNONYM, source_identifier)
                              except Exception as rel_e:
                                   logger.warning(f"Failed to insert Marayum synonym for '{headword}' -> '{syn_word}': {rel_e}")
                                   error_count += 1


                processed_count += 1
                conn.commit() # Commit successful entry

            except Exception as e:
                error_count += 1
                logger.error(f"Error processing Marayum entry '{entry.get('headword', 'UNKNOWN')}': {e}", exc_info=True)
                conn.rollback() # Rollback failed entry

        logger.info(f"Finished processing {filename}: Processed {processed_count}, Definitions Added {definitions_added}, Errors {error_count}")
        return processed_count, error_count

    except FileNotFoundError:
        logger.error(f"Marayum file not found: {filename}")
        return 0, 0
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in Marayum file: {filename}")
        return 0, 0
    except Exception as e:
        logger.error(f"General error processing Marayum file {filename}: {e}", exc_info=True)
        try: cur.connection.rollback()
        except: pass
        return 0, 0