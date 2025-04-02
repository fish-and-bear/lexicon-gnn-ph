\
    # Process a single entry
    def process_entry(cur, entry: Dict, source_identifier: str):
        """Process a single dictionary entry with improved resilience."""
        if 'word' not in entry or not entry['word']:
            logger.warning("Skipping entry without valid 'word' field")
            return None, "No word field" # Indicate critical failure

        word = entry['word']
        word_id = None # Initialize word_id
        critical_error_occurred = False
        error_messages = [] # Collect non-critical errors

        try:
            pos = entry.get('pos', 'unc')
            language_code = entry.get('lang_code', DEFAULT_LANGUAGE_CODE)
            if not language_code or len(language_code) > 10:
                 language_code = DEFAULT_LANGUAGE_CODE
                 logger.warning(f"Invalid/missing lang_code for '{word}', defaulting to '{DEFAULT_LANGUAGE_CODE}'.")

            is_proper_noun = entry.get('proper', False) or pos in ['prop', 'proper noun', 'name']
            is_abbreviation = pos in ['abbrev', 'abbreviation']
            is_initialism = pos in ['init', 'initialism', 'acronym']
            tags_list = entry.get('tags', [])
            if isinstance(tags_list, list):
                if any(t in ['abbreviation', 'abbrev'] for t in tags_list): is_abbreviation = True
                if any(t in ['initialism', 'acronym'] for t in tags_list): is_initialism = True

            baybayin_form, romanized_form = extract_baybayin_info(entry)
            badlit_form, badlit_romanized = extract_badlit_info(entry)
            hyphenation = entry.get('hyphenation')
            hyphenation_json = json.dumps(hyphenation) if hyphenation and isinstance(hyphenation, list) else None
            word_tags_str = ','.join(tags_list) if tags_list else None

            # --- CRITICAL STEP: Get or Create Word ID ---
            try:
                 word_id = get_or_create_word_id(
                    cur, lemma=word, language_code=language_code, has_baybayin=bool(baybayin_form),
                    baybayin_form=baybayin_form, romanized_form=romanized_form or badlit_romanized,
                    badlit_form=badlit_form, hyphenation=hyphenation_json, is_proper_noun=is_proper_noun,
                    is_abbreviation=is_abbreviation, is_initialism=is_initialism, tags=word_tags_str,
                    source_identifier=source_identifier
                )
            except Exception as word_create_err:
                 logger.error(f"CRITICAL FAILURE: Could not get/create word_id for '{word}' ({language_code}). Error: {word_create_err}. Skipping entry.")
                 critical_error_occurred = True
                 # No point continuing if we don't have a word_id
                 return None, f"Word creation failed: {word_create_err}"

            # --- Process Additional Word-Level Data (Non-Critical Failures) ---
            # Wrap each subsequent step in its own try-except block
            try:
                process_pronunciation(cur, word_id, entry, source_identifier)
            except Exception as e:
                msg = f"Failed processing pronunciation for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)

            try:
                process_form_relationships(cur, word_id, entry, language_code, source_identifier)
            except Exception as e:
                msg = f"Failed processing form relationships for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)

            try:
                if 'etymology_text' in entry:
                    process_etymology(cur, word_id, entry['etymology_text'], source_identifier, entry.get('etymology_templates'))
            except Exception as e:
                msg = f"Failed processing etymology for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)

            try:
                if 'head_templates' in entry and entry['head_templates']:
                    process_head_templates(cur, word_id, entry['head_templates'])
            except Exception as e:
                msg = f"Failed processing head templates for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)


            # --- Process Definitions (Senses) ---
            sense_processed_count = 0
            if 'senses' in entry and isinstance(entry['senses'], list):
                for sense_idx, sense in enumerate(entry['senses']):
                    if not sense or not isinstance(sense, dict):
                        logger.debug(f"Skipping invalid sense item {sense_idx} for word ID {word_id}")
                        continue

                    definition_id = None # Reset for each sense
                    try:
                        glosses = sense.get('glosses', []) or sense.get('raw_glosses', [])
                        if not glosses or not isinstance(glosses, list):
                            logger.debug(f"Skipping sense {sense_idx} for word ID {word_id}: missing/invalid glosses.")
                            continue
                        definition_text = '; '.join([g for g in glosses if isinstance(g, str)])
                        if not definition_text:
                            logger.debug(f"Skipping sense {sense_idx} for word ID {word_id}: empty definition text.")
                            continue

                        max_def_length = 4096
                        if len(definition_text) > max_def_length:
                            logger.warning(f"Definition {sense_idx} for word ID {word_id} truncated.")
                            definition_text = definition_text[:max_def_length]

                        examples = []
                        if 'examples' in sense and isinstance(sense['examples'], list):
                            for example in sense['examples']:
                                ex_text = None
                                if isinstance(example, str): ex_text = example
                                elif isinstance(example, dict):
                                    ex_text = example.get('text')
                                    ref = example.get('ref')
                                    tr = example.get('translation') or example.get('english')
                                    if ex_text and tr: ex_text += f" - {tr}"
                                    if ex_text and ref: ex_text += f" (Ref: {ref})"
                                if ex_text: examples.append(ex_text)
                        examples_str = '\n'.join(examples) if examples else None

                        sense_tags = sense.get('tags', [])
                        sense_labels = sense.get('labels', [])
                        all_tags = (sense_tags if isinstance(sense_tags, list) else []) + \
                                   (sense_labels if isinstance(sense_labels, list) else [])
                        tags_str = ','.join(all_tags) if all_tags else None
                        usage_notes = None

                        metadata_dict = {}
                        for key in ['form_of', 'raw_glosses', 'topics', 'taxonomy', 'qualifier']:
                            if key in sense and sense[key]: metadata_dict[key] = sense[key]

                        sense_pos_str = sense.get('pos') or pos
                        standard_pos = standardize_entry_pos(sense_pos_str)

                        # --- Insert Definition ---
                        try:
                            definition_id = insert_definition(
                                cur, word_id, definition_text, part_of_speech=standard_pos,
                                examples=examples_str, usage_notes=usage_notes, tags=tags_str,
                                source_identifier=source_identifier
                            )
                            sense_processed_count += 1
                        except psycopg2.errors.UniqueViolation:
                             logger.debug(f"Definition already exists for word ID {word_id}, sense {sense_idx}")
                             # Try to fetch existing ID if needed later? For now, skip sub-processing.
                             definition_id = None # Ensure it's None if insertion failed due to conflict
                        except Exception as def_error:
                            msg = f"Failed inserting definition for word ID {word_id}, sense {sense_idx}: {def_error}"
                            logger.warning(msg)
                            error_messages.append(msg)
                            definition_id = None # Ensure it's None if insertion failed

                        # --- Process Definition Sub-Data (only if definition_id is valid) ---
                        if definition_id:
                            try:
                                if metadata_dict:
                                     cur.execute("""
                                        UPDATE definitions SET metadata = COALESCE(metadata, '{}'::jsonb) || %s
                                        WHERE id = %s
                                     """, (json.dumps(metadata_dict), definition_id))
                            except Exception as meta_err:
                                msg = f"Failed storing metadata for definition {definition_id}: {meta_err}"
                                logger.warning(msg)
                                error_messages.append(msg)

                            try:
                                if 'categories' in sense and sense['categories']:
                                    process_categories(cur, definition_id, sense['categories'])
                            except Exception as e:
                                msg = f"Failed processing categories for def ID {definition_id}: {e}"
                                logger.warning(msg)
                                error_messages.append(msg)

                            try:
                                if 'links' in sense and sense['links']:
                                    process_links(cur, definition_id, sense['links'])
                            except Exception as e:
                                msg = f"Failed processing links for def ID {definition_id}: {e}"
                                logger.warning(msg)
                                error_messages.append(msg)

                            try:
                                process_sense_relationships(cur, word_id, sense, language_code, source_identifier)
                            except Exception as e:
                                msg = f"Failed processing sense relationships for word ID {word_id}, def ID {definition_id}: {e}"
                                logger.warning(msg)
                                error_messages.append(msg)

                    except Exception as sense_proc_err:
                        # Catch errors during preparation for definition insert (e.g., processing examples)
                        msg = f"Error processing sense {sense_idx} for word ID {word_id} before definition insert: {sense_proc_err}"
                        logger.warning(msg)
                        error_messages.append(msg)
                        continue # Skip to the next sense

                if sense_processed_count == 0 and not critical_error_occurred:
                     # Log if no definitions were processed *successfully*, but don't treat as error unless word creation also failed
                     logger.info(f"No definitions successfully inserted/processed for word '{word}' (ID: {word_id}). Senses might be empty or all failed individually.")


            # --- Process Top-Level Derived/Related (Non-Critical) ---
            try:
                top_level_rels = {'derived': RelationshipType.ROOT_OF, 'related': RelationshipType.RELATED}
                for rel_key, rel_enum in top_level_rels.items():
                    if rel_key in entry and isinstance(entry[rel_key], list):
                        for item in entry[rel_key]:
                            if isinstance(item, dict) and 'word' in item:
                                related_word = item['word']
                                if not related_word or not isinstance(related_word, str): continue
                                try:
                                    related_word_id = get_or_create_word_id(cur, related_word, language_code, source_identifier=source_identifier)
                                    if related_word_id == word_id: continue
                                    metadata = {'confidence': rel_enum.strength, 'from_sense': False}
                                    insert_relation(cur, word_id, related_word_id, rel_enum, source_identifier, metadata)
                                    if rel_enum.bidirectional: insert_relation(cur, related_word_id, word_id, rel_enum, source_identifier, metadata)
                                    else:
                                         inv = rel_enum.get_inverse()
                                         if inv: insert_relation(cur, related_word_id, word_id, inv, source_identifier, metadata)
                                except Exception as top_rel_err_inner:
                                    msg = f"Failed processing top-level {rel_key} relation for word ID {word_id} -> '{related_word}': {top_rel_err_inner}"
                                    logger.warning(msg)
                                    error_messages.append(msg) # Log inner failure but continue loop
            except Exception as e:
                # Catch error in the loop setup itself
                msg = f"Failed processing top-level relationships section for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)

            # Return word_id (even if non-critical errors occurred) and collected error messages
            return word_id, "; ".join(error_messages) if error_messages else None

        except Exception as e:
            # Catch truly unexpected errors during the main processing flow (after word_id creation attempt)
            logger.error(f"UNEXPECTED EXCEPTION during processing entry for word '{word}' (ID: {word_id}): {str(e)}", exc_info=True)
            # If word_id was created, return it but indicate a major issue occurred.
            # If word_id wasn't created (should have been caught earlier), this path might not be reached,
            # but handle defensively.
            return word_id if word_id else None, f"Unhandled entry exception: {e}"

// ... existing code ...
    # Main function processing logic
    stats = {
        "total_entries": total_lines if total_lines != -1 else 0, # Use counted lines if available
        "processed_ok": 0,
        "processed_with_errors": 0, # Entries processed but had some non-critical error
        "failed_entries": 0, # Entries that couldn't be processed at all
        "skipped_json_errors": 0,
        "fallback_entries_used": 0 # Count how many times fallback was triggered - REMOVED FALLBACK CALL
    }
    error_summary = {} # Track types of errors encountered

    # Ensure connection is valid before starting loop
    if conn.closed:
        logger.error("Database connection is closed before starting Kaikki processing loop.")
        stats["failed_entries"] = stats["total_entries"]
        return stats

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            entry_count = 0
            progress_bar = tqdm(total=total_lines, desc=f"Processing {os.path.basename(filename)}", unit=" entries") if total_lines > 0 else None

            for line in f:
                entry_count += 1
                if progress_bar: progress_bar.update(1)

                # --- Transaction Management: SAVEPOINT per entry ---
                savepoint_name = f"kaikki_entry_{entry_count}"
                try:
                    cur.execute(f"SAVEPOINT {savepoint_name}")
                except Exception as sp_error:
                     logger.error(f"CRITICAL: Failed to create savepoint {savepoint_name}. Aborting processing. Error: {sp_error}")
                     stats["failed_entries"] = stats["total_entries"] - entry_count + 1 # Mark remaining as failed
                     if progress_bar: progress_bar.close()
                     return stats # Stop processing

                try:
                    entry = json.loads(line.strip())

                    # Try standard processing first
                    word_id, entry_errors = process_entry(cur, entry, source_identifier) # Returns word_id and potential errors

                    if word_id: # Word was created/found, proceed even if non-critical errors occurred
                        if entry_errors:
                            stats["processed_with_errors"] += 1
                            # Log first part of error summary key
                            first_error = entry_errors.split(';')[0][:100] # Limit key length
                            err_key = f"EntryError: {first_error}"
                            error_summary[err_key] = error_summary.get(err_key, 0) + 1
                            logger.debug(f"Entry {entry_count} ('{entry.get('word', 'N/A')}') processed with errors: {entry_errors}")
                        else:
                            stats["processed_ok"] += 1
                        # Commit successful or partially successful processing for this entry
                        cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    else:
                        # Critical failure within process_entry (likely word creation failed)
                        stats["failed_entries"] += 1
                        reason = entry_errors or "Unknown critical failure" # entry_errors should contain the reason
                        err_key = f"EntryFailedCritically: {reason[:100]}"
                        error_summary[err_key] = error_summary.get(err_key, 0) + 1
                        logger.warning(f"Entry {entry_count} ('{entry.get('word', 'N/A')}') failed critical processing step: {reason}")
                        # Rollback this specific entry
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        # NOTE: Fallback logic removed; process_entry is now resilient.

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {entry_count}")
                    stats["skipped_json_errors"] += 1
                    # Rollback if savepoint might exist
                    try: cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception: pass # Ignore rollback error if savepoint doesn't exist
                except Exception as e:
                    logger.error(f"Unhandled error in main loop at line {entry_count}: {str(e)}", exc_info=True)
                    stats["failed_entries"] += 1
                    err_key = f"LoopException: {type(e).__name__}"
                    error_summary[err_key] = error_summary.get(err_key, 0) + 1
                    # Rollback potentially affected entry
                    try:
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception as rb_err:
                        logger.error(f"Error rolling back savepoint {savepoint_name} after loop exception: {rb_err}")

            if progress_bar: progress_bar.close()
// ... rest of the function ...
