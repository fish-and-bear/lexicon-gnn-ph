@with_transaction(commit=False)  # Manage transactions manually
def process_kaikki_jsonl(cur, filename: str):
    """
    Process Kaikki.org dictionary entries with enhanced source tracking.
    
    Args:
        cur: Database cursor
        filename: Path to Kaikki JSONL file
        
    Returns:
        Dictionary with processing statistics
    """
    # Standardize source identifier consistently
    raw_source_identifier = os.path.basename(filename)
    source_identifier = SourceStandardization.standardize_sources(raw_source_identifier)
    
    # Make sure we have a valid source identifier
    if not source_identifier:
        if 'ceb' in raw_source_identifier.lower():
            source_identifier = "kaikki.org (Cebuano)"
        else:
            source_identifier = "kaikki.org (Tagalog)"
    
    logger.info(f"Processing Kaikki.org dictionary: {filename}")
    logger.info(f"Using standardized source identifier: '{source_identifier}'")
    
    # Initialize statistics with more detail
    stats = {
        "total_entries": 0,
        "processed_entries": 0,
        "error_entries": 0,
        "skipped_entries": 0,
        "fallback_entries": 0,
        "definitions_added": 0,
        "etymologies_added": 0,
        "pronunciations_added": 0,
        "relations_added": 0,
        "baybayin_processed": 0,
        "categories_added": 0
    }
    
    # First count total lines in file for progress reporting
    total_lines = sum(1 for _ in open(filename, 'r', encoding='utf-8'))
    stats["total_entries"] = total_lines
    logger.info(f"Found {total_lines} entries to process")

    # [Schema setup code - unchanged]
    # ...

    # Commit schema changes
    conn = cur.connection
    try:
        conn.commit()
        logger.info("Successfully committed schema changes")
    except Exception as e:
        logger.error(f"Error committing schema changes: {str(e)}")
        conn.rollback()
        return stats
    
    # Create source metadata object
    source_metadata = {
        "name": source_identifier,
        "file": raw_source_identifier,
        "type": "kaikki_jsonl",
        "processed_timestamp": datetime.now().isoformat(),
        "language": "Cebuano" if 'ceb' in raw_source_identifier.lower() else "Tagalog"
    }
    
    # Convert to JSON string for storage
    source_metadata_json = json.dumps(source_metadata)
    
    # Function to get standard relationship type - unchanged
    def get_standard_relationship_type(rel_type):
        """Convert custom relationship types to standard ones defined in RelationshipType enum"""
        if rel_type in RELATIONSHIP_TYPE_MAPPING:
            return RELATIONSHIP_TYPE_MAPPING[rel_type]
        
        # Check if it's already a standard type
        for rel in RelationshipType:
            if rel.rel_value == rel_type:
                return rel_type
                
        logger.warning(f"Unknown relationship type: {rel_type}, using RELATED as fallback")
        return RelationshipType.RELATED.rel_value
    
    # Extract Baybayin info with proper cleaning - unchanged
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract Baybayin script form from an entry, cleaning prefixes."""
        # [Function implementation unchanged]
        # ...
    
    # Process pronunciation with enhanced source tracking
    def process_pronunciation(cur, word_id: int, entry: Dict):
        """Process pronunciation data with source tracking."""
        if 'sounds' not in entry:
            return 0  # Return count of added pronunciations
        
        pronunciations_added = 0
        
        # [Rest of function unchanged, but add source_identifier to metadata]
        for sound in entry['sounds']:
            if 'ipa' in sound:
                # Store IPA pronunciation
                tags = sound.get('tags', [])
                
                # Associate this with pronunciation styles and source
                metadata = {
                    "source": source_identifier,
                    "styles": list(pronunciation_styles) if pronunciation_styles else []
                }
                
                try:
                    cur.execute("""
                        INSERT INTO pronunciations (word_id, type, value, tags, metadata)
                        VALUES (%s, 'ipa', %s, %s, %s)
                        ON CONFLICT (word_id, type, value) DO NOTHING
                    """, (
                        word_id, 
                        sound['ipa'], 
                        json.dumps(tags) if tags else None,
                        json.dumps(metadata)
                    ))
                    pronunciations_added += 1
                except Exception as e:
                    logger.error(f"Error inserting pronunciation for word ID {word_id}: {str(e)}")
            # [Rest of function unchanged]
            
        return pronunciations_added
    
    # Process etymology with enhanced source tracking
    def process_etymology(cur, word_id: int, etymology_text: str, etymology_templates: List[Dict] = None):
        """Process etymology information for a word with source tracking."""
        # Skip if etymology text is empty
        if not etymology_text:
            return 0  # Return whether etymology was added
        
        # Add source to etymology structure
        etymology_structure = {
            "source": source_identifier,
            "processed_timestamp": datetime.now().isoformat()
        }
        
        # [Rest of function, but change standardized_source to source_identifier]
        # ...

        try:
            # [Rest of function unchanged, but use source_identifier instead of "kaikki"]
            # ...
            return 1  # Indicate etymology was added
        except Exception as e:
            logger.error(f"Error saving etymology for word ID {word_id}: {str(e)}")
            return 0  # Indicate etymology failed
    
    # Process a single entry with enhanced source tracking
    def process_entry(cur, entry: Dict):
        """Process a single dictionary entry with enhanced source tracking."""
        if 'word' not in entry:
            logger.warning("Skipping entry without 'word' field")
            return None
            
        try:
            word = entry['word']
            pos = entry.get('pos', '')
            language_code = entry.get('lang_code', 'tl')  # Default to Tagalog if not specified
            
            # [Rest of the function setup unchanged]
            # ...
            
            # Create word metadata with source information
            word_metadata = {
                "source": source_identifier,
                "kaikki_pos": pos,
                "processed_timestamp": datetime.now().isoformat()
            }
            
            # Add special flags to metadata
            if is_proper_noun:
                word_metadata["is_proper_noun"] = True
            if is_abbreviation:
                word_metadata["is_abbreviation"] = True
            if is_initialism:
                word_metadata["is_initialism"] = True
            
            # Convert word metadata to JSON
            word_metadata_json = json.dumps(word_metadata)
            
            # Get or create the word with enhanced metadata
            word_id = get_or_create_word_id(
                cur, 
                word, 
                language_code=language_code,
                has_baybayin=bool(baybayin_form),
                baybayin_form=baybayin_form,
                romanized_form=romanized_form or badlit_romanized,
                badlit_form=badlit_form,
                hyphenation=hyphenation_json,
                is_proper_noun=is_proper_noun,
                is_abbreviation=is_abbreviation,
                is_initialism=is_initialism,
                source_identifier=source_identifier,  # Pass source explicitly
                word_metadata=word_metadata_json,
                tags=','.join(entry.get('tags', [])) if 'tags' in entry else None
            )
            
            # Track statistics
            entry_stats = {
                "pronunciations_added": 0,
                "etymologies_added": 0,
                "definitions_added": 0,
                "relations_added": 0,
                "categories_added": 0
            }
            
            # Process pronunciation information with tracking
            entry_stats["pronunciations_added"] += process_pronunciation(cur, word_id, entry)
            
            # Process form relationships
            process_form_relationships(cur, word_id, entry, language_code)
            
            # Process etymologies with tracking
            if 'etymology_text' in entry:
                entry_stats["etymologies_added"] += process_etymology(
                    cur, 
                    word_id, 
                    entry['etymology_text'], 
                    entry.get('etymology_templates')
                )
            
            # [Process head templates - unchanged]
            # ...
            
            # Process definitions from senses with enhanced metadata
            if 'senses' in entry:
                for sense_idx, sense in enumerate(entry['senses']):
                    if not sense or not isinstance(sense, dict):
                        continue
                    
                    glosses = sense.get('glosses', [])
                    if not glosses:
                        continue
                    
                    # Create definition metadata with source tracking
                    def_metadata = {
                        "source": source_identifier,
                        "sense_index": sense_idx,
                        "kaikki_id": sense.get('id'),
                        "processed_timestamp": datetime.now().isoformat()
                    }
                    
                    # Add other metadata fields from sense
                    for key in ['form_of', 'raw_glosses', 'topics', 'taxonomy']:
                        if key in sense and sense[key]:
                            def_metadata[key] = sense[key]
                    
                    # [Rest of definition processing, but add def_metadata to insert_definition]
                    # ...
                    
                    # Insert the definition with source metadata
                    try:
                        definition_id = insert_definition(
                            cur, 
                            word_id, 
                            definition_text, 
                            part_of_speech=standard_pos,
                            examples=examples_str,
                            usage_notes=usage_notes,
                            tags=','.join(tags) if tags else None,
                            source_identifier=source_identifier,  # Pass source explicitly
                            metadata=json.dumps(def_metadata)
                        )
                        
                        if definition_id:
                            entry_stats["definitions_added"] += 1
                            
                            # [Process categories and links with tracking]
                            # ...
                            
                            # Process relationships with source tracking
                            relation_count = process_sense_relationships(cur, word_id, sense, source_identifier)
                            entry_stats["relations_added"] += relation_count
                    except Exception as e:
                        logger.error(f"Error inserting definition for word ID {word_id}: {str(e)}")
            
            # Update the global stats
            stats["definitions_added"] += entry_stats["definitions_added"]
            stats["etymologies_added"] += entry_stats["etymologies_added"]
            stats["pronunciations_added"] += entry_stats["pronunciations_added"]
            stats["relations_added"] += entry_stats["relations_added"]
            stats["categories_added"] += entry_stats["categories_added"]
            
            return word_id
        except Exception as e:
            # Log the error but don't propagate it
            logger.error(f"Error processing entry for word '{entry.get('word', 'unknown')}': {str(e)}")
            return None
    
    # Update process_sense_relationships to include source tracking and return count
    def process_sense_relationships(cur, word_id: int, sense: Dict, source: str):
        """Process relationships from a sense with source tracking."""
        relations_added = 0
        
        # For each relationship type (synonyms, antonyms, etc.)
        for rel_type, rel_list_key in [
            (RelationshipType.SYNONYM.rel_value, 'synonyms'),
            (RelationshipType.ANTONYM.rel_value, 'antonyms'),
            (RelationshipType.HYPERNYM.rel_value, 'hypernyms'),
            (RelationshipType.HYPONYM.rel_value, 'hyponyms'),
            (RelationshipType.HOLONYM.rel_value, 'holonyms'),
            (RelationshipType.MERONYM.rel_value, 'meronyms'),
            (RelationshipType.ROOT_OF.rel_value, 'derived'),
            (RelationshipType.SEE_ALSO.rel_value, 'see_also')
        ]:
            if rel_list_key in sense and isinstance(sense[rel_list_key], list):
                for rel_idx, rel_item in enumerate(sense[rel_list_key]):
                    if isinstance(rel_item, dict) and 'word' in rel_item:
                        try:
                            rel_word = rel_item['word']
                            rel_word_id = get_or_create_word_id(
                                cur, 
                                rel_word, 
                                language_code='tl',
                                source_identifier=source
                            )
                            
                            # Create metadata with source tracking
                            metadata = {
                                'source': source,
                                'confidence': get_confidence_for_relation(rel_type),
                                'sense_id': sense.get('id'),
                                'index': rel_idx,
                                'processed_timestamp': datetime.now().isoformat()
                            }
                            
                            # Add tags if available
                            if 'tags' in rel_item and isinstance(rel_item['tags'], list):
                                metadata['tags'] = rel_item['tags']
                            
                            # Insert relation with source tracking
                            if insert_relation(
                                cur, 
                                word_id, 
                                rel_word_id, 
                                rel_type, 
                                source_identifier=source,  # Pass source explicitly
                                metadata=metadata
                            ):
                                relations_added += 1
                                
                        except Exception as e:
                            logger.error(f"Error processing {rel_list_key} relation for word ID {word_id}: {str(e)}")
        
        return relations_added
    
    # Helper function for relation confidence
    def get_confidence_for_relation(rel_type: str) -> int:
        """Return confidence score for different relation types."""
        confidence_map = {
            RelationshipType.SYNONYM.rel_value: 90,
            RelationshipType.ANTONYM.rel_value: 90,
            RelationshipType.HYPERNYM.rel_value: 85,
            RelationshipType.HYPONYM.rel_value: 85,
            RelationshipType.HOLONYM.rel_value: 80,
            RelationshipType.MERONYM.rel_value: 80,
            RelationshipType.ROOT_OF.rel_value: 95,
            RelationshipType.SEE_ALSO.rel_value: 70,
            RelationshipType.RELATED.rel_value: 70,
            RelationshipType.VARIANT.rel_value: 85,
            RelationshipType.SPELLING_VARIANT.rel_value: 90,
            RelationshipType.REGIONAL_VARIANT.rel_value: 85
        }
        return confidence_map.get(rel_type, 70)  # Default to 70 if not found
    
    # Main processing logic with progress bar
    try:
        with tqdm(total=total_lines, desc=f"Processing {os.path.basename(filename)}", unit="entry") as pbar:
            with open(filename, 'r', encoding='utf-8') as f:
                # Process entries one by one with a fresh transaction for each
                entry_count = 0
                
                for line in f:
                    try:
                        # Make sure we're in a clean transaction state for each entry
                        conn.rollback()
                        
                        entry = json.loads(line.strip())
                        entry_count += 1
                        
                        # Create savepoint for this entry
                        savepoint_name = f"kaikki_entry_{entry_count}"
                        cur.execute(f"SAVEPOINT {savepoint_name}")
                        
                        # Try standard processing first
                        result = process_entry(cur, entry)
                        
                        # If failed, try fallback method without Baybayin
                        if not result and 'forms' in entry:
                            for form in entry.get('forms', []):
                                if 'tags' in form and 'Baybayin' in form.get('tags', []):
                                    # Only use fallback if it has Baybayin that might be causing issues
                                    logger.info(f"Trying fallback processing for entry with word '{entry.get('word')}'")
                                    result = process_entry_without_baybayin(cur, entry)
                                    if result:
                                        stats["fallback_entries"] += 1
                                    break
                        
                        if result:
                            stats["processed_entries"] += 1
                            cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                            
                            # Commit this entry's transaction
                            try:
                                conn.commit()
                                if stats["processed_entries"] % 100 == 0:
                                    logger.info(f"Successfully processed {stats['processed_entries']} entries so far.")
                            except Exception as commit_error:
                                logger.error(f"Error committing entry {entry_count}: {str(commit_error)}")
                                conn.rollback()
                                # Adjust stats - entry wasn't successfully committed
                                stats["error_entries"] += 1
                                stats["processed_entries"] -= 1
                                if stats["fallback_entries"] > 0:
                                    stats["fallback_entries"] -= 1
                        else:
                            stats["error_entries"] += 1
                            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON on line {entry_count}")
                        stats["skipped_entries"] += 1
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                    except Exception as e:
                        logger.error(f"Error processing line {entry_count}: {str(e)}")
                        stats["error_entries"] += 1
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                    finally:
                        pbar.update(1)
        
        # Log detailed statistics
        logger.info(f"Completed processing {filename}:")
        logger.info(f"  Processed entries: {stats['processed_entries']}")
        logger.info(f"  Skipped entries: {stats['skipped_entries']}")
        logger.info(f"  Error entries: {stats['error_entries']}")
        logger.info(f"  Fallback entries: {stats['fallback_entries']}")
        logger.info(f"  Definitions added: {stats['definitions_added']}")
        logger.info(f"  Etymologies added: {stats['etymologies_added']}")
        logger.info(f"  Pronunciations added: {stats['pronunciations_added']}")
        logger.info(f"  Relations added: {stats['relations_added']}")
        logger.info(f"  Categories added: {stats['categories_added']}")
        
        return stats
    except Exception as e:
        logger.error(f"Error processing Kaikki dictionary: {str(e)}")
        try:
            conn.rollback()
        except Exception:
            pass
        raise