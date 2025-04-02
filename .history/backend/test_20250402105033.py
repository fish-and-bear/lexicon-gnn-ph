
# Add these imports near the top if not already present
import json
import os
import functools # Ensure functools is imported for decorators

# Define constants for reusable values
DEFAULT_LANGUAGE_CODE = "tl"
SOURCE_INFO_FILES_KEY = "files"

# --- NEW HELPER FUNCTION for Word Source Info ---
def update_word_source_info(current_source_info: Optional[Union[str, dict]], new_source_identifier: Optional[str]) -> str:
    """
    Updates the source_info JSON for a word entry, adding a new source identifier
    to a list under the 'files' key if it's not already present.

    Args:
        current_source_info: The current source_info from the words table (can be JSON string or parsed dict, or None).
        new_source_identifier: The identifier (e.g., filename) to add. Can be None or empty.

    Returns:
        A JSON string representation of the updated source_info. Returns '{}' if input is invalid or empty.
    """
    # If no new source is provided, return the existing info (dumped if needed) or an empty JSON string
    if not new_source_identifier:
        if isinstance(current_source_info, dict):
            try:
                return json.dumps(current_source_info)
            except TypeError:
                return '{}'
        elif isinstance(current_source_info, str):
            # Validate existing JSON string before returning
            try:
                json.loads(current_source_info)
                return current_source_info
            except (json.JSONDecodeError, TypeError):
                return '{}'
        else:
            return '{}' # Return empty for None or other types

    source_info_dict = {}
    # Attempt to load current source info into a dictionary
    if isinstance(current_source_info, dict):
        source_info_dict = current_source_info # Already a dict
    elif isinstance(current_source_info, str):
        try:
            loaded_info = json.loads(current_source_info)
            if isinstance(loaded_info, dict):
                source_info_dict = loaded_info
            else:
                 logger.warning(f"Loaded source_info is not a dict ('{current_source_info}'). Resetting.")
                 source_info_dict = {} # Ensure it's a dictionary
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Invalid existing source_info JSON ('{current_source_info}'): {e}. Resetting.")
            source_info_dict = {} # Start fresh if invalid JSON
    else:
        # Handles None or other unexpected types
        if current_source_info is not None:
             logger.warning(f"Unexpected type for current_source_info: {type(current_source_info)}. Resetting.")
        source_info_dict = {}

    # Ensure 'files' key exists and is a list
    files_list = source_info_dict.get(SOURCE_INFO_FILES_KEY)
    if not isinstance(files_list, list):
        source_info_dict[SOURCE_INFO_FILES_KEY] = []
        files_list = source_info_dict[SOURCE_INFO_FILES_KEY]


    # Add new source identifier if not already present
    if new_source_identifier not in files_list:
        files_list.append(new_source_identifier)
        files_list.sort() # Keep it sorted for consistency

    # Ensure other potential top-level keys are preserved if they exist
    # Example: You might store other metadata here later
    # source_info_dict['last_updated_by'] = new_source_identifier # Example

    try:
        # Dump the potentially modified dictionary back to a JSON string
        return json.dumps(source_info_dict)
    except TypeError as e:
        logger.error(f"Failed to dump updated source_info to JSON: {source_info_dict}. Error: {e}")
        # Fallback to empty JSON if dumping fails
        return '{}'


# --- REFACTOR CORE FUNCTIONS ---

# NOTE: Assuming `with_transaction` decorator is defined elsewhere
#       and DatabaseError, logger, normalize_lemma, get_standardized_pos_id,
#       get_uncategorized_pos_id, RelationshipType are available.
#       Also assuming psycopg2 is used for DB interaction.

@with_transaction(commit=True)
def get_or_create_word_id(cur, lemma: str, language_code: str = DEFAULT_LANGUAGE_CODE,
                          source_identifier: Optional[str] = None, # Optional, but recommended
                          check_exists: bool = False, **kwargs) -> int:
    """
    Get the ID of a word, creating it if necessary.
    Updates the word's source_info JSONB field with the provided identifier.

    Args:
        cur: Database cursor.
        lemma: The word lemma.
        language_code: The language code (default 'tl').
        source_identifier: Identifier for the data source (e.g., filename). Recommended.
        check_exists: If True, check existence before attempting insert (less efficient).
        **kwargs: Additional word attributes (e.g., has_baybayin, baybayin_form, root_word_id).

    Returns:
        The word ID.

    Raises:
        ValueError: If lemma is empty.
        DatabaseError: If the operation fails.
    """
    if not lemma:
        raise ValueError("Lemma cannot be empty")

    normalized = normalize_lemma(lemma)
    word_id = None

    # Prepare optional fields from kwargs safely
    has_baybayin = kwargs.get('has_baybayin') # Keep as None if not provided
    baybayin_form = kwargs.get('baybayin_form')
    root_word_id = kwargs.get('root_word_id')
    tags = kwargs.get('tags')
    romanized_form = kwargs.get('romanized_form')
    preferred_spelling = kwargs.get('preferred_spelling')
    idioms_json = kwargs.get('idioms', '[]') # Default to empty JSON array string
    pronunciation_data_json = kwargs.get('pronunciation_data') # Assume already JSON string or None
    word_metadata_json = kwargs.get('word_metadata', '{}') # Default to empty JSON object string

    # Clean up Baybayin if inconsistent
    if has_baybayin is False:
        baybayin_form = None # Ensure form is None if explicitly false
    elif has_baybayin is True and not baybayin_form:
        logger.warning(f"Word '{lemma}' ({language_code}, source: {source_identifier}) marked as has_baybayin but no form provided. Setting has_baybayin to False.")
        has_baybayin = False # Correct the inconsistency

    try:
         # Check if word exists and get its current ID and source_info
         cur.execute("""
             SELECT id, source_info FROM words
             WHERE normalized_lemma = %s AND language_code = %s
         """, (normalized, language_code))
         existing_word = cur.fetchone()

         if existing_word:
             word_id = existing_word[0]
             current_source_info = existing_word[1] # Fetched as dict/list/etc. from JSONB

             # Update existing word's source info using the helper
             updated_source_json = update_word_source_info(current_source_info, source_identifier)

             # Parse the updated JSON string back to compare with the original dict/list
             updated_source_dict = {}
             try:
                 updated_source_dict = json.loads(updated_source_json)
             except (json.JSONDecodeError, TypeError):
                 logger.error(f"Failed to parse updated source JSON for word ID {word_id}: '{updated_source_json}'")
                 # Decide how to handle - maybe skip update? For now, assume it might need update if parsing fails.
                 pass # Let the update proceed cautiously

             # Only update if the source_info content has actually changed
             if updated_source_dict != (current_source_info or {}):
                 cur.execute("""
                     UPDATE words SET source_info = %s, updated_at = CURRENT_TIMESTAMP
                     WHERE id = %s
                 """, (updated_source_json, word_id))
                 logger.debug(f"Word '{lemma}' ({language_code}) found (ID: {word_id}). Updated source_info from source '{source_identifier}'.")
             else:
                 logger.debug(f"Word '{lemma}' ({language_code}) found (ID: {word_id}). Source_info already includes '{source_identifier}' or no update needed.")

             # Add logic here if other fields (e.g., baybayin_form, tags) should be updated
             # based on the new source, potentially requiring merge strategies.
             # For now, we prioritize getting/creating the ID and updating source_info.

         else:
             # Word doesn't exist, insert it
             logger.debug(f"Word '{lemma}' ({language_code}) not found. Creating new entry from source '{source_identifier}'.")
             initial_source_json = update_word_source_info(None, source_identifier)

             cur.execute("""
                 INSERT INTO words (
                     lemma, normalized_lemma, language_code,
                     has_baybayin, baybayin_form, romanized_form, root_word_id,
                     preferred_spelling, tags, source_info,
                     idioms, pronunciation_data, word_metadata
                     -- Add other fields from kwargs as necessary in schema
                 )
                 VALUES (
                     %(lemma)s, %(normalized)s, %(language_code)s,
                     %(has_baybayin)s, %(baybayin_form)s, %(romanized_form)s, %(root_word_id)s,
                     %(preferred_spelling)s, %(tags)s, %(source_info)s,
                     %(idioms)s, %(pronunciation_data)s, %(word_metadata)s
                 )
                 RETURNING id
             """, {
                 'lemma': lemma, 'normalized': normalized, 'language_code': language_code,
                 'has_baybayin': has_baybayin, 'baybayin_form': baybayin_form, 'romanized_form': romanized_form,
                 'root_word_id': root_word_id, 'preferred_spelling': preferred_spelling, 'tags': tags,
                 'source_info': initial_source_json, # Directly use the JSON string
                 'idioms': idioms_json, # Assumed to be valid JSON string or None
                 'pronunciation_data': pronunciation_data_json, # Assumed to be valid JSON string or None
                 'word_metadata': word_metadata_json # Assumed to be valid JSON string or None
             })
             word_id = cur.fetchone()[0]
             logger.info(f"Word '{lemma}' ({language_code}) created (ID: {word_id}) from source '{source_identifier}'.")

    except psycopg2.Error as e:
         logger.error(f"Database error in get_or_create_word_id for '{lemma}' ({language_code}) from source '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
         raise DatabaseError(f"Failed to get/create word ID for '{lemma}' from source '{source_identifier}': {e}") from e
    except Exception as e:
         logger.error(f"Unexpected error in get_or_create_word_id for '{lemma}' ({language_code}) from source '{source_identifier}': {e}", exc_info=True)
         raise # Reraise unexpected errors

    if word_id is None:
        # Should not happen if exceptions are handled, but as a safeguard
        raise DatabaseError(f"Failed to obtain word ID for '{lemma}' ({language_code}) from source '{source_identifier}' after operations.")

    return word_id


@with_transaction(commit=True)
def insert_definition(cur, word_id: int, definition_text: str,
                      source_identifier: str, # MANDATORY
                      part_of_speech: Optional[str] = None, # Changed default to None
                      examples: Optional[str] = None, usage_notes: Optional[str] = None,
                      tags: Optional[str] = None) -> Optional[int]:
    """
    Insert definition data for a word. Uses ON CONFLICT to update existing definitions
    based on (word_id, definition_text, standardized_pos_id), applying a 'last write wins'
    strategy for the 'sources' field for the specific definition record.

    Args:
        cur: Database cursor.
        word_id: The ID of the word this definition belongs to.
        definition_text: The text of the definition.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.
        part_of_speech: Original part of speech string (can be None or empty).
        examples: Usage examples (string).
        usage_notes: Notes on usage (string).
        tags: Tags associated with the definition (string).

    Returns:
        The ID of the inserted/updated definition, or None if failed.
    """
    definition_text = definition_text.strip() if isinstance(definition_text, str) else None
    part_of_speech = part_of_speech.strip() if isinstance(part_of_speech, str) else None

    if not definition_text:
        logger.warning(f"Skipping definition insert for word ID {word_id} from source '{source_identifier}': Missing definition text.")
        return None
    if not source_identifier:
        # This check might be redundant if called correctly, but good safeguard
        logger.error(f"CRITICAL: Skipping definition insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    try:
        # Get standardized POS ID, handle None/empty POS string
        pos_id = None
        if part_of_speech:
             pos_id = get_standardized_pos_id(cur, part_of_speech) # Assumes this handles errors/returns ID
        if pos_id is None:
             pos_id = get_uncategorized_pos_id(cur) # Assumes this returns the ID for 'uncategorized'

        # Prepare data, ensuring None is passed for empty optional fields
        params = {
            'word_id': word_id,
            'def_text': definition_text,
            'orig_pos': part_of_speech, # Store original POS as provided
            'pos_id': pos_id,
            'examples': examples.strip() if isinstance(examples, str) else None,
            'usage_notes': usage_notes.strip() if isinstance(usage_notes, str) else None,
            'tags': tags.strip() if isinstance(tags, str) else None,
            'sources': source_identifier # Use mandatory source_identifier directly
        }

        # Insert or update definition
        # Conflict target includes pos_id to differentiate definitions that are identical
        # except for part of speech.
        cur.execute("""
            INSERT INTO definitions (
                word_id, definition_text, original_pos, standardized_pos_id,
                examples, usage_notes, tags, sources
            )
            VALUES (%(word_id)s, %(def_text)s, %(orig_pos)s, %(pos_id)s,
                    %(examples)s, %(usage_notes)s, %(tags)s, %(sources)s)
            ON CONFLICT (word_id, definition_text, standardized_pos_id)
            DO UPDATE SET
                -- Update optional fields only if the new value is not NULL
                original_pos = COALESCE(EXCLUDED.original_pos, definitions.original_pos),
                examples = COALESCE(EXCLUDED.examples, definitions.examples),
                usage_notes = COALESCE(EXCLUDED.usage_notes, definitions.usage_notes),
                tags = COALESCE(EXCLUDED.tags, definitions.tags),
                -- Overwrite sources: Last write wins for this definition record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, params)
        definition_id = cur.fetchone()[0]
        # Log success with more info
        log_pos = f"POS: '{part_of_speech}' (ID: {pos_id})" if part_of_speech else f"POS ID: {pos_id}"
        logger.debug(f"Inserted/Updated definition (ID: {definition_id}) for word ID {word_id} [{log_pos}] from source '{source_identifier}'. Text: '{definition_text[:50]}...'")
        return definition_id

    except psycopg2.Error as e:
        logger.error(f"Database error inserting definition for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
        return None # Allow process to continue if possible
    except Exception as e:
        logger.error(f"Unexpected error inserting definition for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        return None


@with_transaction(commit=True)
def insert_relation(
    cur,
    from_word_id: int,
    to_word_id: int,
    relation_type: Union[RelationshipType, str],
    source_identifier: str, # MANDATORY
    metadata: Optional[Dict] = None
) -> Optional[int]:
    """
    Insert a relationship between two words. Uses ON CONFLICT to update existing relations
    based on (from_word_id, to_word_id, relation_type), applying a 'last write wins'
    strategy for the 'sources' field for the specific relation record.

    Args:
        cur: Database cursor.
        from_word_id: ID of the source word.
        to_word_id: ID of the target word.
        relation_type: The type of relationship (RelationshipType enum or string).
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.
        metadata: Optional JSON metadata for the relationship (will be stored as JSONB).

    Returns:
        The ID of the inserted/updated relation, or None if failed.
    """
    if from_word_id == to_word_id:
        logger.warning(f"Skipping self-relation for word ID {from_word_id}, type '{relation_type}', source '{source_identifier}'.")
        return None
    if not source_identifier:
         logger.error(f"CRITICAL: Skipping relation insert from {from_word_id} to {to_word_id}: Missing MANDATORY source identifier.")
         return None

    rel_type_enum = None
    rel_type_str = None
    try:
        # Standardize relation type
        if isinstance(relation_type, RelationshipType):
            rel_type_enum = relation_type
            rel_type_str = rel_type_enum.rel_value
        elif isinstance(relation_type, str):
            relation_type_cleaned = relation_type.lower().strip()
            if not relation_type_cleaned:
                 logger.warning(f"Skipping relation insert from {from_word_id} to {to_word_id} (source '{source_identifier}'): Empty relation type string provided.")
                 return None
            try:
                # Attempt to map string to enum value
                rel_type_enum = RelationshipType.from_string(relation_type_cleaned)
                rel_type_str = rel_type_enum.rel_value
            except ValueError:
                # If not a standard enum value, use the cleaned string directly
                rel_type_str = relation_type_cleaned
                logger.debug(f"Using non-standard relation type string '{rel_type_str}' from source '{source_identifier}'.")
        else:
             logger.warning(f"Skipping relation insert from {from_word_id} to {to_word_id} (source '{source_identifier}'): Invalid relation_type type '{type(relation_type)}'.")
             return None

        # Dump metadata safely to JSON string for DB insertion (assuming metadata column is JSONB)
        metadata_json = None
        if metadata is not None: # Check explicitly for None, allow empty dict {}
             if isinstance(metadata, dict):
                  try:
                       metadata_json = json.dumps(metadata)
                  except TypeError as e:
                       logger.warning(f"Could not serialize metadata for relation {from_word_id}->{to_word_id} (source '{source_identifier}'): {e}. Metadata: {metadata}")
                       metadata_json = '{}' # Use empty JSON object string as fallback
             else:
                  logger.warning(f"Metadata provided for relation {from_word_id}->{to_word_id} (source '{source_identifier}') is not a dict: {type(metadata)}. Storing as null.")
                  metadata_json = None # Store null if not a dict

        cur.execute("""
            INSERT INTO relations (from_word_id, to_word_id, relation_type, sources, metadata)
            VALUES (%(from_id)s, %(to_id)s, %(rel_type)s, %(sources)s, %(metadata)s::jsonb) -- Cast metadata to JSONB
            ON CONFLICT (from_word_id, to_word_id, relation_type)
            DO UPDATE SET
                -- Overwrite sources: Last write wins for this relation record
                sources = EXCLUDED.sources,
                -- Update metadata only if the new value is not NULL
                metadata = COALESCE(EXCLUDED.metadata, relations.metadata),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, {
            'from_id': from_word_id, 'to_id': to_word_id, 'rel_type': rel_type_str,
            'sources': source_identifier,
            'metadata': metadata_json # Pass the JSON string (or None)
        })
        relation_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated relation (ID: {relation_id}) {from_word_id}->{to_word_id} ('{rel_type_str}') from source '{source_identifier}'.")
        return relation_id

    except psycopg2.IntegrityError as e:
         # Likely due to non-existent from_word_id or to_word_id (FK constraint violation)
         logger.error(f"Integrity error inserting relation {from_word_id}->{to_word_id} ('{relation_type}') from '{source_identifier}'. Word ID might not exist. Error: {e.pgcode} {e.pgerror}")
         return None
    except psycopg2.Error as e:
        logger.error(f"Database error inserting relation {from_word_id}->{to_word_id} ('{relation_type}') from '{source_identifier}': {e.pgcode} {e.pgerror}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting relation {from_word_id}->{to_word_id} ('{relation_type}') from '{source_identifier}': {e}", exc_info=True)
        return None


@with_transaction(commit=True)
def insert_etymology(
    cur,
    word_id: int,
    etymology_text: str,
    source_identifier: str, # MANDATORY
    normalized_components: Optional[str] = None,
    etymology_structure: Optional[str] = None, # Consider if this should be JSON
    language_codes: Optional[str] = None, # Comma-separated string
) -> Optional[int]:
    """
    Insert etymology data for a word. Uses ON CONFLICT to update existing etymologies
    based on (word_id, etymology_text), applying a 'last write wins' strategy for the
    'sources' field for the specific etymology record.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        etymology_text: The etymological explanation.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.
        normalized_components: Normalized components string.
        etymology_structure: Structural information (string).
        language_codes: Comma-separated language codes involved (string).

    Returns:
        The ID of the inserted/updated etymology record, or None if failed.
    """
    etymology_text = etymology_text.strip() if isinstance(etymology_text, str) else None
    if not etymology_text:
        logger.warning(f"Skipping etymology insert for word ID {word_id} from source '{source_identifier}': Missing etymology text.")
        return None
    if not source_identifier:
        logger.error(f"CRITICAL: Skipping etymology insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    try:
        # Prepare data, ensuring None is passed for empty optional fields
        params = {
            'word_id': word_id,
            'etym_text': etymology_text,
            'norm_comp': normalized_components.strip() if isinstance(normalized_components, str) else None,
            'etym_struct': etymology_structure.strip() if isinstance(etymology_structure, str) else None,
            'lang_codes': language_codes.strip() if isinstance(language_codes, str) else None,
            'sources': source_identifier # Use mandatory source_identifier directly
        }

        cur.execute("""
            INSERT INTO etymologies (
                word_id, etymology_text, normalized_components,
                etymology_structure, language_codes, sources
            )
            VALUES (%(word_id)s, %(etym_text)s, %(norm_comp)s,
                    %(etym_struct)s, %(lang_codes)s, %(sources)s)
            ON CONFLICT (word_id, etymology_text) -- Conflict on word and exact text match
            DO UPDATE SET
                -- Update optional fields only if the new value is not NULL
                normalized_components = COALESCE(EXCLUDED.normalized_components, etymologies.normalized_components),
                etymology_structure = COALESCE(EXCLUDED.etymology_structure, etymologies.etymology_structure),
                language_codes = COALESCE(EXCLUDED.language_codes, etymologies.language_codes),
                -- Overwrite sources: Last write wins for this etymology record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, params)
        etymology_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated etymology (ID: {etymology_id}) for word ID {word_id} from source '{source_identifier}'. Text: '{etymology_text[:50]}...'")
        return etymology_id

    except psycopg2.Error as e:
        logger.error(f"Database error inserting etymology for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting etymology for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        return None


@with_transaction(commit=True)
def insert_pronunciation(cur, word_id: int, pronunciation_data: Union[str, Dict],
                         source_identifier: str) -> Optional[int]:
    """
    Insert pronunciation data for a word. Handles string or dictionary input.
    Uses ON CONFLICT to update existing pronunciations based on (word_id, type, value),
    applying a 'last write wins' strategy for the 'sources' field.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        pronunciation_data: Pronunciation string (assumed IPA) or dictionary
                           (e.g., {'type': 'ipa', 'value': '...', 'tags': [...]}).
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted/updated pronunciation record, or None if failed.
    """
    if not source_identifier:
        logger.error(f"CRITICAL: Skipping pronunciation insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    pron_type = 'ipa' # Default type
    value = None
    tags_list = []
    metadata = {}

    try:
        # Parse input data
        if isinstance(pronunciation_data, dict):
            # Prioritize keys from the dict, but allow source_identifier argument to override 'sources' key if present
            pron_type = pronunciation_data.get('type', 'ipa') or 'ipa' # Default to 'ipa' if empty
            value = pronunciation_data.get('value', '').strip() if isinstance(pronunciation_data.get('value'), str) else None
            tags_input = pronunciation_data.get('tags')
            if isinstance(tags_input, list):
                 tags_list = tags_input
            elif isinstance(tags_input, str):
                 # Simple split if tags are a comma-separated string? Adapt as needed.
                 tags_list = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
                 logger.debug(f"Parsed tags string '{tags_input}' into list for word ID {word_id}.")
            # else: ignore invalid tags format

            metadata_input = pronunciation_data.get('metadata')
            if isinstance(metadata_input, dict):
                metadata = metadata_input
            # else: ignore invalid metadata format

        elif isinstance(pronunciation_data, str):
            value = pronunciation_data.strip()
            # Assumed type is 'ipa' and no tags/metadata provided via string input
        else:
             logger.warning(f"Invalid pronunciation_data type for word ID {word_id} (source '{source_identifier}'): {type(pronunciation_data)}. Skipping.")
             return None

        if not value:
            logger.warning(f"Empty pronunciation value for word ID {word_id} (source '{source_identifier}'). Skipping.")
            return None

        # Safely dump JSON fields (tags and metadata) for DB insertion (assuming JSONB columns)
        tags_json = None
        try:
            tags_json = json.dumps(tags_list) # tags_list will be [] if not provided or invalid format
        except TypeError as e:
            logger.warning(f"Could not serialize tags for pronunciation (word ID {word_id}, source '{source_identifier}'): {e}. Tags: {tags_list}")
            tags_json = '[]' # Fallback to empty JSON array string

        metadata_json = None
        try:
            metadata_json = json.dumps(metadata) # metadata will be {} if not provided or invalid format
        except TypeError as e:
            logger.warning(f"Could not serialize metadata for pronunciation (word ID {word_id}, source '{source_identifier}'): {e}. Metadata: {metadata}")
            metadata_json = '{}' # Fallback to empty JSON object string

        # Prepare parameters for query
        params = {
            'word_id': word_id,
            'type': pron_type,
            'value': value,
            'tags': tags_json,
            'metadata': metadata_json,
            'sources': source_identifier # Use mandatory source_identifier directly
        }

        # Insert or update pronunciation
        cur.execute("""
            INSERT INTO pronunciations (word_id, type, value, tags, metadata, sources)
            VALUES (%(word_id)s, %(type)s, %(value)s, %(tags)s::jsonb, %(metadata)s::jsonb, %(sources)s) -- Cast JSONs
            ON CONFLICT (word_id, type, value) -- Conflict on word, type, and exact value
            DO UPDATE SET
                -- Update tags/metadata only if new value is not NULL (or empty JSON?) - COALESCE prefers new non-null
                tags = COALESCE(EXCLUDED.tags, pronunciations.tags),
                metadata = COALESCE(EXCLUDED.metadata, pronunciations.metadata),
                -- Overwrite sources: Last write wins for this pronunciation record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, params)
        pron_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated pronunciation (ID: {pron_id}, Type: {pron_type}) for word ID {word_id} from source '{source_identifier}'. Value: '{value}'")
        return pron_id

    except psycopg2.Error as e:
        logger.error(f"Database error inserting pronunciation for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting pronunciation for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        return None


@with_transaction(commit=True)
def insert_credit(cur, word_id: int, credit_data: Union[str, Dict],
                  source_identifier: str) -> Optional[int]:
    """
    Insert credit data for a word. Handles string or dictionary input.
    Uses ON CONFLICT to update existing credits based on (word_id, credit),
    applying a 'last write wins' strategy for the 'sources' field.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        credit_data: Credit string or dictionary (e.g., {'text': 'Source Name'}).
                     If dict, the 'text' key is used.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted/updated credit record, or None if failed.
    """
    if not source_identifier:
        logger.error(f"CRITICAL: Skipping credit insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    credit_text = None
    try:
        # Extract credit text
        if isinstance(credit_data, dict):
            # Prioritize 'text' key if dict is provided
            credit_text = credit_data.get('text', '').strip() if isinstance(credit_data.get('text'), str) else None
        elif isinstance(credit_data, str):
            credit_text = credit_data.strip()
        else:
            logger.warning(f"Invalid credit_data type for word ID {word_id} (source '{source_identifier}'): {type(credit_data)}. Skipping.")
            return None

        if not credit_text:
            logger.warning(f"Empty credit text for word ID {word_id} (source '{source_identifier}'). Skipping.")
            return None

        # Prepare parameters
        params = {
            'word_id': word_id,
            'credit': credit_text,
            'sources': source_identifier # Use mandatory source_identifier directly
        }

        # Insert or update credit
        cur.execute("""
            INSERT INTO credits (word_id, credit, sources)
            VALUES (%(word_id)s, %(credit)s, %(sources)s)
            ON CONFLICT (word_id, credit) -- Conflict on word and exact credit text
            DO UPDATE SET
                -- Overwrite sources: Last write wins for this credit record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, params)
        credit_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated credit (ID: {credit_id}) for word ID {word_id} from source '{source_identifier}'. Credit: '{credit_text}'")
        return credit_id

    except psycopg2.Error as e:
        logger.error(f"Database error inserting credit for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting credit for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        return None


@with_transaction(commit=True)
def insert_affixation(
    cur,
    root_id: int,
    affixed_id: int,
    affix_type: str,
    source_identifier: str # MANDATORY
) -> Optional[int]:
    """
    Insert an affixation relationship (e.g., root -> derived word).
    Uses ON CONFLICT to update existing affixations based on (root_word_id, affixed_word_id, affix_type),
    applying a 'last write wins' strategy for the 'sources' field.

    Args:
        cur: Database cursor.
        root_id: ID of the root word.
        affixed_id: ID of the affixed word.
        affix_type: Type of affixation (e.g., 'prefix', 'suffix', 'infix', 'circumfix').
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted/updated affixation record, or None if failed.
    """
    affix_type = affix_type.strip().lower() if isinstance(affix_type, str) else None # Normalize type
    if not affix_type:
         logger.warning(f"Skipping affixation insert for root {root_id}, affixed {affixed_id} (source '{source_identifier}'): Missing affix type.")
         return None
    if not source_identifier:
         logger.error(f"CRITICAL: Skipping affixation insert for root {root_id}, affixed {affixed_id}: Missing MANDATORY source identifier.")
         return None
    if root_id == affixed_id:
         logger.warning(f"Skipping self-affixation for word ID {root_id}, type '{affix_type}', source '{source_identifier}'.")
         return None

    try:
        # Prepare parameters
        params = {
            'root_id': root_id,
            'affixed_id': affixed_id,
            'affix_type': affix_type,
            'sources': source_identifier # Use mandatory source_identifier directly
        }

        cur.execute("""
            INSERT INTO affixations (root_word_id, affixed_word_id, affix_type, sources)
            VALUES (%(root_id)s, %(affixed_id)s, %(affix_type)s, %(sources)s)
            ON CONFLICT (root_word_id, affixed_word_id, affix_type) -- Conflict on exact triplet
            DO UPDATE SET
                -- Overwrite sources: Last write wins for this affixation record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, params)
        affixation_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated affixation (ID: {affixation_id}) {root_id}(root) -> {affixed_id}(affixed) [{affix_type}] from source '{source_identifier}'.")
        return affixation_id

    except psycopg2.IntegrityError as e:
         # Likely due to non-existent root_id or affixed_id (FK constraint violation)
         logger.error(f"Integrity error inserting affixation {root_id}->{affixed_id} ({affix_type}) from '{source_identifier}'. Word ID might not exist. Error: {e.pgcode} {e.pgerror}")
         return None
    except psycopg2.Error as e:
        logger.error(f"Database error inserting affixation {root_id}->{affixed_id} ({affix_type}) from '{source_identifier}': {e.pgcode} {e.pgerror}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting affixation {root_id}->{affixed_id} ({affix_type}) from '{source_identifier}': {e}", exc_info=True)
        return None

# --- END REFACTORED CORE FUNCTIONS ---
