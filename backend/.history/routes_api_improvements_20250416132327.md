# Routes API Improvements

This document outlines necessary changes to `routes.py` to ensure all data migrated by `dictionary_manager.py` is properly exposed through the API.

## Schema Improvements

### WordSchema Issues

1. **Missing pronunciation_data Field**
   - Problem: `pronunciation_data` is stored in the `words` table but not exposed in `WordSchema`
   - Fix: Add `pronunciation_data = fields.Raw(allow_none=True)` to `WordSchema`

2. **JSONB Field Handling**
   - Problem: Fields like `source_info`, `word_metadata`, `idioms`, etc. use `fields.Raw` which may not handle JSONB properly
   - Fix: Ensure `fields.Raw` correctly serializes/deserializes JSONB data or switch to a custom field type

3. **Complete `is_root` Implementation**
   - Problem: The schema has `is_root` but there's no corresponding DB column, it's a computed property
   - Fix: Ensure this is properly calculated in the `_fetch_word_details` function based on `root_word_id`

### Related Schema Issues

1. **Definition Schema Metadata Naming**
   - Problem: In `DefinitionSchema`, field is named `definition_metadata` but DB column is `metadata`
   - Fix: Align naming or add proper mapping logic

2. **Pronunciation Schema Metadata Naming**
   - Problem: In `PronunciationType`, field is named `pronunciation_metadata` but related column may be `metadata`
   - Fix: Align naming or add mapping logic

## API Endpoints and Data Fetching

1. **Incomplete Word Details Fetching**
   - Problem: In `_fetch_word_details`, many fields are properly selected but not all are loaded
   - Fix: Update SQL query to include ALL database columns from the words table

2. **Definition Relations Conditionally Loaded**
   - Problem: `include_definition_relations` defaults to `False`, meaning relation data is skipped by default
   - Fix: Change default to `True` to always load this data

3. **Baybayin Handling Improvements**
   - Problem: Baybayin validation handling may be stricter than the DB schema permits after migration
   - Fix: Ensure validation matches VALID_BAYBAYIN_REGEX pattern from dictionary_manager.py

4. **Missing Relationship Loading**
   - Problem: Some relations from the migration process may not be included by default
   - Fix: Update defaults for all include_* parameters to `True` in `_fetch_word_details` function

## SQL Query Updates

1. **Update Word Detail Query**
   ```python
   sql_word = """
   SELECT id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form,
          romanized_form, root_word_id, preferred_spelling, tags, source_info, 
          word_metadata, data_hash, search_text, idioms, pronunciation_data,
          badlit_form, hyphenation, is_proper_noun, is_abbreviation, is_initialism,
          created_at, updated_at
   FROM words
   WHERE id = :id
   """
   ```

2. **Add Missing Category Fields**
   ```python
   sql_categories = """
   SELECT id, definition_id, category_name, category_kind, parents, category_metadata
   FROM definition_categories
   WHERE definition_id = ANY(:ids)
   """
   ```

## Cache Handling

1. **Cache Consistency**
   - Problem: If schema changes are made, cached objects may be out of date
   - Fix: Update cache versioning or invalidate cache on schema changes

## Parameter Defaults

1. **Standardize Parameter Defaults**
   ```python
   @bp.route("/words/<path:word>", methods=["GET"])
   def get_word(word: str):
       # Add parameter for complete data
       complete = request.args.get('complete', 'false').lower() == 'true'
       
       # If complete is requested, force all includes to True
       if complete:
           # Force all includes to be True
           include_params = {
               'include_definitions': True,
               'include_pronunciations': True,
               'include_etymologies': True,
               'include_relations': True,
               'include_forms': True,
               'include_templates': True,
               'include_credits': True,
               'include_affixations': True,
               'include_root': True,
               'include_derived': True,
               'include_definition_relations': True
           }
       else:
           # Use requested parameters or defaults
           include_params = {
               'include_definitions': request.args.get('include_definitions', 'true').lower() == 'true',
               'include_pronunciations': request.args.get('include_pronunciations', 'true').lower() == 'true',
               'include_etymologies': request.args.get('include_etymologies', 'true').lower() == 'true',
               'include_relations': request.args.get('include_relations', 'true').lower() == 'true',
               'include_forms': request.args.get('include_forms', 'true').lower() == 'true',
               'include_templates': request.args.get('include_templates', 'true').lower() == 'true',
               'include_credits': request.args.get('include_credits', 'true').lower() == 'true',
               'include_affixations': request.args.get('include_affixations', 'true').lower() == 'true',
               'include_root': request.args.get('include_root', 'true').lower() == 'true',
               'include_derived': request.args.get('include_derived', 'true').lower() == 'true',
               'include_definition_relations': request.args.get('include_definition_relations', 'true').lower() == 'true'
           }
   ```

## Summary of Changes

By implementing these changes, the API will correctly expose all data that's migrated and stored in the database by `dictionary_manager.py`, ensuring a consistent experience between the database content and the API responses. 