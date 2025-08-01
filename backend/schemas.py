"""
Marshmallow schemas for API serialization with enhanced validation and performance.
"""

from marshmallow import Schema, fields, pre_dump, post_dump, validates, ValidationError, validate, EXCLUDE, validates_schema, post_load, pre_load
from marshmallow.validate import Length, Range, OneOf
import datetime
from typing import Dict, Any, List, Optional, Union
import json
import logging

class MetadataField(fields.Dict):
    """Custom field for handling JSONB metadata fields."""
    def _serialize(self, value, attr, obj, **kwargs):
        processed_value = {}
        logger = logging.getLogger(__name__) # Get logger instance

        if value is None:
            return {}

        try:
            # Check if it's SQLAlchemy MetaData object - handle this first
            if hasattr(value, '__class__') and value.__class__.__name__ == 'MetaData':
                logger.debug(f"Detected SQLAlchemy MetaData object for field '{attr}', returning empty dict")
                return {}
                
            # Priority 1: If it's already a dict, try creating a new dict from it
            # This validates that it behaves like a standard dict for the constructor.
            if isinstance(value, dict):
                processed_value = dict(value)
            # Priority 2: If it's a non-empty string, try loading as JSON
            elif isinstance(value, str) and value.strip():
                try:
                    loaded_json = json.loads(value)
                    # Ensure the loaded JSON is actually a dictionary
                    if isinstance(loaded_json, dict):
                        processed_value = loaded_json
                    else:
                        logger.warning(f"JSON loaded from string for MetadataField '{attr}' is not a dict: {type(loaded_json)}. Value: {value}. Returning empty dict.")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON string for MetadataField '{attr}': {value}. Returning empty dict.")
            else:
                # If it's neither dict nor valid JSON string, log and default to {}
                logger.warning(f"Invalid type/value for MetadataField '{attr}' on object {obj}: Expected dict or JSON string, got {type(value)}. Value: {repr(value)}. Returning empty dict.")

        except (ValueError, TypeError) as e:
            # Catch errors during dict(value) or json.loads if 'value' is problematic
            logger.warning(f"Error processing MetadataField '{attr}' on object {obj}. Type: {type(value)}, Value: {repr(value)}, Error: {e}. Returning empty dict.")
            processed_value = {} # Ensure fallback to empty dict on error

        # Final check ensure processed_value is a dict before passing to super
        if not isinstance(processed_value, dict):
             processed_value = {}

        # Call the parent serializer with the processed, guaranteed-to-be-dict value
        return super()._serialize(processed_value, attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        processed_value = {}
        logger = logging.getLogger(__name__)

        if value is None:
            return {}
        
        try:
            # Check if it's SQLAlchemy MetaData object
            if hasattr(value, '__class__') and value.__class__.__name__ == 'MetaData':
                logger.debug(f"Detected SQLAlchemy MetaData object for field '{attr}', returning empty dict")
                return {}
                
            if isinstance(value, dict):
                processed_value = dict(value) # Ensure it's a standard dict
            elif isinstance(value, str) and value.strip():
                try:
                    loaded_json = json.loads(value)
                    if isinstance(loaded_json, dict):
                        processed_value = loaded_json
                    else:
                         logger.warning(f"Deserialized JSON for MetadataField '{attr}' is not dict: {type(loaded_json)}. Value: {value}. Returning empty dict.")
                except json.JSONDecodeError:
                    raise ValidationError(f"Invalid JSON format for metadata field '{attr}'.")
            else:
                 logger.warning(f"Deserializing invalid type for MetadataField '{attr}': Expected dict or JSON string, got {type(value)}. Value: {repr(value)}. Returning empty dict.")
        
        except (ValueError, TypeError) as e:
            logger.warning(f"Error deserializing MetadataField '{attr}'. Type: {type(value)}, Value: {repr(value)}, Error: {e}. Returning empty dict.")
            processed_value = {}
            
        # Ensure deserialized value is a dict before passing to super
        if not isinstance(processed_value, dict):
            processed_value = {}
            
        return super()._deserialize(processed_value, attr, data, **kwargs)

class DefinitionLinkSchema(Schema):
    """
    Schema for definition links.
    Reflects storage in link_metadata and provides derived fields.
    """
    id = fields.Integer(dump_only=True)
    definition_id = fields.Integer(required=True)
    link_text = fields.String(required=True) # The visible text of the link
    link_metadata = MetadataField(dump_default={}) # Stores target_url, is_external, sources etc.
    tags = fields.String(allow_none=True) # Added from DB schema (TEXT)
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a string.
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    # Add derived fields during serialization for convenience/compatibility
    target_url = fields.String(dump_only=True)
    is_external = fields.Boolean(dump_only=True)
    is_wikipedia = fields.Boolean(dump_only=True)

    @post_dump(pass_many=True)
    def derive_link_fields(self, data, many, **kwargs):
        """Extract common link properties from link_metadata after dumping."""
        if many:
            for item in data:
                self._add_derived_fields(item)
        else:
            self._add_derived_fields(data)
        return data

    def _add_derived_fields(self, item_data):
        """Helper to add derived fields to a single item's data."""
        metadata = item_data.get('link_metadata', {})
        item_data['target_url'] = metadata.get('target_url', item_data.get('link_text', '')) # Default target to link_text if missing
        item_data['is_external'] = metadata.get('is_external', False)
        item_data['is_wikipedia'] = metadata.get('is_wikipedia', False)
        # We could optionally extract sources here too if needed for the API response
        # item_data['sources'] = metadata.get('sources', [])

class DefinitionCategorySchema(Schema):
    """Schema for definition categories."""
    id = fields.Integer(dump_only=True)
    definition_id = fields.Integer(required=True)
    category_name = fields.String(required=True)
    category_kind = fields.String(allow_none=True) # Matches DB (TEXT)
    parents = fields.List(fields.String(), allow_none=True) # Matches DB (JSONB assumed to store list of strings)
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a string.
    category_metadata = MetadataField(dump_default={}) # Added metadata field
    # Removed tags, description, and category_metadata as they are not in the final DB schema
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class ExampleSchema(Schema):
    """Schema for structured definition examples."""
    id = fields.Integer(dump_only=True)
    definition_id = fields.Integer(dump_only=True)
    example_text = fields.String(required=True)
    translation = fields.String(allow_none=True)
    reference = fields.String(allow_none=True)
    example_type = fields.String(dump_default="example")
    metadata = MetadataField(dump_default={})
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a string.
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    @post_dump(pass_many=True)
    def extract_romanization(self, data, many, **kwargs):
        if many:
            for item in data:
                self._add_romanization(item)
        else:
            self._add_romanization(data)
        return data

    def _add_romanization(self, item_data):
        metadata = item_data.get('metadata', {})
        item_data['romanization'] = metadata.get('romanization')

    class Meta:
        # Ensure fields are serialized in a consistent order
        ordered = True

class DefinitionSchema(Schema):
    """Schema for word definitions."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(dump_only=True)
    definition_text = fields.String(required=True, validate=Length(min=1))
    original_pos = fields.String(validate=Length(min=1, max=100))
    standardized_pos_id = fields.Integer()
    standardized_pos = fields.Nested("PartOfSpeechSchema", dump_only=True)
    notes = fields.String()
    examples = fields.Nested(ExampleSchema, many=True, dump_only=True)
    usage_notes = fields.String()
    cultural_notes = fields.String()
    etymology_notes = fields.String()
    scientific_name = fields.String()
    verified = fields.Boolean(dump_default=False)
    verification_notes = fields.String()
    tags = fields.String(allow_none=True)
    # Handle missing definition_metadata column gracefully
    definition_metadata = MetadataField(dump_default={}, load_default={})
    popularity_score = fields.Float(dump_default=0.0)
    links = fields.List(fields.Nested(DefinitionLinkSchema), dump_default=[])
    categories = fields.List(fields.Nested(DefinitionCategorySchema), dump_default=[])
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a string.
    
    # Track timestamps
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    @pre_dump
    def ensure_metadata(self, data, **kwargs):
        """Ensure definition_metadata is available before serialization."""
        # If the object doesn't have definition_metadata attribute or it's None,
        # add an empty dict to prevent serialization errors
        if not hasattr(data, 'definition_metadata') or data.definition_metadata is None:
            data.definition_metadata = {}
        return data
    
    @post_dump(pass_many=True)
    def format_standardized_pos(self, data, many, **kwargs):
        """Format the standardized part of speech into a string representation."""
        if many:
            for item in data:
                if item.get('standardized_pos'):
                    pos = item['standardized_pos']
                    item['standardized_pos_code'] = pos.get('code')
                    item['standardized_pos_name'] = pos.get('name')
        else:
            if data.get('standardized_pos'):
                pos = data['standardized_pos']
                data['standardized_pos_code'] = pos.get('code')
                data['standardized_pos_name'] = pos.get('name')
        return data

class PartOfSpeechSchema(Schema):
    """Schema for parts of speech."""
    id = fields.Integer(dump_only=True)
    code = fields.String(required=True, validate=Length(min=1, max=10))
    name_en = fields.String(required=True, validate=Length(min=1, max=100))
    name_tl = fields.String(required=True, validate=Length(min=1, max=100))
    description = fields.String()

class PronunciationType(Schema):
    """Schema for pronunciations."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(dump_only=True)
    type = fields.String(required=True, validate=Length(min=1, max=50))
    value = fields.String(required=True)
    tags = MetadataField(dump_default={})  # Changed to JSONB
    pronunciation_metadata = MetadataField(dump_default={}) # Includes source information (display name).
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    # Adjust pre_dump/post_dump if needed to handle tags/metadata format conversion
    # Example: Convert list from DB tags back to list for API if needed
    @pre_dump
    def process_tags(self, data, **kwargs):
        # Handle tags conversion if stored differently in model vs API representation
        # e.g., if tags is stored as {"tags": [...]} in DB but API expects just [...]
        if hasattr(data, 'tags') and isinstance(data.tags, dict) and 'tags' in data.tags:
            # Store the list temporarily if needed for post_dump
            data._processed_tags_list = data.tags['tags']
        elif hasattr(data, 'tags') and isinstance(data.tags, list): # If model already has list
            data._processed_tags_list = data.tags
        else:
            data._processed_tags_list = []
        return data

    @post_dump
    def format_output(self, data, **kwargs):
        # Assign the processed list to the final 'tags' field in the output
        data['tags'] = getattr(data, '_processed_tags_list', [])
        if hasattr(data, '_processed_tags_list'):
            delattr(data, '_processed_tags_list') # Clean up temp attribute
        
        return data

class RelationSchema(Schema):
    """Schema for relations."""
    id = fields.Integer(dump_only=True)
    from_word_id = fields.Integer(required=True)
    to_word_id = fields.Integer(required=True)
    relation_type = fields.String(required=True)
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a comma-separated string.
    metadata = MetadataField(dump_default={}) # Added metadata field
    
    # Fields for nested data (dump_only)
    source_word = fields.Nested(lambda: WordSchema(only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form')), dump_only=True)
    target_word = fields.Nested(lambda: WordSchema(only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form')), dump_only=True)
    target_gloss = fields.String(dump_only=True) # Added based on analysis, likely from target_word

    # Timestamps
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    @post_dump(pass_many=True)
    def format_output(self, data, many, **kwargs):
        """Extract target_gloss."""
        if many:
            for item in data:
                self._format_single_item(item)
        else:
            self._format_single_item(data)
        return data

    def _format_single_item(self, item_data):
        """Helper to extract gloss for a single item."""
        # Extract target_gloss from metadata - CHECK IF METADATA EXISTS
        metadata = item_data.get('metadata', {})
        item_data['target_gloss'] = metadata.get('target_gloss', None) # Get gloss, default to None

class AffixationSchema(Schema):
    """Schema for affixations."""
    id = fields.Integer(dump_only=True)
    root_word_id = fields.Integer(required=True)
    affixed_word_id = fields.Integer(required=True)
    affix_type = fields.String(required=True)
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a comma-separated string.
    root_word = fields.Nested("WordSimpleSchema", dump_only=True)
    affixed_word = fields.Nested("WordSimpleSchema", dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class EtymologySchema(Schema):
    """Schema for etymologies."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(required=True)
    etymology_text = fields.String(required=True)
    normalized_components = fields.String()
    etymology_structure = fields.String()
    language_codes = fields.String(allow_none=True) # Changed to String to match DB TEXT
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a comma-separated string.
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class CreditSchema(Schema):
    """Schema for credits."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(required=True)
    credit = fields.String(required=True)
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a comma-separated string.
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class DefinitionRelationSchema(Schema):
    """Schema for definition relations."""
    id = fields.Integer(dump_only=True)
    definition_id = fields.Integer(required=True)
    word_id = fields.Integer(required=True)
    relation_type = fields.String(required=True)
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a comma-separated string.
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    # Relationships
    definition = fields.Nested("DefinitionSchema", dump_only=True)
    word = fields.Nested("WordSimpleSchema", dump_only=True)

class WordFormSchema(Schema):
    """Schema for word forms."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(required=True)
    form = fields.String(required=True)
    is_canonical = fields.Boolean(dump_default=False)
    is_primary = fields.Boolean(dump_default=False)
    tags = MetadataField(dump_default={})
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a string.
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class WordTemplateSchema(Schema):
    """Schema for word templates."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(required=True)
    template_name = fields.String(required=True)
    args = MetadataField(dump_default={})
    expansion = fields.String(allow_none=True)
    sources = fields.String(allow_none=True) # Stores the source display name(s), typically a string.
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class WordSimpleSchema(Schema):
    """Simple schema for word references."""
    id = fields.Integer(dump_only=True)
    lemma = fields.String(required=True)
    normalized_lemma = fields.String(dump_only=True)
    language_code = fields.String(dump_default='tl')
    has_baybayin = fields.Boolean(dump_default=False)
    baybayin_form = fields.String()
    romanized_form = fields.String(dump_only=True)
    root_word_id = fields.Integer()
    is_root = fields.Boolean(dump_only=True)
    
    @post_dump
    def add_is_root(self, data, **kwargs):
        """Add is_root for convenience."""
        if 'root_word_id' in data:
            data['is_root'] = data['root_word_id'] is None
        return data

class WordSchema(Schema):
    """Schema for word serialization."""
    id = fields.Integer(dump_only=True)
    lemma = fields.String(required=True, validate=Length(min=1))
    normalized_lemma = fields.String(dump_only=True)
    language_code = fields.String(dump_default='tl', validate=Length(min=2, max=20))
    language_name = fields.String(dump_only=True)
    has_baybayin = fields.Boolean(dump_default=False)
    baybayin_form = fields.String(allow_none=True)
    romanized_form = fields.String(dump_only=True)
    root_word_id = fields.Integer(allow_none=True)
    is_root = fields.Boolean(dump_only=True)
    preferred_spelling = fields.String(allow_none=True)
    tags = fields.String(allow_none=True) # Changed to String to match DB TEXT
    idioms = MetadataField(dump_default={})
    pronunciation_data = MetadataField(dump_default={})
    source_info = MetadataField(dump_default={})
    word_metadata = MetadataField(dump_default={})
    data_hash = fields.String(allow_none=True)
    badlit_form = fields.String(allow_none=True)
    hyphenation = MetadataField(dump_default={})
    is_proper_noun = fields.Boolean(dump_default=False)
    is_abbreviation = fields.Boolean(dump_default=False)
    is_initialism = fields.Boolean(dump_default=False)
    completeness_score = fields.Float(dump_default=0.0, validate=Range(min=0.0, max=1.0))
    
    definitions = fields.List(fields.Nested(DefinitionSchema), dump_default=[])
    pronunciations = fields.List(fields.Nested(PronunciationType), dump_default=[])
    etymologies = fields.List(fields.Nested(EtymologySchema), dump_default=[])
    credits = fields.List(fields.Nested(CreditSchema), dump_default=[])
    
    # Relations and affixations
    outgoing_relations = fields.List(fields.Nested(RelationSchema), dump_default=[])
    incoming_relations = fields.List(fields.Nested(RelationSchema), dump_default=[])
    root_affixations = fields.List(fields.Nested(AffixationSchema), dump_default=[])
    affixed_affixations = fields.List(fields.Nested(AffixationSchema), dump_default=[])
    
    # Root word and derived words
    root_word = fields.Nested(WordSimpleSchema, dump_only=True)
    derived_words = fields.List(fields.Nested(WordSimpleSchema), dump_default=[])
    
    # Word forms, templates, and definition relations
    forms = fields.List(fields.Nested(WordFormSchema), dump_default=[])
    templates = fields.List(fields.Nested(WordTemplateSchema), dump_default=[])
    definition_relations = fields.List(fields.Nested(DefinitionRelationSchema), dump_default=[])
    related_definitions = fields.List(fields.Nested(DefinitionRelationSchema), dump_default=[])
    
    # Track timestamps
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    @post_dump
    def process_relations(self, data, **kwargs):
        """Add a combined relations array for convenience."""
        outgoing = data.get('outgoing_relations', [])
        incoming = data.get('incoming_relations', [])
        data['relations'] = outgoing + incoming
        
        # Add affixations combined array
        root_affixations = data.get('root_affixations', [])
        affixed_affixations = data.get('affixed_affixations', [])
        data['affixations'] = root_affixations + affixed_affixations
        
        # Add derived property
        data['is_root'] = data.get('root_word_id') is None
        
        return data

# --- Schemas moved from routes.py ---

class SearchQuerySchema(Schema):
    """Schema for search query parameters."""
    q = fields.Str(required=True)
    mode = fields.Str(validate=validate.OneOf(['all', 'exact', 'prefix', 'suffix']),
                     dump_default='all', load_default='all')
    limit = fields.Int(dump_default=50, load_default=50)
    offset = fields.Int(dump_default=0, load_default=0)
    
    # Sorting options
    sort = fields.Str(dump_default='relevance', load_default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), 
                      dump_default='desc', load_default='desc')
    
    include_full = fields.Bool(dump_default=False, load_default=False)
    include_definitions = fields.Bool(dump_default=True, load_default=True)
    include_pronunciations = fields.Bool(dump_default=True, load_default=True)
    include_etymologies = fields.Bool(dump_default=True, load_default=True)
    include_relations = fields.Bool(dump_default=True, load_default=True)
    include_forms = fields.Bool(dump_default=True, load_default=True)
    include_templates = fields.Bool(dump_default=True, load_default=True)
    include_metadata = fields.Bool(dump_default=True, load_default=True)
    
    # Relation expansion options
    include_related_words = fields.Bool(dump_default=False, load_default=False)
    include_definition_relations = fields.Bool(dump_default=False, load_default=False)
    
    # Filter parameters
    has_etymology = fields.Bool(dump_default=None, load_default=None)
    has_pronunciation = fields.Bool(dump_default=None, load_default=None)
    has_baybayin = fields.Bool(dump_default=None, load_default=None)
    exclude_baybayin = fields.Bool(dump_default=False, load_default=False)
    has_forms = fields.Bool(dump_default=None, load_default=None)
    has_templates = fields.Bool(dump_default=None, load_default=None)
    
    # Advanced filtering
    language = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(dump_default=None, load_default=None)

class SearchFilterSchema(Schema):
    """Schema for validating filter query parameters."""
    # Basic filters
    language = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(dump_default=None, load_default=None)
    
    # Feature filters
    has_baybayin = fields.Bool(dump_default=None, load_default=None)
    has_etymology = fields.Bool(dump_default=None, load_default=None)
    has_pronunciation = fields.Bool(dump_default=None, load_default=None)
    has_forms = fields.Bool(dump_default=None, load_default=None)
    
    # Date range filters
    date_added_from = fields.DateTime(dump_default=None, load_default=None)
    date_added_to = fields.DateTime(dump_default=None, load_default=None)
    date_modified_from = fields.DateTime(dump_default=None, load_default=None)
    date_modified_to = fields.DateTime(dump_default=None, load_default=None)
    
    # Definition and relation count filters
    min_definition_count = fields.Int(validate=validate.Range(min=0), dump_default=None, load_default=None)
    max_definition_count = fields.Int(validate=validate.Range(min=0), dump_default=None, load_default=None)
    min_relation_count = fields.Int(validate=validate.Range(min=0), dump_default=None, load_default=None)
    max_relation_count = fields.Int(validate=validate.Range(min=0), dump_default=None, load_default=None)
    
    # Completeness score range
    min_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    max_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    
    # Specific tags and categories
    tags = fields.List(fields.Str(), dump_default=None, load_default=None)
    categories = fields.List(fields.Str(), dump_default=None, load_default=None)

class StatisticsSchema(Schema):
    """Schema for dictionary statistics."""
    total_words = fields.Int()
    total_definitions = fields.Int()
    total_etymologies = fields.Int()
    total_relations = fields.Int()
    descendant_relations = fields.Int() # Added
    homophone_relations = fields.Int() # Added
    total_affixations = fields.Int()
    words_with_examples = fields.Int()
    words_with_etymology = fields.Int()
    words_with_relations = fields.Int()
    words_with_baybayin = fields.Int()
    words_by_language = fields.Dict(keys=fields.Str(), values=fields.Int())
    words_by_pos = fields.Dict(keys=fields.Str(), values=fields.Int())
    verification_stats = fields.Dict(keys=fields.Str(), values=fields.Int())
    quality_distribution = fields.Dict(keys=fields.Str(), values=fields.Int())
    update_frequency = fields.Dict(keys=fields.Str(), values=fields.Int())

class ExportFilterSchema(Schema):
    """Schema for export filter parameters."""
    language_code = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(dump_default=None, load_default=None)
    has_etymology = fields.Bool(dump_default=None, load_default=None)
    has_pronunciation = fields.Bool(dump_default=None, load_default=None)
    has_baybayin = fields.Bool(dump_default=None, load_default=None)
    min_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    created_after = fields.DateTime(dump_default=None, load_default=None)
    created_before = fields.DateTime(dump_default=None, load_default=None)
    updated_after = fields.DateTime(dump_default=None, load_default=None)
    updated_before = fields.DateTime(dump_default=None, load_default=None)
    include_relations = fields.Bool(dump_default=True, load_default=True)
    include_etymologies = fields.Bool(dump_default=True, load_default=True)
    include_pronunciations = fields.Bool(dump_default=True, load_default=True)
    include_definitions = fields.Bool(dump_default=True, load_default=True)
    include_credits = fields.Bool(dump_default=True, load_default=True)
    include_forms = fields.Bool(dump_default=True, load_default=True)
    include_templates = fields.Bool(dump_default=True, load_default=True)
    include_definition_relations = fields.Bool(dump_default=True, load_default=True)
    limit = fields.Int(validate=validate.Range(min=1, max=10000), dump_default=5000, load_default=5000)
    offset = fields.Int(validate=validate.Range(min=0), dump_default=0, load_default=0)
    format = fields.Str(validate=validate.OneOf(['json', 'csv', 'zip']), dump_default='json', load_default='json')

class QualityAssessmentFilterSchema(Schema):
    """Schema for quality assessment filter parameters."""
    language_code = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(dump_default=None, load_default=None)
    min_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    max_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    created_after = fields.DateTime(dump_default=None, load_default=None)
    created_before = fields.DateTime(dump_default=None, load_default=None)
    updated_after = fields.DateTime(dump_default=None, load_default=None)
    updated_before = fields.DateTime(dump_default=None, load_default=None)
    include_issues = fields.Bool(dump_default=True, load_default=True)
    issue_severity = fields.Str(validate=validate.OneOf(['all', 'critical', 'warning', 'info']), dump_default='all', load_default='all')
    max_results = fields.Int(validate=validate.Range(min=1, max=1000), dump_default=100, load_default=100)

# --- Delete from here onwards --- 