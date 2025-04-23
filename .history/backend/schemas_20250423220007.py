"""
Marshmallow schemas for API serialization with enhanced validation and performance.
"""

from marshmallow import Schema, fields, pre_dump, post_dump, validates, ValidationError, validate
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
    """Schema for definition links."""
    id = fields.Integer(dump_only=True)
    definition_id = fields.Integer(required=True)
    link_text = fields.String(required=True)
    link_target = fields.String(required=True)
    is_wikipedia = fields.Boolean(dump_default=False)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class DefinitionCategorySchema(Schema):
    """Schema for definition categories."""
    id = fields.Integer(dump_only=True)
    definition_id = fields.Integer(required=True)
    category_name = fields.String(required=True)
    category_kind = fields.String(allow_none=True)
    parents = fields.List(fields.String(), allow_none=True)
    tags = MetadataField(dump_default={})
    category_metadata = MetadataField(dump_default={})
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class DefinitionSchema(Schema):
    """Schema for word definitions."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(dump_only=True)
    definition_text = fields.String(required=True, validate=Length(min=1))
    original_pos = fields.String(validate=Length(min=1, max=100))
    standardized_pos_id = fields.Integer()
    standardized_pos = fields.Nested("PartOfSpeechSchema", dump_only=True)
    notes = fields.String()
    examples = fields.List(fields.Dict(), dump_default=[])
    usage_notes = fields.String()
    cultural_notes = fields.String()
    etymology_notes = fields.String()
    scientific_name = fields.String()
    verified = fields.Boolean(dump_default=False)
    verification_notes = fields.String()
    tags = MetadataField(dump_default={})
    metadata = MetadataField(dump_default={})
    popularity_score = fields.Float(dump_default=0.0)
    links = fields.List(fields.Nested(DefinitionLinkSchema), dump_default=[])
    categories = fields.List(fields.Nested(DefinitionCategorySchema), dump_default=[])
    sources = fields.List(fields.String(), dump_default=[])
    
    # Track timestamps
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    @pre_dump
    def process_sources(self, data, **kwargs):
        """Convert sources from string to list if needed."""
        # Use getattr to safely access sources, returning None if not present
        source_value = getattr(data, 'sources', None)
        if source_value and isinstance(source_value, str):
            # Split by comma and strip whitespace
            data._sources_list = [s.strip() for s in source_value.split(',') if s.strip()]
        elif isinstance(source_value, list):
            # If it's already a list (e.g., from previous processing), use it directly
            data._sources_list = source_value
        else:
            # Handle None or other types by defaulting to empty list
            data._sources_list = []
        return data

    @post_dump(pass_many=True)
    def format_standardized_pos(self, data, many, **kwargs):
        """Format the standardized part of speech into a string representation."""
        if many:
            for item in data:
                # Assign the processed list to the final output field
                item['sources'] = getattr(item, '_sources_list', [])
                # Remove the temporary attribute if it exists
                if hasattr(item, '_sources_list'):
                    delattr(item, '_sources_list')
                    
                if item.get('standardized_pos'):
                    pos = item['standardized_pos']
                    item['standardized_pos_code'] = pos.get('code')
                    item['standardized_pos_name'] = pos.get('name')
        else:
            # Assign the processed list to the final output field
            data['sources'] = getattr(data, '_sources_list', [])
            # Remove the temporary attribute if it exists
            if hasattr(data, '_sources_list'):
                delattr(data, '_sources_list')

            if data.get('standardized_pos'):
                pos = data['standardized_pos']
                data['standardized_pos_code'] = pos.get('code')
                data['standardized_pos_name'] = pos.get('name')
        return data

class PartOfSpeechSchema(Schema):
    """Schema for parts of speech."""
    id = fields.Integer(dump_only=True)
    code = fields.String(required=True, validate=Length(min=1, max=10))
    name = fields.String(required=True, validate=Length(min=1, max=100))
    description = fields.String()

class PronunciationType(Schema):
    """Schema for pronunciations."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(dump_only=True)
    type = fields.String(required=True, validate=Length(min=1, max=50))
    value = fields.String(required=True)
    tags = fields.List(fields.String(), dump_default=[])
    pronunciation_metadata = MetadataField(dump_default={})
    sources = fields.String(allow_none=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class RelationSchema(Schema):
    """Schema for relations."""
    id = fields.Integer(dump_only=True)
    from_word_id = fields.Integer(required=True)
    to_word_id = fields.Integer(required=True)
    relation_type = fields.String(required=True)
    sources = fields.List(fields.String(), dump_default=[])
    metadata = MetadataField(dump_default={})
    source_word = fields.Nested("WordSimpleSchema", dump_only=True)
    target_word = fields.Nested("WordSimpleSchema", dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    @pre_dump
    def process_sources(self, data, **kwargs):
        """Convert sources from string to list if needed."""
        if hasattr(data, 'sources') and isinstance(data.sources, str):
            data._sources_list = data.sources.split(', ') if data.sources else []
        return data
    
    @post_dump
    def format_sources(self, data, **kwargs):
        """Ensure sources is a list in output."""
        if isinstance(data.get('sources'), str):
            data['sources'] = data['sources'].split(', ') if data['sources'] else []
        return data

class AffixationSchema(Schema):
    """Schema for affixations."""
    id = fields.Integer(dump_only=True)
    root_word_id = fields.Integer(required=True)
    affixed_word_id = fields.Integer(required=True)
    affix_type = fields.String(required=True)
    sources = fields.List(fields.String(), dump_default=[])
    root_word = fields.Nested("WordSimpleSchema", dump_only=True)
    affixed_word = fields.Nested("WordSimpleSchema", dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    @pre_dump
    def process_sources(self, data, **kwargs):
        """Convert sources from string to list if needed."""
        if hasattr(data, 'sources') and isinstance(data.sources, str):
            data._sources_list = data.sources.split(', ') if data.sources else []
        return data
    
    @post_dump
    def format_sources(self, data, **kwargs):
        """Ensure sources is a list in output."""
        if isinstance(data.get('sources'), str):
            data['sources'] = data['sources'].split(', ') if data['sources'] else []
        return data

class EtymologySchema(Schema):
    """Schema for etymologies."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(required=True)
    etymology_text = fields.String(required=True)
    normalized_components = fields.String()
    etymology_structure = fields.String()
    language_codes = fields.List(fields.String(), dump_default=[])
    sources = fields.List(fields.String(), dump_default=[])
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    @pre_dump
    def process_fields(self, data, **kwargs):
        """Preprocess language_codes and sources from strings to lists."""
        if hasattr(data, 'language_codes') and isinstance(data.language_codes, str):
            data._language_codes_list = data.language_codes.split(',') if data.language_codes else []
            
        if hasattr(data, 'sources') and isinstance(data.sources, str):
            data._sources_list = data.sources.split(', ') if data.sources else []
        return data
    
    @post_dump
    def format_fields(self, data, **kwargs):
        """Ensure fields are properly formatted in output."""
        # Format language_codes as a list
        if isinstance(data.get('language_codes'), str):
            data['language_codes'] = data['language_codes'].split(',') if data['language_codes'] else []
            
        # Format sources as a list
        if isinstance(data.get('sources'), str):
            data['sources'] = data['sources'].split(', ') if data['sources'] else []
            
        return data

class CreditSchema(Schema):
    """Schema for credits."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(required=True)
    credit = fields.String(required=True)
    sources = fields.List(fields.String(), dump_default=[])
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    @pre_dump
    def process_sources(self, data, **kwargs):
        """Convert sources from string to list if needed."""
        if hasattr(data, 'sources') and isinstance(data.sources, str):
            data._sources_list = data.sources.split(', ') if data.sources else []
        return data
    
    @post_dump
    def format_sources(self, data, **kwargs):
        """Ensure sources is a list in output."""
        if isinstance(data.get('sources'), str):
            data['sources'] = data['sources'].split(', ') if data['sources'] else []
        return data

class DefinitionRelationSchema(Schema):
    """Schema for definition relations."""
    id = fields.Integer(dump_only=True)
    definition_id = fields.Integer(required=True)
    word_id = fields.Integer(required=True)
    relation_type = fields.String(required=True)
    sources = fields.String(allow_none=True)
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
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class WordTemplateSchema(Schema):
    """Schema for word templates."""
    id = fields.Integer(dump_only=True)
    word_id = fields.Integer(required=True)
    template_name = fields.String(required=True)
    args = MetadataField(dump_default={})
    expansion = fields.String(allow_none=True)
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
    has_baybayin = fields.Boolean(dump_default=False)
    baybayin_form = fields.String(allow_none=True)
    romanized_form = fields.String(dump_only=True)
    root_word_id = fields.Integer(allow_none=True)
    is_root = fields.Boolean(dump_only=True)
    preferred_spelling = fields.String(allow_none=True)
    tags = fields.String(allow_none=True)
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

# --- Add Schemas retrieved from routes.py.bak ---

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

# Add schema for export/import filters
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