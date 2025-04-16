"""
This module defines Marshmallow schemas for API serialization.

These schemas provide enhanced validation and performance compared to 
direct SQLAlchemy model serialization, and maintain a consistent
API interface even if the underlying data model changes.

Keys in these schemas must match the frontend expectations in the
React application's types.ts file.
"""

from marshmallow import Schema, fields, validate, pre_dump, post_dump
import json
from typing import Dict, Any, List, Optional, Union

# Custom field types
class JSONBField(fields.Field):
    """Field that handles PostgreSQL JSONB data."""
    
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return {}
        
    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value

class StringListField(fields.Field):
    """Field that converts comma-separated strings to/from lists."""
    
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            if not value.strip():
                return []
            return [item.strip() for item in value.split(',')]
        return []
        
    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return None
        if isinstance(value, list):
            return ','.join(str(item) for item in value)
        return value

# Base schemas
class BaseSchema(Schema):
    """Base schema with common metadata fields."""
    id = fields.Int(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    
    # Use StringListField to ensure consistent sources formatting between backend and frontend
    sources = StringListField()
    
    @post_dump
    def handle_empty_fields(self, data, **kwargs):
        """Ensure empty strings and None values are properly formatted."""
        # Convert empty strings to None for consistency
        for key, value in data.items():
            if value == "":
                data[key] = None
        return data

class DefinitionLinkSchema(BaseSchema):
    """Schema for definition links."""
    id = fields.Int(dump_only=True)
    definition_id = fields.Int(required=True)
    link_text = fields.Str(required=True)  # Frontend expects link_text (not link_type)
    target_url = fields.Str(required=True)
    display_text = fields.Str(allow_none=True)
    is_external = fields.Bool(default=False)
    tags = JSONBField()  # Ensure tags field is consistent
    link_metadata = JSONBField()  # Frontend expects link_metadata 
    definition = fields.Nested('DefinitionSchema', only=('id', 'definition_text'), dump_default=None)

class DefinitionCategorySchema(BaseSchema):
    """Schema for definition categories."""
    id = fields.Int(dump_only=True)
    definition_id = fields.Int(required=True)
    category_name = fields.Str(required=True) 
    category_kind = fields.Str(allow_none=True)
    tags = JSONBField()  # Ensure consistent handling for JSONB
    category_metadata = JSONBField()  # Frontend expects category_metadata
    parents = StringListField(dump_default=[])  # Frontend expects array
    definition = fields.Nested('DefinitionSchema', only=('id', 'definition_text'), dump_default=None)

class PartOfSpeechSchema(Schema):
    """Schema for parts of speech."""
    id = fields.Int(dump_only=True)
    code = fields.Str(required=True)
    name_en = fields.Str(required=True)
    name_tl = fields.Str(required=True)
    description = fields.Str(allow_none=True)
    
    # Relationships
    derived_words = fields.List(fields.Nested('WordSchema', only=('id', 'lemma', 'language_code')))
    forms = fields.List(fields.Nested('WordFormSchema', exclude=("word",)))
    templates = fields.List(fields.Nested('WordTemplateSchema', exclude=("word",)))
    completeness_score = fields.Float(dump_only=True)

class DefinitionSchema(BaseSchema):
    """Schema for word definitions."""
    id = fields.Int(dump_only=True)
    definition_text = fields.Str(required=True)
    word_id = fields.Int()
    standardized_pos_id = fields.Int(allow_none=True)
    original_pos = fields.Str(allow_none=True)
    usage_notes = fields.Str(allow_none=True)
    examples = StringListField(dump_default=[])  # Frontend expects array
    sources = StringListField()  # Frontend expects array
    definition_metadata = JSONBField()
    
    # Frontend expects standardized_pos as PartOfSpeech object
    standardized_pos = fields.Nested(PartOfSpeechSchema, dump_default=None)
    
    # Nested relationships
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'), dump_default=None)
    categories = fields.List(fields.Nested(DefinitionCategorySchema, exclude=('definition',)))
    links = fields.List(fields.Nested(DefinitionLinkSchema, exclude=('definition',)))
    
    # Related definitions/words relationships
    definition_relations = fields.List(fields.Nested('DefinitionRelationSchema', exclude=('definition',)))
    related_words = fields.List(fields.Nested('WordSchema', only=('id', 'lemma', 'language_code')))

class DefinitionRelationSchema(BaseSchema):
    """Schema for definition relation data."""
    relation_type = fields.Str(required=True)
    definition_id = fields.Int(required=True) 
    word_id = fields.Int(required=True)
    relation_data = JSONBField()  # Frontend expects relation_data
    definition = fields.Nested(DefinitionSchema, only=('id', 'definition_text'))
    related_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

class EtymologySchema(BaseSchema):
    """Schema for etymology data."""
    etymology_text = fields.Str(required=True)
    normalized_components = StringListField()  # Frontend expects array
    etymology_structure = fields.Str(allow_none=True)
    language_codes = StringListField()  # Frontend expects array
    sources = StringListField()  # Ensure sources is array format
    etymology_data = JSONBField()  # Frontend expects etymology_data
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

class PronunciationSchema(BaseSchema):
    """Schema for pronunciation data."""
    type = fields.Str(validate=validate.OneOf(['ipa', 'respelling', 'audio', 'phonemic', 'x-sampa', 'pinyin', 'jyutping', 'romaji']))
    value = fields.Str(required=True)
    tags = JSONBField()
    pronunciation_metadata = JSONBField()  # Frontend expects pronunciation_metadata
    sources = StringListField()  # Frontend expects array
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

class RelationSchema(BaseSchema):
    """Schema for word relationships."""
    relation_type = fields.Str(required=True)
    relation_data = JSONBField()  # Frontend expects relation_data
    source_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))
    target_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))

class AffixationSchema(BaseSchema):
    """Schema for word affixation data."""
    affix_type = fields.Str(required=True)
    sources = StringListField()  # Frontend expects array
    root_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))
    affixed_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

class CreditSchema(BaseSchema):
    """Schema for word credits."""
    credit = fields.Str(required=True)

class WordFormSchema(BaseSchema):
    """Schema for word forms (inflections, conjugations)."""
    form = fields.Str(required=True)
    tags = JSONBField() 
    is_canonical = fields.Bool(default=False)
    is_primary = fields.Bool(default=False)
    word = fields.Nested('WordSchema', only=('id', 'lemma'))

class WordTemplateSchema(BaseSchema):
    """Schema for word templates."""
    template_name = fields.Str(required=True)
    args = JSONBField()
    expansion = fields.Str(allow_none=True)
    word = fields.Nested('WordSchema', only=('id', 'lemma'))

class WordSchema(BaseSchema):
    """Schema for word entries."""
    id = fields.Int(dump_only=True)
    lemma = fields.Str(required=True)
    normalized_lemma = fields.Str(allow_none=True)
    language_code = fields.Str(required=True)
    has_baybayin = fields.Bool(allow_none=True)
    baybayin_form = fields.Str(allow_none=True)
    romanized_form = fields.Str(allow_none=True)
    root_word_id = fields.Int(allow_none=True)
    preferred_spelling = fields.Str(allow_none=True)
    tags = StringListField()
    idioms = JSONBField()
    source_info = JSONBField()
    word_metadata = JSONBField()
    data_hash = fields.Str(allow_none=True)
    search_text = fields.Str(allow_none=True)
    badlit_form = fields.Str(allow_none=True)
    hyphenation = JSONBField()
    is_proper_noun = fields.Bool(allow_none=True)
    is_abbreviation = fields.Bool(allow_none=True)
    is_initialism = fields.Bool(allow_none=True)
    is_root = fields.Bool(allow_none=True)
    completeness_score = fields.Float(dump_only=True)
    
    # Relationships - ensure names match frontend expectations
    definitions = fields.List(fields.Nested(DefinitionSchema, exclude=("word",)))
    etymologies = fields.List(fields.Nested(EtymologySchema, exclude=("word",)))
    pronunciations = fields.List(fields.Nested(PronunciationSchema, exclude=("word",)))
    credits = fields.List(fields.Nested(CreditSchema, exclude=("word",)))
    outgoing_relations = fields.List(fields.Nested(RelationSchema, exclude=("source_word",)))
    incoming_relations = fields.List(fields.Nested(RelationSchema, exclude=("target_word",)))
    root_affixations = fields.List(fields.Nested(AffixationSchema, exclude=("root_word",)))
    affixed_affixations = fields.List(fields.Nested(AffixationSchema, exclude=("affixed_word",)))

    # Root word relationship 
    root_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'), dump_default=None)
    derived_words = fields.List(fields.Nested('WordSchema', only=('id', 'lemma', 'language_code')))

    # Forms and templates
    forms = fields.List(fields.Nested(WordFormSchema, exclude=("word",)))
    templates = fields.List(fields.Nested(WordTemplateSchema, exclude=("word",)))
    
    # Definition relations
    definition_relations = fields.List(fields.Nested(DefinitionRelationSchema, exclude=("related_word",)))
    related_definitions = fields.List(fields.Nested(DefinitionSchema, exclude=("word", "related_words")))
    
    @post_dump
    def process_tags(self, data, **kwargs):
        """Ensure tags is properly formatted as array for frontend."""
        if 'tags' in data and isinstance(data['tags'], str):
            if data['tags']:
                data['tags'] = [tag.strip() for tag in data['tags'].split(',')]
            else:
                data['tags'] = []
        return data 