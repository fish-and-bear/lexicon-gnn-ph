"""
GraphQL schema for the Filipino Dictionary API with comprehensive type definitions and relationships.
"""

import graphene
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from sqlalchemy import func, or_, and_, not_, case
from models import (
    Word as WordModel,
    Definition as DefinitionModel,
    Etymology as EtymologyModel,
    Relation as RelationModel,
    Affixation as AffixationModel,
    PartOfSpeech as PartOfSpeechModel,
    Pronunciation as PronunciationModel,
    DefinitionRelation as DefinitionRelationModel
)
from database import db_session, cached_query
from dictionary_manager import RelationshipType, RelationshipCategory
import json
from typing import List, Dict, Any, Optional
from marshmallow import Schema, fields, validate
import re
from datetime import datetime
from prometheus_client import Counter
from collections import OrderedDict

# Metrics
GRAPHQL_QUERIES = Counter('graphql_queries_total', 'Total GraphQL queries', ['operation'])
GRAPHQL_ERRORS = Counter('graphql_errors_total', 'Total GraphQL errors', ['error_type'])

# Enums and Types for Relationships
class RelationshipTypeEnum(graphene.String):
    """GraphQL enum for relationship types."""

    @staticmethod
    def serialize(value):
        """Convert enum value to string."""
        if isinstance(value, RelationshipType):
            return value.value[0]  # Use the string value from the tuple
        return str(value) if value else None

    @staticmethod
    def parse_value(value):
        """Convert string to enum value."""
        try:
            return RelationshipType[value].value[0]
        except (KeyError, ValueError):
            return None

class RelationshipCategoryEnum(graphene.String):
    """GraphQL enum for relationship categories."""

    @staticmethod
    def serialize(value):
        """Convert enum value to string."""
        if isinstance(value, RelationshipCategory):
            return value.value
        return str(value) if value else None

    @staticmethod
    def parse_value(value):
        """Convert string to enum value."""
        try:
            return RelationshipCategory[value].value
        except (KeyError, ValueError):
            return None

class RelationshipMetadata(graphene.ObjectType):
    """Type for relationship metadata."""
    type = graphene.String(required=True)
    category = graphene.String(required=True)
    bidirectional = graphene.Boolean(required=True)
    inverse = graphene.String()
    transitive = graphene.Boolean(required=True)
    strength = graphene.Float(required=True)
    description = graphene.String()

    @staticmethod
    def from_relationship_type(rel_type: RelationshipType) -> 'RelationshipMetadata':
        """Convert a RelationshipType to RelationshipMetadata."""
        return RelationshipMetadata(
            type=rel_type.value[0],
            category=rel_type.category.value,
            bidirectional=rel_type.bidirectional,
            inverse=rel_type.inverse.value[0] if rel_type.inverse else None,
            transitive=rel_type.transitive,
            strength=rel_type.strength,
            description=rel_type.description
        )

class VerificationStatusEnum(graphene.String):
    """GraphQL enum for verification status."""

    @staticmethod
    def serialize(value):
        """Convert enum value to string."""
        return str(value) if value else None

    @staticmethod
    def parse_value(value):
        """Convert string to enum value."""
        valid_values = {'unverified', 'verified', 'needs_review', 'disputed'}
        return value if value in valid_values else None

class MetaInfoType(graphene.ObjectType):
    """Type for generic meta info fields."""
    strength = graphene.Float(description='Confidence strength of the relationship')
    confidence = graphene.Float(description='Confidence score of the data')
    tags = graphene.List(graphene.String, description='Associated tags')
    english_equivalent = graphene.String(description='English equivalent or translation')
    notes = graphene.String(description='Additional notes')
    source_details = graphene.JSONString(description='Detailed source information')
    created_at = graphene.DateTime(description='Creation timestamp')
    updated_at = graphene.DateTime(description='Last update timestamp')
    verification_status = graphene.String(description='Verification status')
    verification_notes = graphene.String(description='Notes from verification process')
    last_verified_at = graphene.DateTime(description='Last verification timestamp')

class PronunciationType(SQLAlchemyObjectType):
    """Type for pronunciation data."""
    class Meta:
        model = PronunciationModel
        interfaces = (graphene.relay.Node,)
    
    type = graphene.String()
    value = graphene.String()
    variants = graphene.List(graphene.String)
    phonemes = graphene.List(graphene.String)
    stress_pattern = graphene.String()
    syllable_count = graphene.Int()
    is_primary = graphene.Boolean()
    dialect = graphene.String()
    region = graphene.String()
    usage_frequency = graphene.Float()
    sources = graphene.List(graphene.String)
    meta_info = graphene.Field(MetaInfoType)

class EtymologyComponentType(graphene.ObjectType):
    """Type for etymology components."""
    text = graphene.String()
    language = graphene.String()
    meaning = graphene.String()
    notes = graphene.String()
    confidence = graphene.Float()
    is_reconstructed = graphene.Boolean()
    period = graphene.String()
    source = graphene.String()

class EtymologyType(SQLAlchemyObjectType):
    """Type for word etymologies."""
    class Meta:
        model = EtymologyModel
        interfaces = (graphene.relay.Node,)
    
    components = graphene.List(EtymologyComponentType)
    language_codes = graphene.List(graphene.String)
    structure = graphene.JSONString()
    confidence_score = graphene.Float()
    sources = graphene.List(graphene.String)
    meta_info = graphene.Field(MetaInfoType)
    
    def resolve_components(self, info):
        if not self.normalized_components:
            return []
        try:
            components = json.loads(self.normalized_components)
            return [EtymologyComponentType(**comp) for comp in components]
        except json.JSONDecodeError:
            return []

class ExampleType(graphene.ObjectType):
    """Type for usage examples."""
    text = graphene.String()
    translation = graphene.String()
    notes = graphene.String()
    source = graphene.String()
    tags = graphene.List(graphene.String)
    context = graphene.String()
    dialect = graphene.String()
    region = graphene.String()
    period = graphene.String()
    register = graphene.String()
    meta_info = graphene.Field(MetaInfoType)

class DefinitionType(SQLAlchemyObjectType):
    """Type for word definitions."""
    class Meta:
        model = DefinitionModel
        interfaces = (graphene.relay.Node,)
    
    definition_text = graphene.String()
    part_of_speech = graphene.Field(lambda: PartOfSpeechType)
    examples = graphene.List(ExampleType)
    usage_notes = graphene.List(graphene.String)
    register = graphene.String()
    domain = graphene.String()
    dialect = graphene.String()
    region = graphene.String()
    time_period = graphene.String()
    frequency = graphene.Float()
    confidence_score = graphene.Float()
    tags = graphene.List(graphene.String)
    sources = graphene.List(graphene.String)
    meta_info = graphene.Field(MetaInfoType)
    related_definitions = graphene.List(lambda: DefinitionType)

class RelationType(SQLAlchemyObjectType):
    """Type for word relationships."""
    class Meta:
        model = RelationModel
        interfaces = (graphene.relay.Node,)
    
    relation_type = graphene.String()
    category = graphene.String()
    bidirectional = graphene.Boolean()
    strength = graphene.Float()
    from_word = graphene.Field(lambda: WordType)
    to_word = graphene.Field(lambda: WordType)
    sources = graphene.List(graphene.String)
    meta_info = graphene.Field(MetaInfoType)
    metadata = graphene.Field(RelationshipMetadata)

    def resolve_metadata(self, info):
        """Resolve the full relationship metadata."""
        try:
            rel_type = RelationshipType[self.relation_type]
            return RelationshipMetadata.from_relationship_type(rel_type)
        except (KeyError, ValueError):
            return None

    def resolve_relation_type(self, info):
        """Resolve the relationship type."""
        try:
            rel_type = RelationshipType[self.relation_type]
            return rel_type.value[0]
        except (KeyError, ValueError):
            return self.relation_type

    def resolve_category(self, info):
        """Resolve the relationship category."""
        try:
            rel_type = RelationshipType[self.relation_type]
            return rel_type.category.value
        except (KeyError, ValueError):
            return None

class AffixationType(SQLAlchemyObjectType):
    """Type for word affixations."""
    class Meta:
        model = AffixationModel
        interfaces = (graphene.relay.Node,)
    
    affix_type = graphene.String()
    position = graphene.String()
    value = graphene.String()
    root_word = graphene.Field(lambda: WordType)
    affixed_word = graphene.Field(lambda: WordType)
    sources = graphene.List(graphene.String)
    meta_info = graphene.Field(MetaInfoType)

class PartOfSpeechType(SQLAlchemyObjectType):
    """Type for parts of speech."""
    class Meta:
        model = PartOfSpeechModel
        interfaces = (graphene.relay.Node,)
    
    code = graphene.String()
    name_en = graphene.String()
    name_tl = graphene.String()
    description = graphene.String()
    meta_info = graphene.Field(MetaInfoType)

class WordType(SQLAlchemyObjectType):
    """Type for dictionary words."""
    class Meta:
        model = WordModel
        interfaces = (graphene.relay.Node,)
    
    lemma = graphene.String()
    normalized_lemma = graphene.String()
    language_code = graphene.String()
    has_baybayin = graphene.Boolean()
    baybayin_form = graphene.String()
    romanized_form = graphene.String()
    is_root = graphene.Boolean()
    root_word = graphene.Field(lambda: WordType)
    preferred_spelling = graphene.String()
    alternative_spellings = graphene.List(graphene.String)
    syllable_count = graphene.Int()
    pronunciation_guide = graphene.String()
    stress_pattern = graphene.String()
    formality_level = graphene.String()
    usage_frequency = graphene.Float()
    geographic_region = graphene.String()
    time_period = graphene.String()
    cultural_notes = graphene.String()
    grammatical_categories = graphene.List(graphene.String)
    semantic_domains = graphene.List(graphene.String)
    etymology_confidence = graphene.Float()
    data_quality_score = graphene.Float()
    tags = graphene.List(graphene.String)
    definitions = graphene.List(DefinitionType)
    etymologies = graphene.List(EtymologyType)
    pronunciations = graphene.List(PronunciationType)
    relations = graphene.List(RelationType)
    affixations = graphene.List(AffixationType)
    meta_info = graphene.Field(MetaInfoType)

class Query(graphene.ObjectType):
    """Root query type with comprehensive search capabilities."""
    node = graphene.relay.Node.Field()
    
    # Single item queries
    word = graphene.Field(
        WordType,
        lemma=graphene.String(required=True),
        language=graphene.String()
    )
    
    definition = graphene.Field(
        DefinitionType,
        id=graphene.ID(required=True)
    )
    
    etymology = graphene.Field(
        EtymologyType,
        id=graphene.ID(required=True)
    )
    
    # List queries with filtering
    all_words = graphene.List(
        WordType,
        language=graphene.String(),
        has_baybayin=graphene.Boolean(),
        has_etymology=graphene.Boolean(),
        min_quality=graphene.Float(),
        limit=graphene.Int(),
        offset=graphene.Int()
    )
    
    search_words = graphene.List(
        WordType,
        query=graphene.String(required=True),
        mode=graphene.String(),
        language=graphene.String(),
        include_relations=graphene.Boolean(),
        min_similarity=graphene.Float(),
        limit=graphene.Int()
    )
    
    related_words = graphene.List(
        WordType,
        word_id=graphene.ID(required=True),
        relation_type=graphene.Field(RelationshipTypeEnum),
        max_distance=graphene.Int()
    )
    
    # Statistics queries
    word_statistics = graphene.Field(
        graphene.JSONString,
        language=graphene.String()
    )
    
    etymology_statistics = graphene.Field(
        graphene.JSONString,
        language=graphene.String()
    )
    
    # Implementation of resolvers
    def resolve_word(self, info, lemma, language=None):
        query = WordModel.query.filter(WordModel.normalized_lemma == normalize_lemma(lemma))
        if language:
            query = query.filter(WordModel.language_code == language)
        return query.first()
    
    def resolve_definition(self, info, id):
        return DefinitionModel.query.get(id)
    
    def resolve_etymology(self, info, id):
        return EtymologyModel.query.get(id)
    
    @cached_query(timeout=300)
    def resolve_all_words(self, info, language=None, has_baybayin=None,
                         has_etymology=None, min_quality=None,
                         limit=100, offset=0):
        query = WordModel.query
        
        if language:
            query = query.filter(WordModel.language_code == language)
        if has_baybayin is not None:
            query = query.filter(WordModel.has_baybayin == has_baybayin)
        if has_etymology is not None:
            if has_etymology:
                query = query.join(WordModel.etymologies)
            else:
                query = query.outerjoin(WordModel.etymologies).filter(
                    EtymologyModel.id.is_(None)
                )
        if min_quality is not None:
            query = query.filter(WordModel.quality_score >= min_quality)
        
        return query.order_by(WordModel.lemma).offset(offset).limit(limit).all()
    
    def resolve_search_words(self, info, query, mode='all', language=None,
                           include_relations=True, min_similarity=0.3, limit=10):
        GRAPHQL_QUERIES.labels(operation='search_words').inc()
        
        try:
            base_query = WordModel.query
            
            if mode == 'exact':
                base_query = base_query.filter(
                    WordModel.normalized_lemma == normalize_lemma(query)
                )
            elif mode == 'baybayin':
                base_query = base_query.filter(
                    WordModel.has_baybayin == True,
                    WordModel.baybayin_form.ilike(f"%{query}%")
                )
            else:
                base_query = base_query.filter(
                    WordModel.search_text.op('@@')(
                        func.plainto_tsquery('simple', query)
                    )
                )
            
            if language:
                base_query = base_query.filter(WordModel.language_code == language)
            
            results = base_query.order_by(
                func.similarity(WordModel.lemma, query).desc()
            ).limit(limit).all()
            
            return [r for r in results if calculate_similarity(r.lemma, query) >= min_similarity]
            
        except Exception as e:
            GRAPHQL_ERRORS.labels(error_type='search_error').inc()
            logger.error("Search error", error=str(e))
            return []
    
    @cached_query(timeout=3600)
    def resolve_word_statistics(self, info, language=None):
        try:
            query = db_session.query(
                func.count(WordModel.id).label('total_words'),
                func.count(case([(WordModel.has_baybayin == True, 1)])).label('baybayin_words'),
                func.count(case([(WordModel.root_word_id.is_(None), 1)])).label('root_words'),
                func.avg(WordModel.quality_score).label('avg_quality')
            )
            
            if language:
                query = query.filter(WordModel.language_code == language)
            
            stats = query.first()
            return {
                'total_words': stats.total_words,
                'baybayin_words': stats.baybayin_words,
                'root_words': stats.root_words,
                'average_quality': float(stats.avg_quality) if stats.avg_quality else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            GRAPHQL_ERRORS.labels(error_type='statistics_error').inc()
            logger.error("Statistics error", error=str(e))
            return {}

schema = graphene.Schema(query=Query)