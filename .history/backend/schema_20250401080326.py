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
    DefinitionRelation as DefinitionRelationModel,
    Credit as CreditModel
)
from database import db_session, cached_query
from dictionary_manager import RelationshipType, RelationshipCategory
import json
from typing import List, Dict, Any, Optional
from marshmallow import Schema, fields, validate
import re
from datetime import datetime
from prometheus_client import Counter
import logging
from unidecode import unidecode

# Configure logging
logger = logging.getLogger(__name__)

# Metrics
GRAPHQL_QUERIES = Counter('graphql_queries_total', 'Total GraphQL queries', ['operation'])
GRAPHQL_ERRORS = Counter('graphql_errors_total', 'Total GraphQL errors', ['error_type'])

def normalize_lemma(text: str) -> str:
    """Normalize a lemma by removing diacritics and converting to lowercase."""
    if not text:
        return ""
    return unidecode(text).lower().strip()

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two strings."""
    if not text1 or not text2:
        return 0.0
    text1 = normalize_lemma(text1)
    text2 = normalize_lemma(text2)
    if text1 == text2:
        return 1.0
    # Simple substring match for now
    if text1 in text2 or text2 in text1:
        return 0.8
    # Levenshtein distance could be added here
    return 0.0

# Types for Relationships
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

class CreditType(SQLAlchemyObjectType):
    """Type for word credits."""
    class Meta:
        model = CreditModel
        interfaces = (graphene.relay.Node,)
    
    credit = graphene.String()
    sources = graphene.String()
    created_at = graphene.DateTime()
    updated_at = graphene.DateTime()
    word = graphene.Field(lambda: WordType)

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
    
    etymology_text = graphene.String()
    normalized_components = graphene.String()
    etymology_structure = graphene.String()
    language_codes = graphene.List(graphene.String)
    sources = graphene.String()
    created_at = graphene.DateTime()
    updated_at = graphene.DateTime()
    
    # Relationships
    word = graphene.Field(lambda: WordType)
    
    def resolve_language_codes(self, info):
        """Get language codes as a list."""
        if not self.language_codes:
            return []
        try:
            return json.loads(self.language_codes) if isinstance(self.language_codes, str) else self.language_codes
        except json.JSONDecodeError:
            return [code.strip() for code in self.language_codes.split(',') if code.strip()]

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
    original_pos = graphene.String()
    standardized_pos_id = graphene.Int()
    examples = graphene.List(ExampleType)
    usage_notes = graphene.String()
    tags = graphene.List(graphene.String)
    sources = graphene.String()
    created_at = graphene.DateTime()
    updated_at = graphene.DateTime()
    
    # Relationships
    word = graphene.Field(lambda: WordType)
    standardized_pos = graphene.Field(PartOfSpeechType)
    
    def resolve_examples(self, info):
        """Resolve examples to a list."""
        if not self.examples:
            return []
        try:
            return json.loads(self.examples) if isinstance(self.examples, str) else self.examples
        except json.JSONDecodeError:
            return [self.examples]
    
    def resolve_tags(self, info):
        """Resolve tags to a list."""
        if not self.tags:
            return []
        try:
            return json.loads(self.tags) if isinstance(self.tags, str) else self.tags
        except json.JSONDecodeError:
            return [tag.strip() for tag in self.tags.split(',') if tag.strip()]

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
    root_word_id = graphene.Int()
    preferred_spelling = graphene.String()
    tags = graphene.List(graphene.String)
    idioms = graphene.JSONString()
    pronunciation_data = graphene.JSONString()
    source_info = graphene.JSONString()
    word_metadata = graphene.JSONString()
    data_hash = graphene.String()
    created_at = graphene.DateTime()
    updated_at = graphene.DateTime()
    search_text = graphene.String()
    badlit_form = graphene.String()
    hyphenation = graphene.JSONString()
    is_proper_noun = graphene.Boolean()
    is_abbreviation = graphene.Boolean()
    is_initialism = graphene.Boolean()
    is_root = graphene.Boolean()
    
    # Relationships
    definitions = graphene.List(DefinitionType)
    etymologies = graphene.List(EtymologyType)
    pronunciations = graphene.List(PronunciationType)
    credits = graphene.List(lambda: CreditType)
    root_word = graphene.Field(lambda: WordType)
    derived_words = graphene.List(lambda: WordType)
    outgoing_relations = graphene.List(RelationType)
    incoming_relations = graphene.List(RelationType)
    root_affixations = graphene.List(AffixationType)
    affixed_affixations = graphene.List(AffixationType)
    
    def resolve_tags(self, info):
        """Resolve tags to a list."""
        if not self.tags:
            return []
        if isinstance(self.tags, list):
            return self.tags
        try:
            return json.loads(self.tags)
        except json.JSONDecodeError:
            return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
    def resolve_is_root(self, info):
        """Resolve is_root property."""
        return self.root_word_id is None

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
    
    get_word = graphene.Field(
        WordType,
        word=graphene.String(required=True),
        language=graphene.String()
    )
    
    baybayin_words = graphene.Field(
        graphene.List(WordType),
        text=graphene.String(required=True)
    )
    
    get_word_network = graphene.Field(
        WordType,
        word=graphene.String(required=True),
        max_depth=graphene.Int(default_value=2)
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
        relation_type=graphene.String(),
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
    
    def resolve_word(self, info, lemma, language=None):
        query = WordModel.query.filter(WordModel.normalized_lemma == normalize_lemma(lemma))
        if language:
            query = query.filter(WordModel.language_code == language)
        return query.first()
    
    def resolve_definition(self, info, id):
        return DefinitionModel.query.get(id)
    
    def resolve_etymology(self, info, id):
        return EtymologyModel.query.get(id)
    
    def resolve_get_word(self, info, word, language=None):
        """Resolve get_word query with enhanced error handling."""
        try:
            if not word:
                raise ValueError("Invalid word")
            
            normalized = normalize_lemma(word)
            query = WordModel.query.filter(WordModel.normalized_lemma == normalized)
            
            if language:
                query = query.filter(WordModel.language_code == language)
            
            result = query.first()
            if not result:
                logger.info("Word not found", word=word, language=language)
            
            return result
        except Exception as e:
            logger.error("Error in get_word resolver", error=str(e), traceback=True)
            return None
    
    def resolve_baybayin_words(self, info, text):
        """Resolve baybayin_words query with enhanced validation."""
        try:
            if not text:
                return []
            
            # Validate Baybayin text
            if not any(0x1700 <= ord(c) <= 0x171F for c in text):
                logger.warning("Invalid Baybayin text", text=text)
                return []
            
            return WordModel.query.filter(
                WordModel.has_baybayin == True,
                WordModel.baybayin_form.ilike(f"%{text}%")
            ).all()
        except Exception as e:
            logger.error("Error in baybayin_words resolver", error=str(e), traceback=True)
            return []
    
    def resolve_get_word_network(self, info, word, max_depth=2):
        """Resolve get_word_network query with depth control."""
        try:
            if not word:
                return None
            
            base_word = WordModel.query.filter(
                WordModel.normalized_lemma == normalize_lemma(word)
            ).first()
            
            if not base_word:
                return None
            
            # The network is built through the relationships defined in the model
            # The depth is controlled by the GraphQL query itself
            return base_word
        except Exception as e:
            logger.error("Error in get_word_network resolver", error=str(e), traceback=True)
            return None
    
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
            normalized_query = normalize_lemma(query)
            
            if mode == 'exact':
                base_query = base_query.filter(
                    WordModel.normalized_lemma == normalized_query
                )
            elif mode == 'baybayin':
                base_query = base_query.filter(
                    WordModel.has_baybayin == True,
                    WordModel.baybayin_form.ilike(f"%{query}%")
                )
            else:
                # Use ILIKE for basic text search
                base_query = base_query.filter(
                    or_(
                        WordModel.normalized_lemma.ilike(f"%{normalized_query}%"),
                        WordModel.lemma.ilike(f"%{query}%")
                    )
                )
            
            if language:
                base_query = base_query.filter(WordModel.language_code == language)
            
            # Order by exact matches first, then by similarity
            results = base_query.order_by(
                case(
                    [(WordModel.normalized_lemma == normalized_query, 0)],
                    else_=1
                ),
                WordModel.lemma
            ).limit(limit).all()
            
            return results
            
        except Exception as e:
            GRAPHQL_ERRORS.labels(error_type='search_error').inc()
            logger.error("Search error", error=str(e), traceback=True)
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