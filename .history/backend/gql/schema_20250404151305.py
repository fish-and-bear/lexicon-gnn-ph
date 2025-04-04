"""
GraphQL schema for the Filipino Dictionary API.
"""

import graphene
from graphene import ObjectType, String, Int, List, Field, Boolean, JSONString, Float, Enum
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from flask import current_app
from database import db
from backend.models import Word, Definition, Etymology, Relation, Affixation, PartOfSpeech, Pronunciation, Credit
from dictionary_manager import (
    RelationshipType, RelationshipCategory, BaybayinRomanizer,
    normalize_lemma, extract_etymology_components, extract_language_codes,
    standardize_source_identifier
)
from sqlalchemy import func, not_, or_
from sqlalchemy.orm import selectinload
import json
import re
from typing import Dict, Any, Optional

# No need to create tables, they already exist
# if not hasattr(Word, '__table__'):
#     db.create_all()

# Enums for standardized types
class RelationshipTypeEnum(Enum):
    """Enum for relationship types."""
    SYNONYM = 'SYNONYM'
    ANTONYM = 'ANTONYM'
    HYPERNYM = 'HYPERNYM'
    HYPONYM = 'HYPONYM'
    MERONYM = 'MERONYM'
    HOLONYM = 'HOLONYM'
    DERIVED = 'DERIVED'
    ROOT = 'ROOT'
    VARIANT = 'VARIANT'
    COGNATE = 'COGNATE'
    COMPOUND = 'COMPOUND'
    COMPONENT = 'COMPONENT'

class RelationshipCategoryEnum(Enum):
    """Enum for relationship categories."""
    SEMANTIC = 'SEMANTIC'
    MORPHOLOGICAL = 'MORPHOLOGICAL'
    ETYMOLOGICAL = 'ETYMOLOGICAL'
    ORTHOGRAPHIC = 'ORTHOGRAPHIC'
    PHONOLOGICAL = 'PHONOLOGICAL'

class MetaInfoType(graphene.ObjectType):
    """Type for generic meta info fields."""
    strength = graphene.Float()
    tags = graphene.List(graphene.String)
    english_equivalent = graphene.String()
    notes = graphene.String()
    source_details = graphene.JSONString()

class ExampleType(graphene.ObjectType):
    """Type for word usage examples."""
    text = graphene.String(required=True)
    translation = graphene.String()
    notes = graphene.String()
    source = graphene.String()
    tags = graphene.List(graphene.String)

class NetworkWordInfoType(graphene.ObjectType):
    """Type for word info in network."""
    lemma = graphene.String()
    pos = graphene.String()
    definition = graphene.String()

class NodeType(graphene.ObjectType):
    """Type for network nodes."""
    id = graphene.String()
    group = graphene.String()
    info = graphene.Field(NetworkWordInfoType)
    meta_info = graphene.Field(MetaInfoType)

class LinkType(graphene.ObjectType):
    """Type for network links."""
    source = graphene.String()
    target = graphene.String()
    type = graphene.String()
    meta_info = graphene.Field(MetaInfoType)

class WordNetworkType(graphene.ObjectType):
    """Type for word relationship networks."""
    nodes = graphene.List(NodeType)
    links = graphene.List(LinkType)
    meta_info = graphene.Field(MetaInfoType)

class BaybayinWordType(graphene.ObjectType):
    """Type for Baybayin word processing results."""
    original = graphene.String()
    baybayin = graphene.String()
    latin = graphene.String()

class WordType(SQLAlchemyObjectType):
    """Type for words."""
    class Meta:
        model = Word
        interfaces = (graphene.relay.Node,)
        exclude_fields = ('search_text', 'data_hash', 'root_word_id', 'source_info')

    definitions = List(lambda: DefinitionType)
    etymologies = List(lambda: EtymologyType)
    pronunciations = List(lambda: PronunciationType)
    credits = List(lambda: CreditType)
    outgoing_relations = List(lambda: RelationType)
    incoming_relations = List(lambda: RelationType)
    root_affixations = List(lambda: AffixationType)
    affixed_affixations = List(lambda: AffixationType)
    root_word = Field(lambda: WordType)
    derived_words = List(lambda: WordType)

    def resolve_romanized_form(self, info):
        if self.has_baybayin and self.baybayin_form:
            try:
                romanizer = BaybayinRomanizer()
                return romanizer.romanize(self.baybayin_form)
            except Exception as e:
                current_app.logger.error(f"Error romanizing baybayin for word {self.id}: {e}")
                return f"Error romanizing: {e}"
        return self.romanized_form or self.lemma

    sources = String()
    def resolve_sources(self, info):
        source_info = self.source_info or {}
        files = source_info.get('files', [])
        if isinstance(files, list):
            standardized_list = [standardize_source_identifier(s) for s in files]
            return ', '.join(sorted(list(set(standardized_list))))
        return None

class DefinitionType(SQLAlchemyObjectType):
    """Type for word definitions."""
    class Meta:
        model = Definition
        interfaces = (graphene.relay.Node,)
        exclude_fields = ('word_id', 'standardized_pos_id')

    word = Field(lambda: WordType)
    standardized_pos = Field(lambda: PartOfSpeechType)

    examples_list = List(ExampleType, name="examples")
    def resolve_examples_list(self, info):
        raw_examples = self.examples
        if not raw_examples:
            return []
        try:
            if raw_examples.strip().startswith('['):
                examples_data = json.loads(raw_examples)
            else:
                examples_data = [line.strip() for line in raw_examples.split('\\n') if line.strip()]

            result_list = []
            for item in examples_data:
                if isinstance(item, dict):
                    valid_item = {k: v for k, v in item.items() if hasattr(ExampleType, k)}
                    result_list.append(ExampleType(**valid_item))
                elif isinstance(item, str):
                    result_list.append(ExampleType(text=item))
            return result_list
        except (json.JSONDecodeError, TypeError) as e:
            current_app.logger.error(f"Error parsing examples for definition {self.id}: {e}")
            return [ExampleType(text=raw_examples)]

    sources_str = String(name="sources")
    def resolve_sources_str(self, info):
        if not self.sources:
            return None
        if isinstance(self.sources, str):
            source_list = [s.strip() for s in self.sources.split(',') if s.strip()]
            standardized_list = [standardize_source_identifier(s) for s in source_list]
            return ', '.join(sorted(list(set(standardized_list))))
        return self.sources

class EtymologyType(SQLAlchemyObjectType):
    """Enhanced type for word etymology."""
    class Meta:
        model = Etymology
        interfaces = (graphene.relay.Node,)
        exclude_fields = ('word_id',)

    word = Field(lambda: WordType)
    components = List(JSONString)
    def resolve_components(self, info):
        if not self.etymology_text:
            return []
        try:
            extracted = extract_etymology_components(self.etymology_text)
            if isinstance(extracted, dict) and 'original_text' in extracted:
                return [json.dumps(extracted)]
            elif isinstance(extracted, list):
                serializable_comps = []
                for comp in extracted:
                    try:
                        serializable_comps.append(json.dumps(comp))
                    except TypeError:
                        serializable_comps.append(json.dumps(str(comp)))
                return serializable_comps
            return []
        except Exception as e:
            current_app.logger.error(f"Error extracting components for etymology {self.id}: {e}")
            return []

    language_codes_list = List(String, name="language_codes")
    def resolve_language_codes_list(self, info):
        if not self.language_codes:
            return []
        return [code.strip() for code in self.language_codes.split(',') if code.strip()]

    sources_str = String(name="sources")
    def resolve_sources_str(self, info):
        if not self.sources:
            return None
        if isinstance(self.sources, str):
            source_list = [s.strip() for s in self.sources.split(',') if s.strip()]
            standardized_list = [standardize_source_identifier(s) for s in source_list]
            return ', '.join(sorted(list(set(standardized_list))))
        return self.sources

class PartOfSpeechType(SQLAlchemyObjectType):
    """Type for parts of speech."""
    class Meta:
        model = PartOfSpeech
        interfaces = (graphene.relay.Node,)

    definitions = List(lambda: DefinitionType)

class PronunciationType(SQLAlchemyObjectType):
    """Type for word pronunciations."""
    class Meta:
        model = Pronunciation
        interfaces = (graphene.relay.Node,)
        exclude_fields = ('word_id',)

    word = Field(lambda: WordType)
    tags = JSONString()
    metadata = JSONString()

    sources_str = String(name="sources")
    def resolve_sources_str(self, info):
        if not self.sources:
            return None
        if isinstance(self.sources, str):
            source_list = [s.strip() for s in self.sources.split(',') if s.strip()]
            standardized_list = [standardize_source_identifier(s) for s in source_list]
            return ', '.join(sorted(list(set(standardized_list))))
        return self.sources

class CreditType(SQLAlchemyObjectType):
    """Type for word credits."""
    class Meta:
        model = Credit
        interfaces = (graphene.relay.Node,)
        exclude_fields = ('word_id',)

    word = Field(lambda: WordType)

    sources_str = String(name="sources")
    def resolve_sources_str(self, info):
        if not self.sources:
            return None
        if isinstance(self.sources, str):
            source_list = [s.strip() for s in self.sources.split(',') if s.strip()]
            standardized_list = [standardize_source_identifier(s) for s in source_list]
            return ', '.join(sorted(list(set(standardized_list))))
        return self.sources

class RelationType(SQLAlchemyObjectType):
    """Enhanced type for word relationships."""
    class Meta:
        model = Relation
        interfaces = (graphene.relay.Node,)
        exclude_fields = ('from_word_id', 'to_word_id')

    source_word = Field(lambda: WordType)
    target_word = Field(lambda: WordType)
    metadata = JSONString()

    category = String()
    def resolve_category(self, info):
        try:
            rel_enum = RelationshipType.from_string(self.relation_type)
            return rel_enum.category.value if rel_enum else RelationshipCategory.OTHER.value
        except Exception as e:
            current_app.logger.warning(f"Error resolving category for relation type '{self.relation_type}': {e}")
            return RelationshipCategory.OTHER.value

    bidirectional = Boolean()
    def resolve_bidirectional(self, info):
        try:
            rel_enum = RelationshipType.from_string(self.relation_type)
            return rel_enum.bidirectional if rel_enum else False
        except Exception as e:
            current_app.logger.warning(f"Error resolving bidirectionality for relation type '{self.relation_type}': {e}")
            return False

    sources_str = String(name="sources")
    def resolve_sources_str(self, info):
        if not self.sources:
            return None
        if isinstance(self.sources, str):
            source_list = [s.strip() for s in self.sources.split(',') if s.strip()]
            standardized_list = [standardize_source_identifier(s) for s in source_list]
            return ', '.join(sorted(list(set(standardized_list))))
        return self.sources

class AffixationType(SQLAlchemyObjectType):
    """Type for word affixations."""
    class Meta:
        model = Affixation
        interfaces = (graphene.relay.Node,)
        exclude_fields = ('root_word_id', 'affixed_word_id')

    root_word = Field(lambda: WordType)
    affixed_word = Field(lambda: WordType)

    sources_str = String(name="sources")
    def resolve_sources_str(self, info):
        if not self.sources:
            return None
        if isinstance(self.sources, str):
            source_list = [s.strip() for s in self.sources.split(',') if s.strip()]
            standardized_list = [standardize_source_identifier(s) for s in source_list]
            return ', '.join(sorted(list(set(standardized_list))))
        return self.sources

class BaybayinResultType(ObjectType):
    """Type for Baybayin processing results."""
    original = String()
    baybayin = String()
    romanized = String()
    is_valid = Boolean()
    error = String()

class Query(ObjectType):
    """Root query type."""
    node = graphene.relay.Node.Field()
    
    # Word queries
    get_word = Field(
        WordType,
        id=Int(),
        lemma=String(),
        language=String()
    )
    search_words = List(
        WordType,
        query=String(required=True),
        language=String(),
        pos=String(),
        has_baybayin=Boolean(),
        has_etymology=Boolean(),
        limit=Int(default_value=20)
    )
    
    # Parts of speech query
    parts_of_speech = List(PartOfSpeechType)
    
    def resolve_get_word(self, info, id=None, lemma=None, language=None):
        """Get word by ID or lemma using ORM with eager loading."""
        if id is None and lemma is None:
            raise ValueError("Must provide either 'id' or 'lemma' to get_word")

        query = Word.query

        if id is not None:
            query = query.filter(Word.id == id)
        elif lemma is not None:
            query = query.filter(Word.normalized_lemma == normalize_lemma(lemma))
            if language:
                query = query.filter(Word.language_code == language)

        query = query.options(
            selectinload(Word.definitions).selectinload(Definition.standardized_pos),
            selectinload(Word.pronunciations),
            selectinload(Word.credits),
            selectinload(Word.etymologies),
            selectinload(Word.outgoing_relations).selectinload(Relation.target_word),
            selectinload(Word.incoming_relations).selectinload(Relation.source_word),
            selectinload(Word.root_affixations).selectinload(Affixation.affixed_word),
            selectinload(Word.affixed_affixations).selectinload(Affixation.root_word),
            selectinload(Word.derived_words),
            selectinload(Word.root_word)
        )

        return query.first()

    def resolve_search_words(self, info, query, language=None, pos=None,
                           has_baybayin=None, has_etymology=None, limit=20):
        """Search words with filters using ORM."""
        base_query = Word.query

        norm_query = normalize_lemma(query)
        base_query = base_query.filter(
            or_(
                Word.lemma.ilike(f'%{query}%'),
                Word.normalized_lemma.ilike(f'%{norm_query}%')
            )
        )

        if language:
            base_query = base_query.filter(Word.language_code == language)
        if pos:
            base_query = base_query.join(Word.definitions).join(Definition.standardized_pos).filter(
                PartOfSpeech.code == pos.lower()
            ).distinct()
        if has_baybayin is not None:
            base_query = base_query.filter(Word.has_baybayin == has_baybayin)
        if has_etymology is not None:
            if has_etymology:
                base_query = base_query.join(Word.etymologies).distinct()
            else:
                subquery = db.session.query(Etymology.word_id).distinct()
                base_query = base_query.filter(Word.id.notin_(subquery))

        base_query = base_query.limit(limit).options(
            selectinload(Word.definitions).selectinload(Definition.standardized_pos)
        )

        return base_query.all()

    def resolve_parts_of_speech(self, info):
        """Get all parts of speech."""
        return PartOfSpeech.query.order_by(PartOfSpeech.code).all()

# Create the schema
schema = graphene.Schema(query=Query, types=[
    WordType, DefinitionType, EtymologyType, RelationType, AffixationType,
    PartOfSpeechType, PronunciationType, CreditType, ExampleType, MetaInfoType,
    NodeType, LinkType, WordNetworkType, BaybayinWordType, BaybayinResultType
])

# Function to initialize GraphQL (assuming it's used elsewhere, e.g., in app.py)
def init_graphql():
    # This function might configure things or simply return the schema
    return schema, Query

# Create the schema
schema = graphene.Schema(query=Query, types=[
    WordType, DefinitionType, EtymologyType, RelationType, AffixationType,
    PartOfSpeechType, PronunciationType, CreditType, ExampleType, MetaInfoType,
    NodeType, LinkType, WordNetworkType, BaybayinWordType, BaybayinResultType
])

# Function to initialize GraphQL (assuming it's used elsewhere, e.g., in app.py)
def init_graphql():
    # This function might configure things or simply return the schema
    return schema, Query 