"""
GraphQL schema for the Filipino Dictionary API.
"""

import graphene
from graphene import ObjectType, String, Int, List, Field, Boolean, JSONString, Float, Enum
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from flask import current_app
from ..database import db
from ..models import Word, Definition, Etymology, Relation, Affixation, PartOfSpeech, Pronunciation
from dictionary_manager import (
    RelationshipType, RelationshipCategory, BaybayinRomanizer,
    normalize_lemma, extract_etymology_components, extract_language_codes
)
from sqlalchemy import func, not_
import json
import re
from typing import Dict, Any, Optional

# Ensure models are initialized
if not hasattr(Word, '__table__'):
    db.create_all()

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

class DefinitionType(SQLAlchemyObjectType):
    """Type for word definitions."""
    class Meta:
        model = Definition
        interfaces = (graphene.relay.Node,)

class EtymologyComponentType(ObjectType):
    """Type for etymology components with enhanced fields."""
    text = String(required=True)
    language = String()
    is_uncertain = Boolean()
    is_reconstructed = Boolean()
    period = String()
    notes = String()

class RelatedWordType(ObjectType):
    """Type for related words in etymology."""
    id = Int()
    lemma = String()
    language_code = String()
    component_text = String()

class EtymologyType(SQLAlchemyObjectType):
    """Enhanced type for word etymology."""
    class Meta:
        model = Etymology
        interfaces = (graphene.relay.Node,)

class RelationType(SQLAlchemyObjectType):
    """Enhanced type for word relationships."""
    class Meta:
        model = Relation
        interfaces = (graphene.relay.Node,)

class AffixationType(SQLAlchemyObjectType):
    """Type for word affixations."""
    class Meta:
        model = Affixation
        interfaces = (graphene.relay.Node,)

class PartOfSpeechType(SQLAlchemyObjectType):
    """Type for parts of speech."""
    class Meta:
        model = PartOfSpeech
        interfaces = (graphene.relay.Node,)

class PronunciationType(SQLAlchemyObjectType):
    """Type for word pronunciations."""
    class Meta:
        model = Pronunciation
        interfaces = (graphene.relay.Node,)

class BaybayinResultType(ObjectType):
    """Type for Baybayin processing results."""
    original = String()
    baybayin = String()
    romanized = String()
    is_valid = Boolean()
    error = String()

class Query(graphene.ObjectType):
    """Root query type."""
    node = graphene.relay.Node.Field()
    
    # Word queries
    get_word = Field(
        WordType,
        word=String(required=True),
        language=String()
    )
    search_words = List(
        WordType,
        query=String(required=True),
        language=String(),
        pos=String(),
        has_baybayin=Boolean(),
        has_etymology=Boolean(),
        limit=Int()
    )
    get_word_network = graphene.Field(
        WordNetworkType,
        word=graphene.String(required=True)
    )
    
    # Parts of speech query
    parts_of_speech = graphene.List(PartOfSpeechType)
    
    # Baybayin processing
    process_baybayin = List(
        BaybayinResultType,
        text=String(required=True)
    )
    
    def resolve_get_word(self, info, word, language=None):
        """Get word by lemma."""
        query = WordType.get_query(info).filter(func.lower(Word.lemma) == func.lower(word))
        if language:
            query = query.filter(Word.language_code == language)
        return query.first()
    
    def resolve_search_words(self, info, query, language=None, pos=None,
                           has_baybayin=None, has_etymology=None, limit=20):
        """Search words with filters."""
        base_query = WordType.get_query(info).filter(Word.lemma.ilike(f'%{query}%'))
        
        if language:
            base_query = base_query.filter(Word.language_code == language)
        if pos:
            base_query = base_query.join(Word.definitions).join(Definition.standardized_pos).filter(
                PartOfSpeech.code == pos
            )
        if has_baybayin is not None:
            base_query = base_query.filter(Word.has_baybayin == has_baybayin)
        if has_etymology is not None:
            base_query = base_query.join(Word.etymologies) if has_etymology else \
                        base_query.filter(not_(Word.etymologies.any()))
        
        return base_query.limit(limit).all()
    
    def resolve_get_word_network(self, info, word):
        """Get word and its relationships."""
        word_obj = WordType.get_query(info).filter(func.lower(Word.lemma) == func.lower(word)).first()
        if not word_obj:
            return None
            
        nodes = []
        links = []
        
        # Add the main word node
        nodes.append(NodeType(
            id=word_obj.lemma,
            group="main",
            info=NetworkWordInfoType(
                lemma=word_obj.lemma,
                pos=word_obj.definitions[0].standardized_pos.code if word_obj.definitions else None,
                definition=word_obj.definitions[0].text if word_obj.definitions else None
            )
        ))
        
        # Add related words
        for relation in word_obj.relations:
            related = relation.related_word
            if related:
                nodes.append(NodeType(
                    id=related.lemma,
                    group="related",
                    info=NetworkWordInfoType(
                        lemma=related.lemma,
                        pos=related.definitions[0].standardized_pos.code if related.definitions else None,
                        definition=related.definitions[0].text if related.definitions else None
                    )
                ))
                links.append(LinkType(
                    source=word_obj.lemma,
                    target=related.lemma,
                    type=relation.relation_type
                ))
        
        return WordNetworkType(nodes=nodes, links=links)
    
    def resolve_parts_of_speech(self, info):
        """Get all parts of speech."""
        return PartOfSpeechType.get_query(info).all()
    
    def resolve_process_baybayin(self, info, text):
        romanizer = BaybayinRomanizer()
        results = []
        
        for word in text.split():
            result = {'original': word}
            
            if romanizer.is_baybayin(word):
                result.update({
                    'baybayin': word,
                    'romanized': romanizer.romanize(word),
                    'is_valid': romanizer.validate_text(word)
                })
            else:
                try:
                    baybayin = transliterate_to_baybayin(word)
                    if baybayin:
                        result.update({
                            'baybayin': baybayin,
                            'romanized': word,
                            'is_valid': romanizer.validate_text(baybayin)
                        })
                    else:
                        result['error'] = 'Could not transliterate to Baybayin'
                except Exception as e:
                    result['error'] = str(e)
            
            results.append(result)
        
        return results

# Create the schema
schema = graphene.Schema(query=Query) 