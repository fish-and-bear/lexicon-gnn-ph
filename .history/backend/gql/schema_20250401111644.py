"""
GraphQL schema for the Filipino Dictionary API.
"""

import graphene
from graphene import ObjectType, String, Int, List, Field, Boolean, JSONString, Float, Enum
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from flask import current_app
from models import db, Word, Definition, Etymology, Relation, Affixation, PartOfSpeech, Pronunciation
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

class WordType(ObjectType):
    """Type for words."""
    id = Int()
    lemma = String()
    normalized_lemma = String()
    language_code = String()
    has_baybayin = Boolean()
    baybayin_form = String()
    romanized_form = String()
    verification_status = String()
    definitions = List(lambda: DefinitionType)
    etymologies = List(lambda: EtymologyType)
    relations = List(lambda: RelationType)
    affixations = List(lambda: AffixationType)
    
    @staticmethod
    def resolve_definitions(word, info):
        return word.definitions
    
    @staticmethod
    def resolve_etymologies(word, info):
        return word.etymologies
    
    @staticmethod
    def resolve_relations(word, info):
        return word.relations
    
    @staticmethod
    def resolve_affixations(word, info):
        return word.affixations

class DefinitionType(ObjectType):
    """Type for word definitions."""
    id = Int()
    definition_text = String()
    examples = List(lambda: ExampleType)
    part_of_speech = Field(lambda: PartOfSpeechType)
    
    @staticmethod
    def resolve_examples(definition, info):
        if not definition.examples:
            return []
        try:
            examples = json.loads(definition.examples)
            return [ExampleType(**ex) if isinstance(ex, dict) else ExampleType(text=str(ex)) 
                   for ex in (examples if isinstance(examples, list) else [examples])]
        except json.JSONDecodeError:
            return [ExampleType(text=line.strip()) for line in definition.examples.split('\n') if line.strip()]
    
    @staticmethod
    def resolve_part_of_speech(definition, info):
        return definition.standardized_pos

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

class EtymologyType(ObjectType):
    """Enhanced type for word etymology."""
    id = Int()
    etymology_text = String()
    components = List(EtymologyComponentType)
    language_codes = List(String)
    structure = JSONString()
    confidence_score = Float()
    sources = List(String)
    related_words = List(RelatedWordType)
    metadata = JSONString()
    
    @staticmethod
    def resolve_components(etymology, info):
        return extract_etymology_components(etymology.etymology_text)
    
    @staticmethod
    def resolve_language_codes(etymology, info):
        return extract_language_codes(etymology.etymology_text)
    
    @staticmethod
    def resolve_sources(etymology, info):
        return etymology.get_sources_list()
    
    @staticmethod
    def resolve_related_words(etymology, info):
        components = extract_etymology_components(etymology.etymology_text)
        related_words = []
        for comp in components:
            if comp.get('text'):
                related = Word.query.filter(
                    Word.normalized_lemma == normalize_lemma(comp['text']),
                    Word.language_code == comp.get('language', '')
                ).first()
                if related:
                    related_words.append({
                        'id': related.id,
                        'lemma': related.lemma,
                        'language_code': related.language_code,
                        'component_text': comp['text']
                    })
        return related_words

class RelationType(ObjectType):
    """Enhanced type for word relationships."""
    id = Int()
    relation_type = String()
    category = String()
    word = Field(lambda: WordType)
    bidirectional = Boolean()
    strength = Float()
    confidence_score = Float()
    sources = List(String)
    
    @staticmethod
    def resolve_relation_type(relation, info):
        return relation.type
    
    @staticmethod
    def resolve_category(relation, info):
        rel_type = RelationshipType.from_string(relation.type)
        return rel_type.category.value if rel_type else None
    
    @staticmethod
    def resolve_word(relation, info):
        return relation.target_word
    
    @staticmethod
    def resolve_bidirectional(relation, info):
        rel_type = RelationshipType.from_string(relation.type)
        return rel_type.bidirectional if rel_type else False
    
    @staticmethod
    def resolve_strength(relation, info):
        rel_type = RelationshipType.from_string(relation.type)
        return rel_type.strength if rel_type else 0.0
    
    @staticmethod
    def resolve_sources(relation, info):
        return relation.get_sources_list()

class AffixationType(ObjectType):
    """Type for word affixations."""
    id = Int()
    root_word = Field(lambda: WordType)
    affixed_word = Field(lambda: WordType)
    affix = String()
    affix_type = String()
    derived_form = String()
    meaning = String()
    notes = String()
    created_at = String()
    updated_at = String()
    
    @staticmethod
    def resolve_root_word(affixation, info):
        return affixation.root_word
    
    @staticmethod
    def resolve_affixed_word(affixation, info):
        return affixation.affixed_word

class PartOfSpeechType(ObjectType):
    """Type for parts of speech."""
    id = Int()
    code = String()
    name_en = String()
    name_tl = String()
    description = String()
    meta_info = JSONString()
    examples = List(lambda: WordType)
    
    @staticmethod
    def resolve_examples(pos, info):
        return Word.query.join(Definition).filter(
            Definition.standardized_pos_id == pos.id
        ).limit(5).all()

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
        query = Word.query.filter(func.lower(Word.lemma) == func.lower(word))
        if language:
            query = query.filter(Word.language_code == language)
        return query.first()
    
    def resolve_search_words(self, info, query, language=None, pos=None,
                           has_baybayin=None, has_etymology=None, limit=20):
        """Search words with filters."""
        base_query = Word.query.filter(Word.lemma.ilike(f'%{query}%'))
        
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
        word_obj = Word.query.filter(func.lower(Word.lemma) == func.lower(word)).first()
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
        return PartOfSpeech.query.all()
    
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