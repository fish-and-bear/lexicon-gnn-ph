"""
GraphQL schema for the Filipino Dictionary API.
"""

import graphene
from graphene import ObjectType, String, Int, List, Field, Boolean, JSONString
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from backend.models import (
    Word, Definition, Etymology, Relation, Affixation, 
    PartOfSpeech, Pronunciation
)
from sqlalchemy import func
import re
from typing import Dict, Any, Optional

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

class WordType(SQLAlchemyObjectType):
    """Type for words."""
    class Meta:
        model = Word
        interfaces = (graphene.relay.Node,)
        
    definitions = graphene.List(lambda: DefinitionType)
    etymology = graphene.List(lambda: EtymologyType)
    relations = graphene.List(lambda: RelationType)
    affixations = graphene.List(lambda: AffixationType)
    
    def resolve_definitions(self, info):
        return self.definitions
        
    def resolve_etymology(self, info):
        return self.etymologies
        
    def resolve_relations(self, info):
        return self.relations
        
    def resolve_affixations(self, info):
        return self.affixations

class DefinitionType(SQLAlchemyObjectType):
    """Type for word definitions."""
    class Meta:
        model = Definition
        interfaces = (graphene.relay.Node,)
    
    examples = graphene.List(ExampleType)
    part_of_speech = graphene.Field(lambda: PartOfSpeechType)
    
    def resolve_examples(self, info):
        if not self.examples:
            return []
        try:
            examples = json.loads(self.examples)
            return [ExampleType(**ex) if isinstance(ex, dict) else ExampleType(text=str(ex)) 
                   for ex in (examples if isinstance(examples, list) else [examples])]
        except json.JSONDecodeError:
            return [ExampleType(text=line.strip()) for line in self.examples.split('\n') if line.strip()]
    
    def resolve_part_of_speech(self, info):
        return self.standardized_pos

class EtymologyType(SQLAlchemyObjectType):
    """Type for word etymology."""
    class Meta:
        model = Etymology
        interfaces = (graphene.relay.Node,)
    
    components = graphene.List(lambda: EtymologyComponentType)
    
    def resolve_components(self, info):
        return [EtymologyComponentType(text=comp.text, language=comp.language)
                for comp in self.components] if self.components else []

class EtymologyComponentType(graphene.ObjectType):
    """Type for etymology components."""
    text = graphene.String(required=True)
    language = graphene.String()

class RelationType(SQLAlchemyObjectType):
    """Type for word relationships."""
    class Meta:
        model = Relation
        interfaces = (graphene.relay.Node,)
    
    word = graphene.Field(WordType)
    relation_type = graphene.String()
    
    def resolve_word(self, info):
        return self.related_word
    
    def resolve_relation_type(self, info):
        return self.relation_type

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
    
    examples = graphene.List(lambda: WordType)
    
    def resolve_examples(self, info):
        return Word.query.join(Definition).filter(
            Definition.standardized_pos_id == self.id
        ).limit(5).all()

class Query(graphene.ObjectType):
    """Root query type."""
    node = graphene.relay.Node.Field()
    
    # Word queries
    get_word = graphene.Field(
        WordType,
        word=graphene.String(required=True),
        language=graphene.String()
    )
    search_words = graphene.List(
        WordType,
        query=graphene.String(required=True),
        language=graphene.String(),
        pos=graphene.String(),
        has_baybayin=graphene.Boolean(),
        has_etymology=graphene.Boolean(),
        limit=graphene.Int()
    )
    get_word_network = graphene.Field(
        WordType,
        word=graphene.String(required=True)
    )
    
    # Parts of speech query
    parts_of_speech = graphene.List(PartOfSpeechType)
    
    # Baybayin processing
    baybayin_words = graphene.List(
        WordType,
        text=graphene.String(required=True)
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
                        base_query.filter(~Word.etymologies.any())
        
        return base_query.limit(limit).all()
    
    def resolve_get_word_network(self, info, word):
        """Get word and its relationships."""
        return Word.query.filter(func.lower(Word.lemma) == func.lower(word)).first()
    
    def resolve_parts_of_speech(self, info):
        """Get all parts of speech."""
        return PartOfSpeech.query.all()
    
    def resolve_baybayin_words(self, info, text):
        """Get words with Baybayin script."""
        return Word.query.filter(Word.has_baybayin == True).filter(
            Word.baybayin_form.ilike(f'%{text}%')
        ).all()

schema = graphene.Schema(query=Query) 