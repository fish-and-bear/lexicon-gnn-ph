"""
GraphQL schema for the Filipino Dictionary API.
"""

import graphene
from graphene import ObjectType, String, Int, List, Field
from graphene_sqlalchemy import SQLAlchemyObjectType
from models import Word, Definition, Etymology, Relation, Affixation, PartOfSpeech, Pronunciation

class WordType(SQLAlchemyObjectType):
    class Meta:
        model = Word
        interfaces = (graphene.relay.Node,)
        load_only = ()

class DefinitionType(SQLAlchemyObjectType):
    class Meta:
        model = Definition
        interfaces = (graphene.relay.Node,)
        load_only = ()

class EtymologyType(SQLAlchemyObjectType):
    class Meta:
        model = Etymology
        interfaces = (graphene.relay.Node,)
        load_only = ()

class RelationType(SQLAlchemyObjectType):
    class Meta:
        model = Relation
        interfaces = (graphene.relay.Node,)
        load_only = ()

class AffixationType(SQLAlchemyObjectType):
    class Meta:
        model = Affixation
        interfaces = (graphene.relay.Node,)
        load_only = ()

class PartOfSpeechType(SQLAlchemyObjectType):
    class Meta:
        model = PartOfSpeech
        interfaces = (graphene.relay.Node,)
        load_only = ()

class PronunciationType(SQLAlchemyObjectType):
    class Meta:
        model = Pronunciation
        interfaces = (graphene.relay.Node,)
        load_only = ()

class Query(ObjectType):
    word = Field(WordType, id=Int(required=True))
    words = List(WordType)
    search_words = List(WordType, query=String(required=True))

    def resolve_word(self, info, id):
        return Word.query.get(id)

    def resolve_words(self, info):
        return Word.query.all()

    def resolve_search_words(self, info, query):
        return Word.query.filter(
            Word.lemma.ilike(f'%{query}%')
        ).all()

schema = graphene.Schema(query=Query) 