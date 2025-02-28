import graphene
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from sqlalchemy import func
from models import (
    Word as WordModel,
    Definition as DefinitionModel,
    Meaning as MeaningModel,
    Etymology as EtymologyModel,
    EtymologyComponent as EtymologyComponentModel,
    Form as FormModel,
    HeadTemplate as HeadTemplateModel,
    Inflection as InflectionModel,
    Language as LanguageModel,
    Derivative as DerivativeModel,
    Example as ExampleModel,
    AssociatedWord as AssociatedWordModel,
    AlternateForm as AlternateFormModel,
    Hypernym as HypernymModel,
    Hyponym as HyponymModel,
    Meronym as MeronymModel,
    Holonym as HolonymModel
)
from database import db_session

class PronunciationType(graphene.ObjectType):
    text = graphene.String()
    ipa = graphene.String()
    audio = graphene.List(graphene.String)

class EtymologyComponentType(SQLAlchemyObjectType):
    class Meta:
        model = EtymologyComponentModel
        interfaces = (graphene.relay.Node, )

class EtymologyType(SQLAlchemyObjectType):
    class Meta:
        model = EtymologyModel
        interfaces = (graphene.relay.Node, )
    
    components = graphene.List(EtymologyComponentType)
    parsed = graphene.List(graphene.String)

    def resolve_components(self, info):
        return sorted(self.components, key=lambda x: x.order)

    def resolve_parsed(self, info):
        from word_network import parse_etymology
        return parse_etymology(self.etymology_text)

class MeaningType(SQLAlchemyObjectType):
    class Meta:
        model = MeaningModel
        interfaces = (graphene.relay.Node, )

class DefinitionType(SQLAlchemyObjectType):
    class Meta:
        model = DefinitionModel
        interfaces = (graphene.relay.Node, )
    
    meanings = graphene.List(MeaningType)
    examples = graphene.List(graphene.String)

    def resolve_examples(self, info):
        return [e.example for e in self.examples]

class RelationshipsType(graphene.ObjectType):
    root_word = graphene.String()
    derivatives = graphene.List(graphene.String)
    synonyms = graphene.List(graphene.String)
    antonyms = graphene.List(graphene.String)
    associated_words = graphene.List(graphene.String)
    related_terms = graphene.List(graphene.String)
    hypernyms = graphene.List(graphene.String)
    hyponyms = graphene.List(graphene.String)
    meronyms = graphene.List(graphene.String)
    holonyms = graphene.List(graphene.String)

class FormType(SQLAlchemyObjectType):
    class Meta:
        model = FormModel
        interfaces = (graphene.relay.Node, )

class HeadTemplateType(SQLAlchemyObjectType):
    class Meta:
        model = HeadTemplateModel
        interfaces = (graphene.relay.Node, )

class InflectionType(SQLAlchemyObjectType):
    class Meta:
        model = InflectionModel
        interfaces = (graphene.relay.Node, )

class WordType(SQLAlchemyObjectType):
    class Meta:
        model = WordModel
        interfaces = (graphene.relay.Node, )

    pronunciation = graphene.Field(PronunciationType)
    etymology = graphene.Field(EtymologyType)
    definitions = graphene.List(DefinitionType)
    relationships = graphene.Field(RelationshipsType)
    forms = graphene.List(FormType)
    languages = graphene.List(graphene.String)
    head_templates = graphene.List(HeadTemplateType)
    inflections = graphene.List(InflectionType)
    alternate_forms = graphene.List(graphene.String)

    def resolve_pronunciation(self, info):
        return PronunciationType(
            text=self.pronunciation,
            ipa=", ".join([p for p in self.audio_pronunciation or [] if p.startswith("/")]),
            audio=[p for p in self.audio_pronunciation or [] if p and not p.startswith("/")]
        )

    def resolve_relationships(self, info):
        return RelationshipsType(
            root_word=self.root_word,
            derivatives=[d.derivative for d in self.derivatives],
            synonyms=[s.word for s in self.synonyms],
            antonyms=[a.word for a in self.antonyms],
            associated_words=[aw.associated_word for aw in self.associated_words],
            related_terms=[rt.word for rt in self.related_terms],
            hypernyms=[h.hypernym for h in self.hypernyms],
            hyponyms=[h.hyponym for h in self.hyponyms],
            meronyms=[m.meronym for m in self.meronyms],
            holonyms=[h.holonym for h in self.holonyms]
        )

    def resolve_languages(self, info):
        return [l.code for l in self.languages]

    def resolve_alternate_forms(self, info):
        return [af.alternate_form for af in self.alternate_forms]

class NetworkWordInfoType(graphene.ObjectType):
    word = graphene.String()
    definition = graphene.String()
    derivatives = graphene.List(graphene.String)
    root_word = graphene.String()
    synonyms = graphene.List(graphene.String)
    antonyms = graphene.List(graphene.String)
    associated_words = graphene.List(graphene.String)
    etymology = graphene.Field(EtymologyType)

class NodeType(graphene.ObjectType):
    id = graphene.String()
    group = graphene.String()
    info = graphene.Field(NetworkWordInfoType)

class LinkType(graphene.ObjectType):
    source = graphene.String()
    target = graphene.String()
    type = graphene.String()

class WordNetworkType(graphene.ObjectType):
    nodes = graphene.List(NodeType)
    links = graphene.List(LinkType)

class Query(graphene.ObjectType):
    node = graphene.relay.Node.Field()
    all_words = SQLAlchemyConnectionField(WordType)
    get_word = graphene.Field(WordType, word=graphene.String(required=True))
    get_word_network = graphene.Field(WordNetworkType, word=graphene.String(required=True), depth=graphene.Int(), breadth=graphene.Int())
    search_words = graphene.List(WordType, query=graphene.String())

    def resolve_get_word(self, info, word):
        return WordModel.query.filter(func.lower(func.unaccent(WordModel.word)) == func.lower(func.unaccent(word))).first()

    def resolve_get_word_network(self, info, word, depth=2, breadth=10):
        from word_network import get_related_words
        network = get_related_words(word, depth, breadth)
        return WordNetworkType(
            nodes=[NodeType(
                id=node['id'],
                group=node['group'],
                info=NetworkWordInfoType(**node['info'])
            ) for node in network['nodes']],
            links=[LinkType(**link) for link in network['links']]
        )

    def resolve_search_words(self, info, query):
        return WordModel.query.filter(func.lower(func.unaccent(WordModel.word)).contains(func.lower(func.unaccent(query)))).all()

schema = graphene.Schema(query=Query)