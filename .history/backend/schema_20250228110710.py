import graphene
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from sqlalchemy import func
from models import (
    Word as WordModel,
    Definition as DefinitionModel,
    PartsOfSpeech as PartsOfSpeechModel,
    Etymology as EtymologyModel,
    Relation as RelationModel,
    Form as FormModel,
    Language as LanguageModel,
    Affixation as AffixationModel,
    Example as ExampleModel,
    DefinitionRelation as DefinitionRelationModel,
)
from database import db_session

class PronunciationType(graphene.ObjectType):
    ipa = graphene.String()
    audio = graphene.List(graphene.String)
    hyphenation = graphene.String()
    sounds = graphene.List(graphene.JSONString)

class RelationType(SQLAlchemyObjectType):
    class Meta:
        model = RelationModel
        interfaces = (graphene.relay.Node, )

class EtymologyType(SQLAlchemyObjectType):
    class Meta:
        model = EtymologyModel
        interfaces = (graphene.relay.Node, )
    
    components = graphene.List(graphene.String)
    language_codes = graphene.List(graphene.String)

    def resolve_components(self, info):
        if self.normalized_components:
            import json
            try:
                return json.loads(self.normalized_components)
            except json.JSONDecodeError:
                return self.normalized_components.split(';') if self.normalized_components else []
        return []

    def resolve_language_codes(self, info):
        return self.language_codes.split(',') if self.language_codes else []

class DefinitionRelationType(SQLAlchemyObjectType):
    class Meta:
        model = DefinitionRelationModel
        interfaces = (graphene.relay.Node, )

class PartsOfSpeechType(SQLAlchemyObjectType):
    class Meta:
        model = PartsOfSpeechModel
        interfaces = (graphene.relay.Node, )

class DefinitionType(SQLAlchemyObjectType):
    class Meta:
        model = DefinitionModel
        interfaces = (graphene.relay.Node, )
    
    part_of_speech = graphene.Field(PartsOfSpeechType)
    examples = graphene.List(graphene.String)
    usage_notes = graphene.List(graphene.String)
    related_words = graphene.List(graphene.String)
    sources = graphene.List(graphene.String)

    def resolve_part_of_speech(self, info):
        return self.standardized_pos

    def resolve_examples(self, info):
        if self.examples:
            import json
            try:
                examples = json.loads(self.examples)
                if isinstance(examples, list):
                    return examples
                return [str(examples)]
            except json.JSONDecodeError:
                return [line.strip() for line in self.examples.split('\n') if line.strip()]
        return []

    def resolve_usage_notes(self, info):
        if self.usage_notes:
            import json
            try:
                notes = json.loads(self.usage_notes)
                if isinstance(notes, list):
                    return notes
                return [str(notes)]
            except json.JSONDecodeError:
                return [line.strip() for line in self.usage_notes.split('\n') if line.strip()]
        return []

    def resolve_sources(self, info):
        return self.sources.split(', ') if self.sources else []

    def resolve_related_words(self, info):
        return [relation.word.word for relation in self.word_relations]

class RelationshipsType(graphene.ObjectType):
    root_word = graphene.String()
    synonyms = graphene.List(graphene.String)
    antonyms = graphene.List(graphene.String)
    related = graphene.List(graphene.String)
    derived_from = graphene.List(graphene.String)
    derived_forms = graphene.List(graphene.String)

class FormType(SQLAlchemyObjectType):
    class Meta:
        model = FormModel
        interfaces = (graphene.relay.Node, )

class AffixationType(SQLAlchemyObjectType):
    class Meta:
        model = AffixationModel
        interfaces = (graphene.relay.Node, )

class LanguageType(SQLAlchemyObjectType):
    class Meta:
        model = LanguageModel
        interfaces = (graphene.relay.Node, )

class IdiomType(graphene.ObjectType):
    idiom = graphene.String()
    meaning = graphene.String()
    examples = graphene.List(graphene.String)

class WordType(SQLAlchemyObjectType):
    class Meta:
        model = WordModel
        interfaces = (graphene.relay.Node, )

    pronunciation = graphene.Field(PronunciationType)
    etymology = graphene.Field(EtymologyType)
    definitions = graphene.List(DefinitionType)
    relationships = graphene.Field(RelationshipsType)
    has_baybayin = graphene.Boolean()
    baybayin_form = graphene.String()
    romanized_form = graphene.String()
    preferred_spelling = graphene.String()
    language_code = graphene.String()
    tags = graphene.List(graphene.String)
    idioms = graphene.List(IdiomType)

    def resolve_pronunciation(self, info):
        if self.pronunciation_data:
            import json
            try:
                data = json.loads(self.pronunciation_data) if isinstance(self.pronunciation_data, str) else self.pronunciation_data
                return PronunciationType(
                    ipa=data.get('ipa'),
                    audio=data.get('audio', []),
                    hyphenation=data.get('hyphenation'),
                    sounds=data.get('sounds', [])
                )
            except (json.JSONDecodeError, AttributeError):
                return None
        return None

    def resolve_etymology(self, info):
        return self.etymologies[0] if self.etymologies else None

    def resolve_relationships(self, info):
        from_relations = {r.relation_type: r.to_word.word for r in self.from_relations}
        to_relations = {r.relation_type: r.from_word.word for r in self.to_relations}

        # Group relations by type
        synonyms = []
        antonyms = []
        related = []
        derived_from = []
        derived_forms = []

        for relation in self.from_relations:
            rel_type = relation.relation_type.lower()
            rel_word = relation.to_word.word
            
            if rel_type == 'synonym':
                synonyms.append(rel_word)
            elif rel_type == 'antonym':
                antonyms.append(rel_word)
            elif rel_type == 'related':
                related.append(rel_word)
            elif rel_type in ('derived_from', 'root_of'):
                derived_from.append(rel_word)
            elif rel_type in ('derivative', 'derived_form'):
                derived_forms.append(rel_word)

        return RelationshipsType(
            root_word=self.root_word.word if self.root_word else None,
            synonyms=synonyms,
            antonyms=antonyms,
            related=related,
            derived_from=derived_from,
            derived_forms=derived_forms
        )

    def resolve_tags(self, info):
        return self.tags.split(',') if self.tags else []

    def resolve_idioms(self, info):
        if not self.idioms or self.idioms == '[]':
            return []
        
        import json
        try:
            idioms_data = json.loads(self.idioms) if isinstance(self.idioms, str) else self.idioms
            return [
                IdiomType(
                    idiom=idiom.get('idiom', '') or idiom.get('text', ''),
                    meaning=idiom.get('meaning', ''),
                    examples=idiom.get('examples', [])
                )
                for idiom in idioms_data
                if isinstance(idiom, dict) and (idiom.get('idiom') or idiom.get('text')) and idiom.get('meaning')
            ]
        except (json.JSONDecodeError, AttributeError):
            return []

class NetworkWordInfoType(graphene.ObjectType):
    word = graphene.String()
    definition = graphene.String()
    derivatives = graphene.List(graphene.String)
    root_word = graphene.String()
    synonyms = graphene.List(graphene.String)
    antonyms = graphene.List(graphene.String)
    baybayin = graphene.String()
    has_baybayin = graphene.Boolean()

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
    baybayin_words = graphene.List(WordType, limit=graphene.Int())

    def resolve_get_word(self, info, word):
        return WordModel.query.filter(func.lower(func.unaccent(WordModel.lemma)) == func.lower(func.unaccent(word))).first()

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
        # Use tsquery for better text search
        return WordModel.query.filter(
            WordModel.search_text.op('@@')(func.plainto_tsquery('simple', query))
        ).order_by(
            func.similarity(WordModel.lemma, query).desc()
        ).limit(20).all()

    def resolve_baybayin_words(self, info, limit=10):
        return WordModel.query.filter(WordModel.has_baybayin == True).limit(limit).all()

schema = graphene.Schema(query=Query)