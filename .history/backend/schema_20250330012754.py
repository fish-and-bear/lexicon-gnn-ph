import graphene
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from sqlalchemy import func
from models import (
    Word as WordModel,
    Definition as DefinitionModel,
    PartsOfSpeech as PartsOfSpeechModel,
    Etymology as EtymologyModel,
    Relation as RelationModel,
    Affixation as AffixationModel,
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
        if hasattr(self, 'definition_relations'):
            return [relation.word.lemma for relation in self.definition_relations]
        return []

class AffixationType(SQLAlchemyObjectType):
    class Meta:
        model = AffixationModel
        interfaces = (graphene.relay.Node, )
        
    root_word = graphene.String()
    affixed_word = graphene.String()
    
    def resolve_root_word(self, info):
        return self.root_word.lemma if self.root_word else None
        
    def resolve_affixed_word(self, info):
        return self.affixed_word.lemma if self.affixed_word else None

class RelationshipsType(graphene.ObjectType):
    root_word = graphene.String()
    synonyms = graphene.List(graphene.String)
    antonyms = graphene.List(graphene.String)
    related = graphene.List(graphene.String)
    derived_from = graphene.List(graphene.String)
    derived_forms = graphene.List(graphene.String)
    affixations = graphene.Field(lambda: AffixationsType)

class AffixationsType(graphene.ObjectType):
    as_root = graphene.List(graphene.String)
    as_affixed = graphene.List(graphene.String)

class IdiomType(graphene.ObjectType):
    idiom = graphene.String()
    meaning = graphene.String()
    examples = graphene.List(graphene.String)

class BadlitType(graphene.ObjectType):
    """Type for Badlit script information."""
    badlit_form = graphene.String()
    hyphenation = graphene.String()

class WordMetadataType(graphene.ObjectType):
    """Type for word metadata."""
    is_proper_noun = graphene.Boolean()
    is_abbreviation = graphene.Boolean()
    is_initialism = graphene.Boolean()
    source_info = graphene.JSONString()

class WordType(SQLAlchemyObjectType):
    class Meta:
        model = WordModel
        interfaces = (graphene.relay.Node, )

    pronunciation = graphene.Field(PronunciationType)
    etymology = graphene.Field(EtymologyType)
    etymologies = graphene.List(EtymologyType)
    definitions = graphene.List(DefinitionType)
    relationships = graphene.Field(RelationshipsType)
    has_baybayin = graphene.Boolean()
    baybayin_form = graphene.String()
    romanized_form = graphene.String()
    badlit = graphene.Field(BadlitType)  # New field
    metadata = graphene.Field(WordMetadataType)  # New field
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
        
    def resolve_etymologies(self, info):
        return self.etymologies if hasattr(self, 'etymologies') else []

    def resolve_relationships(self, info):
        # Group relations by type
        synonyms = []
        antonyms = []
        related = []
        derived_from = []
        derived_forms = []

        # Process relations from this word to others
        for relation in self.relations_from:
            rel_type = relation.relation_type.lower()
            rel_word = relation.to_word.lemma
            
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

        # Process relations from others to this word
        for relation in self.relations_to:
            rel_type = relation.relation_type.lower()
            if rel_type == 'derived_from' or rel_type == 'root_of':
                derived_forms.append(relation.from_word.lemma)

        # Process affixations
        affixations_as_root = [aff.affixed_word.lemma for aff in self.affixations_as_root] if hasattr(self, 'affixations_as_root') else []
        affixations_as_affixed = [aff.root_word.lemma for aff in self.affixations_as_affixed] if hasattr(self, 'affixations_as_affixed') else []
        
        return RelationshipsType(
            root_word=self.root_word.lemma if self.root_word else None,
            synonyms=synonyms,
            antonyms=antonyms,
            related=related,
            derived_from=derived_from,
            derived_forms=derived_forms,
            affixations=AffixationsType(
                as_root=affixations_as_root,
                as_affixed=affixations_as_affixed
            )
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

    def resolve_badlit(self, info):
        """Resolve Badlit script information."""
        return BadlitType(
            badlit_form=self.badlit_form,
            hyphenation=self.hyphenation
        ) if self.badlit_form else None

    def resolve_metadata(self, info):
        """Resolve word metadata."""
        return WordMetadataType(
            is_proper_noun=self.is_proper_noun,
            is_abbreviation=self.is_abbreviation,
            is_initialism=self.is_initialism,
            source_info=self.source_info
        )

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
    parts_of_speech = graphene.List(PartsOfSpeechType)
    etymology_components = graphene.List(graphene.String, word=graphene.String(required=True))

    def resolve_get_word(self, info, word):
        return WordModel.query.filter(func.lower(func.unaccent(WordModel.lemma)) == func.lower(func.unaccent(word))).first()

    def resolve_get_word_network(self, info, word, depth=2, breadth=10):
        # Import here to avoid circular imports
        try:
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
        except ImportError:
            # Fallback if word_network module is not available
            word_obj = WordModel.query.filter(func.lower(func.unaccent(WordModel.lemma)) == 
                                           func.lower(func.unaccent(word))).first()
            if not word_obj:
                return WordNetworkType(nodes=[], links=[])
                
            # Build a simple network
            nodes = [NodeType(
                id=str(word_obj.id), 
                group="root",
                info=NetworkWordInfoType(
                    word=word_obj.lemma,
                    has_baybayin=word_obj.has_baybayin,
                    baybayin=word_obj.baybayin_form,
                    definition=word_obj.definitions[0].definition_text if word_obj.definitions else "",
                    synonyms=[],
                    antonyms=[],
                    derivatives=[],
                    root_word=None
                )
            )]
            links = []
            
            return WordNetworkType(nodes=nodes, links=links)

    def resolve_search_words(self, info, query):
        # Use tsquery for better text search
        return WordModel.query.filter(
            WordModel.search_text.op('@@')(func.plainto_tsquery('simple', query))
        ).order_by(
            func.similarity(WordModel.lemma, query).desc()
        ).limit(20).all()

    def resolve_baybayin_words(self, info, limit=10):
        return WordModel.query.filter(WordModel.has_baybayin == True).limit(limit).all()
        
    def resolve_parts_of_speech(self, info):
        return PartsOfSpeechModel.query.all()
        
    def resolve_etymology_components(self, info, word):
        word_obj = WordModel.query.filter(func.lower(func.unaccent(WordModel.lemma)) == 
                                        func.lower(func.unaccent(word))).first()
        if not word_obj or not word_obj.etymologies:
            return []
            
        components = []
        for etymology in word_obj.etymologies:
            if etymology.normalized_components:
                try:
                    import json
                    comps = json.loads(etymology.normalized_components)
                    if isinstance(comps, list):
                        components.extend(comps)
                    else:
                        components.extend(etymology.normalized_components.split(';'))
                except:
                    components.extend(etymology.normalized_components.split(';'))
                    
        return list(set(components))  # Return unique components

schema = graphene.Schema(query=Query)