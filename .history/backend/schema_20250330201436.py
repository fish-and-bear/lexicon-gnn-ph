import graphene
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from sqlalchemy import func, or_, and_, not_, case
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
import json
from typing import List, Dict, Any, Optional
from marshmallow import Schema, fields, validate
import re

class MetadataType(graphene.ObjectType):
    """Type for generic metadata fields."""
    strength = graphene.Float()
    tags = graphene.List(graphene.String)
    english_equivalent = graphene.String()
    notes = graphene.String()
    source_details = graphene.JSONString()

class PronunciationType(graphene.ObjectType):
    """Type for pronunciation data."""
    ipa = graphene.String(description="International Phonetic Alphabet representation")
    audio = graphene.List(graphene.String, description="List of audio file URLs")
    hyphenation = graphene.String(description="Syllable separation")
    sounds = graphene.List(graphene.JSONString, description="Detailed sound components")
    stress_pattern = graphene.String(description="Word stress pattern")
    phonemes = graphene.List(graphene.String, description="List of phonemes")
    variants = graphene.List(graphene.String, description="Pronunciation variants")

class RelationType(SQLAlchemyObjectType):
    """Type for word relationships."""
    class Meta:
        model = RelationModel
        interfaces = (graphene.relay.Node, )
    
    metadata = graphene.Field(MetadataType)
    strength = graphene.Float()
    tags = graphene.List(graphene.String)
    source_details = graphene.JSONString()
    
    def resolve_metadata(self, info):
        if not self.metadata:
            return None
        return MetadataType(**self.metadata)
    
    def resolve_strength(self, info):
        return self.metadata.get('strength') if self.metadata else None
    
    def resolve_tags(self, info):
        return self.metadata.get('tags', []) if self.metadata else []
    
    def resolve_source_details(self, info):
        return self.metadata.get('source_details') if self.metadata else None

class EtymologyComponentType(graphene.ObjectType):
    """Type for etymology components."""
    text = graphene.String()
    language = graphene.String()
    meaning = graphene.String()
    notes = graphene.String()
    confidence = graphene.Float()

class EtymologyType(SQLAlchemyObjectType):
    """Type for word etymologies."""
    class Meta:
        model = EtymologyModel
        interfaces = (graphene.relay.Node, )
    
    components = graphene.List(EtymologyComponentType)
    language_codes = graphene.List(graphene.String)
    structure = graphene.JSONString()
    confidence = graphene.Float()
    
    def resolve_components(self, info):
        if not self.normalized_components:
            return []
        try:
            components = json.loads(self.normalized_components)
            if isinstance(components, list):
                return [EtymologyComponentType(**comp) if isinstance(comp, dict) else 
                       EtymologyComponentType(text=comp) for comp in components]
            return []
        except json.JSONDecodeError:
            components = self.normalized_components.split(';')
            return [EtymologyComponentType(text=comp.strip()) for comp in components if comp.strip()]
    
    def resolve_language_codes(self, info):
        return self.language_codes.split(',') if self.language_codes else []
    
    def resolve_structure(self, info):
        if not self.etymology_structure:
            return None
        try:
            return json.loads(self.etymology_structure)
        except json.JSONDecodeError:
            return None
    
    def resolve_confidence(self, info):
        if not self.etymology_structure:
            return None
        try:
            structure = json.loads(self.etymology_structure)
            return structure.get('confidence')
        except json.JSONDecodeError:
            return None

class DefinitionRelationType(SQLAlchemyObjectType):
    """Type for definition-specific relationships."""
    class Meta:
        model = DefinitionRelationModel
        interfaces = (graphene.relay.Node, )
    
    metadata = graphene.Field(MetadataType)
    
    def resolve_metadata(self, info):
        if not hasattr(self, 'metadata') or not self.metadata:
            return None
        return MetadataType(**self.metadata)

class PartsOfSpeechType(SQLAlchemyObjectType):
    """Type for parts of speech."""
    class Meta:
        model = PartsOfSpeechModel
        interfaces = (graphene.relay.Node, )
    
    word_count = graphene.Int()
    
    def resolve_word_count(self, info):
        return DefinitionModel.query.filter_by(standardized_pos_id=self.id).count()

class ExampleType(graphene.ObjectType):
    """Type for usage examples."""
    text = graphene.String()
    translation = graphene.String()
    notes = graphene.String()
    source = graphene.String()
    tags = graphene.List(graphene.String)

class DefinitionType(SQLAlchemyObjectType):
    """Type for word definitions."""
    class Meta:
        model = DefinitionModel
        interfaces = (graphene.relay.Node, )
    
    part_of_speech = graphene.Field(PartsOfSpeechType)
    examples = graphene.List(ExampleType)
    usage_notes = graphene.List(graphene.String)
    related_words = graphene.List(graphene.String)
    sources = graphene.List(graphene.String)
    tags = graphene.List(graphene.String)
    metadata = graphene.Field(MetadataType)
    
    def resolve_part_of_speech(self, info):
        return self.standardized_pos
    
    def resolve_examples(self, info):
        if not self.examples:
            return []
        try:
            examples = json.loads(self.examples)
            if isinstance(examples, list):
                return [ExampleType(**ex) if isinstance(ex, dict) else 
                       ExampleType(text=str(ex)) for ex in examples]
            return [ExampleType(text=str(examples))]
        except json.JSONDecodeError:
            return [ExampleType(text=line.strip()) 
                   for line in self.examples.split('\n') if line.strip()]
    
    def resolve_usage_notes(self, info):
        if not self.usage_notes:
            return []
        try:
            notes = json.loads(self.usage_notes)
            return notes if isinstance(notes, list) else [str(notes)]
        except json.JSONDecodeError:
            return [line.strip() for line in self.usage_notes.split('\n') if line.strip()]
    
    def resolve_sources(self, info):
        return self.sources.split(', ') if self.sources else []
    
    def resolve_related_words(self, info):
        if hasattr(self, 'definition_relations'):
            return [relation.word.lemma for relation in self.definition_relations]
        return []
    
    def resolve_tags(self, info):
        if not self.tags:
            return []
        try:
            return json.loads(self.tags) if isinstance(self.tags, str) else self.tags
        except json.JSONDecodeError:
            return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
    def resolve_metadata(self, info):
        if not hasattr(self, 'metadata') or not self.metadata:
            return None
        return MetadataType(**self.metadata)

class AffixationType(SQLAlchemyObjectType):
    """Type for word affixations."""
    class Meta:
        model = AffixationModel
        interfaces = (graphene.relay.Node, )
    
    root_word = graphene.String()
    affixed_word = graphene.String()
    metadata = graphene.Field(MetadataType)
    examples = graphene.List(ExampleType)
    
    def resolve_root_word(self, info):
        return self.root_word.lemma if self.root_word else None
    
    def resolve_affixed_word(self, info):
        return self.affixed_word.lemma if self.affixed_word else None
    
    def resolve_metadata(self, info):
        if not hasattr(self, 'metadata') or not self.metadata:
            return None
        return MetadataType(**self.metadata)
    
    def resolve_examples(self, info):
        if not hasattr(self, 'examples') or not self.examples:
            return []
        try:
            examples = json.loads(self.examples)
            return [ExampleType(**ex) if isinstance(ex, dict) else 
                   ExampleType(text=str(ex)) for ex in examples]
        except (json.JSONDecodeError, AttributeError):
            return []

class RelationshipsType(graphene.ObjectType):
    """Type for word relationships summary."""
    root_word = graphene.String()
    synonyms = graphene.List(graphene.String)
    antonyms = graphene.List(graphene.String)
    related = graphene.List(graphene.String)
    derived_from = graphene.List(graphene.String)
    derived_forms = graphene.List(graphene.String)
    affixations = graphene.Field(lambda: AffixationsType)
    variants = graphene.List(graphene.String)
    hypernyms = graphene.List(graphene.String)
    hyponyms = graphene.List(graphene.String)
    metadata = graphene.Field(MetadataType)

class AffixationsType(graphene.ObjectType):
    """Type for word affixation summary."""
    as_root = graphene.List(graphene.String)
    as_affixed = graphene.List(graphene.String)
    metadata = graphene.Field(MetadataType)

class IdiomType(graphene.ObjectType):
    """Type for idiomatic expressions."""
    idiom = graphene.String()
    meaning = graphene.String()
    examples = graphene.List(ExampleType)
    notes = graphene.String()
    tags = graphene.List(graphene.String)
    source = graphene.String()

class QualityMetricsType(graphene.ObjectType):
    """Type for word quality metrics."""
    total_score = graphene.Int()
    completeness = graphene.Float()
    accuracy = graphene.Float()
    richness = graphene.Float()
    has_baybayin = graphene.Boolean()
    has_pronunciation = graphene.Boolean()
    has_etymology = graphene.Boolean()
    has_examples = graphene.Boolean()
    has_relations = graphene.Boolean()
    definition_count = graphene.Int()
    relation_count = graphene.Int()
    example_count = graphene.Int()

class WordType(SQLAlchemyObjectType):
    """Type for dictionary words."""
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
    preferred_spelling = graphene.String()
    language_code = graphene.String()
    tags = graphene.List(graphene.String)
    idioms = graphene.List(IdiomType)
    quality_metrics = graphene.Field(QualityMetricsType)
    metadata = graphene.Field(MetadataType)
    source_info = graphene.JSONString()
    
    def resolve_pronunciation(self, info):
        if not self.pronunciation_data:
            return None
        try:
            data = json.loads(self.pronunciation_data) if isinstance(self.pronunciation_data, str) else self.pronunciation_data
            return PronunciationType(
                ipa=data.get('ipa'),
                audio=data.get('audio', []),
                hyphenation=data.get('hyphenation'),
                sounds=data.get('sounds', []),
                stress_pattern=data.get('stress_pattern'),
                phonemes=data.get('phonemes', []),
                variants=data.get('variants', [])
            )
        except (json.JSONDecodeError, AttributeError):
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
        variants = []
        hypernyms = []
        hyponyms = []
        
        # Process outgoing relations
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
            elif rel_type == 'variant':
                variants.append(rel_word)
            elif rel_type == 'hypernym_of':
                hyponyms.append(rel_word)
            elif rel_type == 'hyponym_of':
                hypernyms.append(rel_word)
        
        # Process incoming relations
        for relation in self.relations_to:
            rel_type = relation.relation_type.lower()
            if rel_type == 'derived_from' or rel_type == 'root_of':
                derived_forms.append(relation.from_word.lemma)
        
        # Process affixations
        affixations_as_root = [aff.affixed_word.lemma for aff in self.affixations_as_root] if hasattr(self, 'affixations_as_root') else []
        affixations_as_affixed = [aff.root_word.lemma for aff in self.affixations_as_affixed] if hasattr(self, 'affixations_as_affixed') else []
        
        return RelationshipsType(
            root_word=self.root_word.lemma if self.root_word else None,
            synonyms=list(set(synonyms)),
            antonyms=list(set(antonyms)),
            related=list(set(related)),
            derived_from=list(set(derived_from)),
            derived_forms=list(set(derived_forms)),
            variants=list(set(variants)),
            hypernyms=list(set(hypernyms)),
            hyponyms=list(set(hyponyms)),
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
        
        try:
            idioms_data = json.loads(self.idioms) if isinstance(self.idioms, str) else self.idioms
            return [
                IdiomType(
                    idiom=idiom.get('idiom', '') or idiom.get('text', ''),
                    meaning=idiom.get('meaning', ''),
                    examples=[ExampleType(**ex) if isinstance(ex, dict) else ExampleType(text=ex) 
                             for ex in idiom.get('examples', [])],
                    notes=idiom.get('notes'),
                    tags=idiom.get('tags', []),
                    source=idiom.get('source')
                )
                for idiom in idioms_data
                if isinstance(idiom, dict) and (idiom.get('idiom') or idiom.get('text')) and idiom.get('meaning')
            ]
        except (json.JSONDecodeError, AttributeError):
            return []
    
    def resolve_quality_metrics(self, info):
        if not hasattr(self, 'calculate_data_quality_score'):
            return None
            
        total_score = self.calculate_data_quality_score()
        definition_count = len(self.definitions) if self.definitions else 0
        relation_count = (
            len(self.relations_from) if hasattr(self, 'relations_from') else 0
        ) + (
            len(self.relations_to) if hasattr(self, 'relations_to') else 0
        )
        example_count = sum(
            len(d.get_examples_list()) for d in self.definitions
            if hasattr(d, 'get_examples_list')
        ) if self.definitions else 0
        
        return QualityMetricsType(
            total_score=total_score,
            completeness=total_score / 100.0,
            accuracy=0.8,  # Default value, could be calculated based on verification status
            richness=min(1.0, (definition_count + relation_count + example_count) / 20.0),
            has_baybayin=self.has_baybayin,
            has_pronunciation=bool(self.pronunciation_data),
            has_etymology=bool(self.etymologies),
            has_examples=example_count > 0,
            has_relations=relation_count > 0,
            definition_count=definition_count,
            relation_count=relation_count,
            example_count=example_count
        )
    
    def resolve_metadata(self, info):
        if not hasattr(self, 'metadata') or not self.metadata:
            return None
        return MetadataType(**self.metadata)
    
    def resolve_source_info(self, info):
        return self.source_info if self.source_info else None

class NetworkWordInfoType(graphene.ObjectType):
    """Type for word network information."""
    word = graphene.String()
    definition = graphene.String()
    derivatives = graphene.List(graphene.String)
    root_word = graphene.String()
    synonyms = graphene.List(graphene.String)
    antonyms = graphene.List(graphene.String)
    baybayin = graphene.String()
    has_baybayin = graphene.Boolean()
    quality_score = graphene.Int()
    metadata = graphene.Field(MetadataType)

class NodeType(graphene.ObjectType):
    """Type for network nodes."""
    id = graphene.String()
    group = graphene.String()
    info = graphene.Field(NetworkWordInfoType)
    metadata = graphene.Field(MetadataType)

class LinkType(graphene.ObjectType):
    """Type for network links."""
    source = graphene.String()
    target = graphene.String()
    type = graphene.String()
    metadata = graphene.Field(MetadataType)

class WordNetworkType(graphene.ObjectType):
    """Type for word relationship networks."""
    nodes = graphene.List(NodeType)
    links = graphene.List(LinkType)
    metadata = graphene.Field(MetadataType)

class SearchQuerySchema(Schema):
    """Schema for search query parameters."""
    q = fields.Str(required=True, validate=validate.Length(min=1))
    limit = fields.Int(validate=validate.Range(min=1, max=50), default=10)
    pos = fields.Str(validate=validate.OneOf(['n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix']))
    language = fields.Str()  # Accept any valid language code
    include_baybayin = fields.Bool(default=True)
    min_similarity = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.3)
    mode = fields.Str(validate=validate.OneOf(['all', 'exact', 'phonetic', 'baybayin']), default='all')
    sort = fields.Str(validate=validate.OneOf(['relevance', 'alphabetical', 'created', 'updated']), default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), default='desc')

class WordQuerySchema(Schema):
    """Schema for word query parameters."""
    language_code = fields.Str()  # Accept any valid language code
    include_definitions = fields.Bool(default=True)
    include_relations = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)

class WordSchema(Schema):
    """Schema for word entries."""
    id = fields.Int(dump_only=True)
    lemma = fields.Str(required=True)
    normalized_lemma = fields.Str()
    language_code = fields.Str(required=True)  # Accept any valid language code
    has_baybayin = fields.Bool()
    baybayin_form = fields.Str()
    romanized_form = fields.Str()
    is_root = fields.Bool()
    root_word = fields.Nested('self', only=('id', 'lemma', 'language_code'))
    preferred_spelling = fields.Str()
    alternative_spellings = fields.List(fields.Str())
    syllable_count = fields.Int()
    pronunciation_guide = fields.Str()
    stress_pattern = fields.Str()
    formality_level = fields.Str()
    usage_frequency = fields.Float()
    geographic_region = fields.Str()
    time_period = fields.Str()
    cultural_notes = fields.Str()
    grammatical_categories = fields.List(fields.Str())
    semantic_domains = fields.List(fields.Str())
    etymology_confidence = fields.Float()
    data_quality_score = fields.Float()
    pronunciation_data = fields.Dict()
    tags = fields.List(fields.Str())
    idioms = fields.List(fields.Dict())
    source_info = fields.Dict()
    meta_info = fields.Dict()  # Changed from metadata
    verification_status = fields.Str(validate=validate.OneOf([
        'unverified', 'verified', 'needs_review', 'disputed'
    ]))
    verification_notes = fields.Str()
    last_verified_at = fields.DateTime()
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    etymology = fields.Nested(EtymologySchema)
    definitions = fields.List(fields.Nested(DefinitionSchema))
    pronunciations = fields.List(fields.Nested(PronunciationSchema))
    relations = fields.List(fields.Nested(RelationSchema))
    affixations = fields.Dict(keys=fields.Str(), values=fields.List(fields.Nested(AffixationSchema)))

class Query(graphene.ObjectType):
    """Root query type."""
    node = graphene.relay.Node.Field()
    all_words = SQLAlchemyConnectionField(WordType)
    get_word = graphene.Field(
        WordType,
        word=graphene.String(required=True),
        language=graphene.String()
    )
    get_word_network = graphene.Field(
        WordNetworkType,
        word=graphene.String(required=True),
        depth=graphene.Int(),
        breadth=graphene.Int(),
        include_etymology=graphene.Boolean(),
        include_affixes=graphene.Boolean()
    )
    search_words = graphene.List(
        WordType,
        query=graphene.String(),
        language=graphene.String(),
        pos=graphene.String(),
        has_baybayin=graphene.Boolean(),
        has_etymology=graphene.Boolean(),
        min_quality=graphene.Int(),
        limit=graphene.Int()
    )
    baybayin_words = graphene.List(
        WordType,
        limit=graphene.Int(),
        min_quality=graphene.Int(),
        has_etymology=graphene.Boolean()
    )
    parts_of_speech = graphene.List(PartsOfSpeechType)
    etymology_components = graphene.List(
        graphene.String,
        word=graphene.String(required=True),
        language=graphene.String()
    )
    
    def resolve_get_word(self, info, word, language=None):
        """Get word by lemma, removing any trailing numbers."""
        # Remove trailing numbers from the word
        clean_word = re.sub(r'\d+$', '', word)
        query = WordModel.query.filter(
            func.lower(func.unaccent(WordModel.lemma)) == func.lower(func.unaccent(clean_word))
        )
        if language:
            query = query.filter(WordModel.language_code == language)
        return query.first()
    
    def resolve_get_word_network(self, info, word, depth=2, breadth=10,
                               include_etymology=True, include_affixes=True):
        try:
            from word_network import get_related_words
            network = get_related_words(
                word, depth, breadth,
                include_etymology=include_etymology,
                include_affixes=include_affixes
            )
            return WordNetworkType(
                nodes=[NodeType(
                    id=node['id'],
                    group=node['group'],
                    info=NetworkWordInfoType(**node['info']),
                    metadata=MetadataType(**node.get('metadata', {})) if node.get('metadata') else None
                ) for node in network['nodes']],
                links=[LinkType(
                    source=link['source'],
                    target=link['target'],
                    type=link['type'],
                    metadata=MetadataType(**link.get('metadata', {})) if link.get('metadata') else None
                ) for link in network['links']],
                metadata=MetadataType(**network.get('metadata', {})) if network.get('metadata') else None
            )
        except ImportError:
            word_obj = WordModel.query.filter(
                func.lower(func.unaccent(WordModel.lemma)) == 
                func.lower(func.unaccent(word))
            ).first()
            
            if not word_obj:
                return WordNetworkType(nodes=[], links=[])
            
            # Build simple network
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
                    root_word=None,
                    quality_score=word_obj.calculate_data_quality_score()
                )
            )]
            
            return WordNetworkType(nodes=nodes, links=[])
    
    def resolve_search_words(self, info, query=None, language=None, pos=None,
                           has_baybayin=None, has_etymology=None,
                           min_quality=None, limit=20):
        """Search words, handling numeric suffixes appropriately."""
        base_query = WordModel.query
        
        if query:
            # Remove trailing numbers from search query
            clean_query = re.sub(r'\d+$', '', query)
            base_query = base_query.filter(
                WordModel.search_text.op('@@')(func.plainto_tsquery('simple', clean_query))
            )
        
        if language:
            base_query = base_query.filter(WordModel.language_code == language)
        
        if pos:
            base_query = base_query.join(WordModel.definitions).join(DefinitionModel.standardized_pos).filter(
                PartsOfSpeechModel.code == pos
            )
        
        if has_baybayin is not None:
            base_query = base_query.filter(WordModel.has_baybayin == has_baybayin)
        
        if has_etymology is not None:
            if has_etymology:
                base_query = base_query.join(WordModel.etymologies)
            else:
                base_query = base_query.outerjoin(WordModel.etymologies).filter(
                    EtymologyModel.id.is_(None)
                )
        
        if min_quality is not None:
            base_query = base_query.filter(
                WordModel.calculate_data_quality_score() >= min_quality
            )
        
        return base_query.order_by(
            func.similarity(WordModel.lemma, query).desc() if query else WordModel.lemma
        ).limit(limit).all()
    
    def resolve_baybayin_words(self, info, limit=10, min_quality=None,
                             has_etymology=None):
        query = WordModel.query.filter(WordModel.has_baybayin == True)
        
        if min_quality is not None:
            query = query.filter(
                WordModel.calculate_data_quality_score() >= min_quality
            )
        
        if has_etymology is not None:
            if has_etymology:
                query = query.join(WordModel.etymologies)
            else:
                query = query.outerjoin(WordModel.etymologies).filter(
                    EtymologyModel.id.is_(None)
                )
        
        return query.limit(limit).all()
    
    def resolve_parts_of_speech(self, info):
        return PartsOfSpeechModel.query.all()
    
    def resolve_etymology_components(self, info, word, language=None):
        word_obj = WordModel.query.filter(
            func.lower(func.unaccent(WordModel.lemma)) == 
            func.lower(func.unaccent(word))
        )
        
        if language:
            word_obj = word_obj.filter(WordModel.language_code == language)
            
        word_obj = word_obj.first()
        
        if not word_obj or not word_obj.etymologies:
            return []
        
        components = []
        for etymology in word_obj.etymologies:
            if etymology.normalized_components:
                try:
                    comps = json.loads(etymology.normalized_components)
                    if isinstance(comps, list):
                        components.extend(comps)
                    else:
                        components.extend(etymology.normalized_components.split(';'))
                except:
                    components.extend(etymology.normalized_components.split(';'))
        
        return list(set(components))

schema = graphene.Schema(query=Query)