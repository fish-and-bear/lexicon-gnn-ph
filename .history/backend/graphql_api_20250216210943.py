from ariadne import ObjectType, QueryType, make_executable_schema
from ariadne.asgi import GraphQL
from models import Word, Definition, Meaning, Etymology, EtymologyComponent, Form, HeadTemplate, Inflection, Language
from database import db_session
from sqlalchemy.orm import joinedload
from sqlalchemy import func
from caching import multi_level_cache

type_defs = """
    type Word {
        word: String!
        pronunciation: Pronunciation
        etymology: Etymology
        definitions: [Definition]
        relationships: Relationships
        forms: [Form]
        languages: [String]
        tags: [String]
        headTemplates: [HeadTemplate]
        inflections: [Inflection]
        alternateForms: [String]
    }

    type Pronunciation {
        text: String
        ipa: String
        audio: [String]
    }

    type Etymology {
        kaikki: String
        components: [EtymologyComponent]
        text: [String]
        parsed: [String]
    }

    type EtymologyComponent {
        component: String
        order: Int
    }

    type Definition {
        partOfSpeech: String
        meanings: [Meaning]
        usageNotes: [String]
        examples: [String]
        tags: [String]
    }

    type Meaning {
        definition: String
        source: String
    }

    type Relationships {
        rootWord: String
        derivatives: [String]
        synonyms: [String]
        antonyms: [String]
        associatedWords: [String]
        relatedTerms: [String]
        hypernyms: [String]
        hyponyms: [String]
        meronyms: [String]
        holonyms: [String]
    }

    type Form {
        form: String
        tags: [String]
    }

    type HeadTemplate {
        name: String
        args: JSON
        expansion: String
    }

    type Inflection {
        name: String
        args: JSON
    }

    type Query {
        getWord(word: String!): Word
        searchWords(query: String!, page: Int = 1, perPage: Int = 20): WordSearchResult
        getWordNetwork(word: String!, depth: Int = 2, breadth: Int = 10): WordNetwork
    }

    type WordSearchResult {
        words: [WordSearchItem]
        page: Int
        perPage: Int
        total: Int
    }

    type WordSearchItem {
        word: String!
        id: Int!
    }

    type WordNetwork {
        nodes: [NetworkNode]
        links: [NetworkLink]
    }

    type NetworkNode {
        id: String!
        group: String!
        info: NetworkWordInfo!
    }

    type NetworkLink {
        source: String!
        target: String!
        type: String!
    }

    type NetworkWordInfo {
        word: String!
        definition: String
        derivatives: [String]
        rootWord: String
        synonyms: [String]
        antonyms: [String]
        associatedWords: [String]
        etymology: NetworkEtymology
    }

    type NetworkEtymology {
        parsed: [String]
        text: [String]
    }

    scalar JSON
"""

query = QueryType()
word = ObjectType("Word")

@multi_level_cache
def get_word_with_relations(word):
    return Word.query.options(
        joinedload(Word.definitions).joinedload(Definition.meanings).joinedload(Meaning.source),
        joinedload(Word.etymologies).joinedload(Etymology.components),
        joinedload(Word.forms),
        joinedload(Word.derivatives),
        joinedload(Word.synonyms),
        joinedload(Word.antonyms),
        joinedload(Word.associated_words),
        joinedload(Word.related_terms),
        joinedload(Word.hypernyms),
        joinedload(Word.hyponyms),
        joinedload(Word.meronyms),
        joinedload(Word.holonyms),
        joinedload(Word.languages),
        joinedload(Word.head_templates),
        joinedload(Word.inflections),
        joinedload(Word.alternate_forms),
    ).filter(func.lower(func.unaccent(Word.word)) == word.lower()).first()

@query.field("getWord")
def resolve_get_word(_, info, word):
    word_entry = get_word_with_relations(word)
    if word_entry:
        return {
            "word": word_entry.word,
            "pronunciation": resolve_pronunciation(word_entry, _),
            "etymology": resolve_etymology(word_entry, _),
            "definitions": resolve_definitions(word_entry, _),
            "relationships": resolve_relationships(word_entry, _),
            "forms": resolve_forms(word_entry, _),
            "languages": resolve_languages(word_entry, _),
            "tags": resolve_tags(word_entry, _),
            "headTemplates": resolve_head_templates(word_entry, _),
            "inflections": resolve_inflections(word_entry, _),
            "alternateForms": resolve_alternate_forms(word_entry, _),
        }
    return None


@query.field("searchWords")
def resolve_search_words(_, info, query, page=1, perPage=20, filter=None):
    base_query = Word.query.filter(Word.word.ilike(f"%{query}%"))
    
    if filter == 'is_real_word:true':
        base_query = base_query.filter(Word.is_real_word == True)
    
    total = base_query.count()
    words = base_query.offset((page - 1) * perPage).limit(perPage).all()
    
    return {
        "words": [{"word": w.word, "id": w.id} for w in words],
        "page": page,
        "perPage": perPage,
        "total": total,
    }

@query.field("getWordNetwork")
def resolve_get_word_network(_, info, word, depth=2, breadth=10):
    # Implement the word network logic here
    # This is a placeholder implementation
    return {
        "nodes": [],
        "links": []
    }

@word.field("pronunciation")
def resolve_pronunciation(word, _):
    return {
        "text": word.pronunciation,
        "ipa": ", ".join([p for p in word.audio_pronunciation or [] if p.startswith("/")]),
        "audio": [p for p in word.audio_pronunciation or [] if p and not p.startswith("/")],
    }

@word.field("etymology")
def resolve_etymology(word, _):
    if word.etymologies:
        etymology = word.etymologies[0]
        return {
            "kaikki": word.kaikki_etymology,
            "components": [{"component": comp.component, "order": comp.order} for comp in etymology.components],
            "text": [etymology.etymology_text],
            "parsed": [],  # Implement parsing logic if needed
        }
    return None

@word.field("definitions")
def resolve_definitions(word, _):
    return [{
        "partOfSpeech": definition.part_of_speech,
        "meanings": [{
            "definition": meaning.meaning,
            "source": meaning.source.source_name if meaning.source else None
        } for meaning in definition.meanings],
        "usageNotes": definition.usage_notes,
        "examples": [example.example for example in definition.examples],
        "tags": definition.tags,
    } for definition in word.definitions]

@word.field("relationships")
def resolve_relationships(word, _):
    return {
        "rootWord": word.root_word,
        "derivatives": [d.derivative for d in word.derivatives],
        "synonyms": [s.word for s in word.synonyms],
        "antonyms": [a.word for a in word.antonyms],
        "associatedWords": [aw.associated_word for aw in word.associated_words],
        "relatedTerms": [rt.word for rt in word.related_terms],
        "hypernyms": [h.hypernym for h in word.hypernyms],
        "hyponyms": [h.hyponym for h in word.hyponyms],
        "meronyms": [m.meronym for m in word.meronyms],
        "holonyms": [h.holonym for h in word.holonyms],
    }

@word.field("forms")
def resolve_forms(word, _):
    return [{"form": f.form, "tags": f.tags} for f in word.forms]

@word.field("languages")
def resolve_languages(word, _):
    return [lang.code for lang in word.languages]

@word.field("tags")
def resolve_tags(word, _):
    return word.tags

@word.field("headTemplates")
def resolve_head_templates(word, _):
    return [{"name": ht.template_name, "args": ht.args, "expansion": ht.expansion} for ht in word.head_templates]

@word.field("inflections")
def resolve_inflections(word, _):
    return [{"name": infl.name, "args": infl.args} for infl in word.inflections]

@word.field("alternateForms")
def resolve_alternate_forms(word, _):
    return [af.alternate_form for af in word.alternate_forms]

schema = make_executable_schema(type_defs, [query, word])
graphql_app = GraphQL(schema, debug=True)