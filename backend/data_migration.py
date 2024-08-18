import json
import logging
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
from models import (
    Word, Definition, Meaning, Source, Form,
    HeadTemplate, Etymology, EtymologyComponent, Language, Derivative, 
    Example, Hypernym, Hyponym, Meronym, Holonym, AssociatedWord, 
    AlternateForm, Inflection
)
from database import db_session, engine
from sqlalchemy.orm.exc import FlushError
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_data(file_path):
    logger.info(f"Loading JSON data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"Successfully loaded {len(data)} entries from the JSON file.")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        raise
    except IOError as e:
        logger.error(f"Error reading file: {e}")
        raise

def get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        session.flush()
        return instance

def process_etymology(session, etymology_data, word):
    logger.info(f"Processing etymology for word: {word.word}")
    logger.debug(f"Etymology data: {etymology_data}")

    if not etymology_data:
        logger.info(f"No etymology data for word: {word.word}")
        return

    etymology = Etymology(word=word, etymology_text=str(etymology_data))
    session.add(etymology)

    if isinstance(etymology_data, str):
        # If it's a string, we just save it as etymology_text
        logger.info(f"Saved string etymology for word: {word.word}")
        return
    
    if isinstance(etymology_data, dict):
        # If it's a dictionary, look for 'etymology_components' key
        components = etymology_data.get('etymology_components', [])
    elif isinstance(etymology_data, list):
        # If it's a list, treat it as components directly
        components = etymology_data
    else:
        logger.warning(f"Unexpected etymology data type for word {word.word}: {type(etymology_data)}")
        return

    for i, component in enumerate(components):
        etym_component = EtymologyComponent(
            etymology=etymology,
            component=str(component),
            order=i
        )
        session.add(etym_component)
    
    logger.info(f"Processed {len(components)} etymology components for word: {word.word}")

    session.flush()

def process_definitions(session, definitions_data, word):
    for definition_data in definitions_data:
        definition = Definition(
            word=word,
            part_of_speech=definition_data['part_of_speech'],
            usage_notes=definition_data.get('usage_notes', []),
            tags=definition_data.get('tags', [])
        )
        session.add(definition)

        for source_name, meanings in definition_data.get('meanings_by_source', {}).items():
            source = get_or_create(session, Source, source_name=source_name)
            for meaning_text in meanings:
                meaning = Meaning(definition=definition, source=source, meaning=meaning_text)
                session.add(meaning)

        for example in definition_data.get('examples', []):
            ex = Example(word=word, definition=definition, example=example)
            session.add(ex)

def process_forms(session, forms_data, word):
    for form_data in forms_data:
        form = Form(
            word=word,
            form=form_data['form'],
            tags=form_data.get('tags', [])
        )
        session.add(form)

def process_head_templates(session, templates_data, word):
    for template_data in templates_data:
        template = HeadTemplate(
            word=word,
            template_name=template_data['name'],
            args=template_data.get('args', {}),
            expansion=template_data.get('expansion', '')
        )
        session.add(template)

def process_associated_words(session, associated_words_data, word):
    for associated_word in associated_words_data:
        assoc = AssociatedWord(word=word, associated_word=associated_word)
        session.add(assoc)

def process_alternate_forms(session, alternate_forms_data, word):
    for alternate_form in alternate_forms_data:
        alt_form = AlternateForm(word=word, alternate_form=alternate_form)
        session.add(alt_form)

def process_semantic_relations(session, relations_data, word, relation_class):
    for relation in relations_data:
        rel = relation_class(word=word, **{relation_class.__tablename__[:-1]: relation})
        session.add(rel)

def process_inflections(session, inflections_data, word):
    for inflection_data in inflections_data:
        inflection = Inflection(
            word=word,
            name=inflection_data['name'],
            args=inflection_data.get('args', {})
        )
        session.add(inflection)

def migrate_word(session, word_data):
    logger.info(f"Processing word: {word_data['word']}")

    try:
        # Create or get the word
        word = get_or_create(session, Word, word=word_data['word'])

        # Update word attributes
        word.pronunciation = word_data.get('pronunciation')
        word.audio_pronunciation = word_data.get('audio_pronunciation', [])
        word.tags = word_data.get('tags', [])
        word.kaikki_etymology = word_data.get('kaikki_etymology')
        word.variant = word_data.get('variant')

        # Process root word
        if word_data.get('root_word'):
            root_word = get_or_create(session, Word, word=word_data['root_word'])
            word.root_word = root_word.word

        # Process etymology
        process_etymology(session, word_data.get('etymology'), word)
    
        # Process etymology components separately if they exist
        if 'etymology_components' in word_data:
            process_etymology(session, word_data['etymology_components'], word)

        # Process definitions
        process_definitions(session, word_data.get('definitions', []), word)

        # Process forms
        process_forms(session, word_data.get('forms', []), word)

        # Process head templates
        process_head_templates(session, word_data.get('head_templates', []), word)

        # Process languages
        for language_code in word_data.get('language_codes', []):
            language = get_or_create(session, Language, code=language_code)
            if language not in word.languages:
                word.languages.append(language)

        # Process derivatives
        for derivative in word_data.get('derivatives', []):
            if isinstance(derivative, str):
                deriv = Derivative(word=word, derivative=derivative)
                session.add(deriv)

        # Process examples
        for example in word_data.get('examples', []):
            ex = Example(word=word, example=example)
            session.add(ex)

        # Process associated words
        process_associated_words(session, word_data.get('associated_words', []), word)

        # Process alternate forms
        process_alternate_forms(session, word_data.get('alternate_forms', []), word)

        # Process semantic relations
        process_semantic_relations(session, word_data.get('hypernyms', []), word, Hypernym)
        process_semantic_relations(session, word_data.get('hyponyms', []), word, Hyponym)
        process_semantic_relations(session, word_data.get('meronyms', []), word, Meronym)
        process_semantic_relations(session, word_data.get('holonyms', []), word, Holonym)

        # Process inflections
        process_inflections(session, word_data.get('inflections', []), word)

        # Process synonyms
        for synonym in word_data.get('synonyms', []):
            syn = get_or_create(session, Word, word=synonym)
            if syn not in word.synonyms:
                word.synonyms.append(syn)

        # Process antonyms
        for antonym in word_data.get('antonyms', []):
            ant = get_or_create(session, Word, word=antonym)
            if ant not in word.antonyms:
                word.antonyms.append(ant)

        # Process related terms
        for related_term in word_data.get('related_terms', []):
            rel = get_or_create(session, Word, word=related_term)
            if rel not in word.related_terms:
                word.related_terms.append(rel)

        session.commit()
        logger.info(f"Successfully processed word: {word.word}")

    except FlushError as e:
        session.rollback()
        logger.error(f"Flush error while processing word '{word_data['word']}': {e}")
        # Attempt to update the existing record
        try:
            existing_word = session.query(Word).filter(Word.word == word_data['word']).first()
            if existing_word:
                update_existing_word(session, existing_word, word_data)
                session.commit()
                logger.info(f"Updated existing word: {existing_word.word}")
        except Exception as update_error:
            session.rollback()
            logger.error(f"Error updating existing word '{word_data['word']}': {update_error}")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error while processing word '{word_data['word']}': {e}")
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error while processing word '{word_data['word']}': {e}")

def update_existing_word(session, existing_word, word_data):
    # Update basic attributes
    existing_word.pronunciation = word_data.get('pronunciation')
    existing_word.audio_pronunciation = word_data.get('audio_pronunciation', [])
    existing_word.tags = word_data.get('tags', [])
    existing_word.kaikki_etymology = word_data.get('kaikki_etymology')
    existing_word.variant = word_data.get('variant')

    # Clear existing relationships
    existing_word.etymologies.clear()
    existing_word.definitions.clear()
    existing_word.forms.clear()
    existing_word.head_templates.clear()
    existing_word.languages.clear()
    existing_word.derivatives.clear()
    existing_word.examples.clear()
    existing_word.associated_words.clear()
    existing_word.alternate_forms.clear()
    existing_word.synonyms.clear()
    existing_word.antonyms.clear()
    existing_word.related_terms.clear()
    existing_word.inflections.clear()

    # Re-process all relationships
    process_etymology(session, word_data.get('etymology'), existing_word)
    process_definitions(session, word_data.get('definitions', []), existing_word)
    process_forms(session, word_data.get('forms', []), existing_word)
    process_head_templates(session, word_data.get('head_templates', []), existing_word)
    process_inflections(session, word_data.get('inflections', []), existing_word)

    for language_code in word_data.get('language_codes', []):
        language = get_or_create(session, Language, code=language_code)
        existing_word.languages.append(language)

    for derivative in word_data.get('derivatives', []):
        if isinstance(derivative, str):
            deriv = Derivative(word=existing_word, derivative=derivative)
            session.add(deriv)

    for example in word_data.get('examples', []):
        ex = Example(word=existing_word, example=example)
        session.add(ex)

    process_associated_words(session, word_data.get('associated_words', []), existing_word)
    process_alternate_forms(session, word_data.get('alternate_forms', []), existing_word)

    process_semantic_relations(session, word_data.get('hypernyms', []), existing_word, Hypernym)
    process_semantic_relations(session, word_data.get('hyponyms', []), existing_word, Hyponym)
    process_semantic_relations(session, word_data.get('meronyms', []), existing_word, Meronym)
    process_semantic_relations(session, word_data.get('holonyms', []), existing_word, Holonym)

    for synonym in word_data.get('synonyms', []):
        syn = get_or_create(session, Word, word=synonym)
        existing_word.synonyms.append(syn)

    for antonym in word_data.get('antonyms', []):
        ant = get_or_create(session, Word, word=antonym)
        existing_word.antonyms.append(ant)

    for related_term in word_data.get('related_terms', []):
        rel = get_or_create(session, Word, word=related_term)
        existing_word.related_terms.append(rel)

    session.add(existing_word)

def migrate_data(json_data):
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        if 'words' not in existing_tables:
            logger.warning("'words' table not found in the database. Creating tables...")
            Base.metadata.create_all(engine)

        total_words = len(json_data)
        with tqdm(total=total_words, desc="Migrating words") as pbar:
            for word_key, word_data in json_data.items():
                migrate_word(db_session, word_data)
                pbar.update(1)
    except Exception as e:
        logger.error(f"Error occurred during migration: {e}")
    finally:
        db_session.remove()

if __name__ == "__main__":
    file_path = '../data/processed_filipino_dictionary.json'
    json_data = load_json_data(file_path)
    
    migrate_data(json_data)
    logger.info("Data migration completed successfully.")