import json
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
from models import (
    get_engine, create_tables, Word, Definition, Meaning, Source, Form,
    HeadTemplate, Etymology, EtymologyComponent, EtymologyTemplate,
    Language, Derivative, Example, Hypernym, Hyponym, Meronym, Holonym,
    Pronunciation, AssociatedWord
)
from sqlalchemy.orm.exc import FlushError
from sqlalchemy import and_
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
    if etymology_data:
        etymology = Etymology(word=word, etymology_text=etymology_data)
        session.add(etymology)
        
        # Process etymology components if available
        if 'etymology_components' in etymology_data:
            for i, component in enumerate(etymology_data['etymology_components']):
                etym_component = EtymologyComponent(
                    etymology=etymology,
                    component=component,
                    order=i
                )
                session.add(etym_component)
        
        # Process etymology template if available
        if 'etymology_template' in etymology_data:
            template_data = etymology_data['etymology_template']
            etym_template = EtymologyTemplate(
                etymology=etymology,
                template_name=template_data.get('name'),
                args=template_data.get('args')
            )
            session.add(etym_template)

def process_associated_words(session, associated_words_data, word):
    for associated_word in associated_words_data:
        if isinstance(associated_word, str):
            assoc = AssociatedWord(word=word, associated_word=associated_word)
        elif isinstance(associated_word, dict):
            assoc = AssociatedWord(
                word=word,
                associated_word=associated_word.get('word'),
                relationship_type=associated_word.get('type')
            )
        session.add(assoc)

def migrate_word(session, word_data):
    logger.info(f"Processing word: {word_data['word']}")

    try:
        # Create or get the word
        word = get_or_create(session, Word, word=word_data['word'])

        # Handle pronunciation
        if 'pronunciation' in word_data:
            pronunciation = get_or_create(session, Pronunciation, pronunciation=word_data['pronunciation'])
            word.pronunciation = pronunciation

        # Update word attributes
        word.audio_pronunciation = word_data.get('audio_pronunciation', [])
        word.tags = word_data.get('tags', [])
        word.kaikki_etymology = word_data.get('kaikki_etymology')

        # Process root word
        if word_data.get('root_word'):
            root_word = get_or_create(session, Word, word=word_data['root_word'])
            word.root_word = root_word.word

        # Process languages
        for language_code in word_data.get('language_codes', []):
            language = get_or_create(session, Language, code=language_code)
            if language not in word.languages:
                word.languages.append(language)

        # Process etymology
        process_etymology(session, word_data.get('etymology'), word)

        # Process definitions
        for definition_data in word_data.get('definitions', []):
            definition = Definition(
                word=word,
                part_of_speech=definition_data['part_of_speech'],
                usage_notes=definition_data.get('usage_notes', []),
                tags=definition_data.get('tags', [])
            )
            session.add(definition)

            for source_name, meanings in definition_data['meanings_by_source'].items():
                source = get_or_create(session, Source, source_name=source_name)
                for meaning_text in meanings:
                    meaning = Meaning(definition=definition, source=source, meaning=meaning_text)
                    session.add(meaning)

        # Process forms
        for form_data in word_data.get('forms', []):
            form = Form(
                word=word,
                form=form_data['form'],
                tags=form_data.get('tags', [])
            )
            session.add(form)

        # Process head templates
        for template_data in word_data.get('head_templates', []):
            template = HeadTemplate(
                word=word,
                template_name=template_data['name'],
                args=template_data.get('args', {}),
                expansion=template_data.get('expansion', '')
            )
            session.add(template)

        # Process derivatives
        for derivative in word_data.get('derivatives', []):
            if isinstance(derivative, str):
                deriv = Derivative(word=word, derivative=derivative)
                session.add(deriv)

        # Process examples
        for example in word_data.get('examples', []):
            ex = Example(word=word, example=example)
            session.add(ex)

        # Process semantic relations
        for relation_class, relation_data in [
            (Hypernym, word_data.get('hypernyms', [])),
            (Hyponym, word_data.get('hyponyms', [])),
            (Meronym, word_data.get('meronyms', [])),
            (Holonym, word_data.get('holonyms', []))
        ]:
            for relation in relation_data:
                rel = relation_class(word=word, **{relation_class.__tablename__[:-1]: relation})
                session.add(rel)

        # Process associated words
        process_associated_words(session, word_data.get('associated_words', []), word)

        # Process synonyms and antonyms
        for synonym in word_data.get('synonyms', []):
            syn = get_or_create(session, Word, word=synonym)
            if syn not in word.synonyms:
                word.synonyms.append(syn)

        for antonym in word_data.get('antonyms', []):
            ant = get_or_create(session, Word, word=antonym)
            if ant not in word.antonyms:
                word.antonyms.append(ant)

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
    existing_word.audio_pronunciation = word_data.get('audio_pronunciation', [])
    existing_word.tags = word_data.get('tags', [])
    existing_word.kaikki_etymology = word_data.get('kaikki_etymology')

    # Update pronunciation
    if 'pronunciation' in word_data:
        pronunciation = get_or_create(session, Pronunciation, pronunciation=word_data['pronunciation'])
        existing_word.pronunciation = pronunciation

    # Update languages
    unique_language_codes = set(word_data.get('language_codes', []))
    existing_language_ids = set(lang.id for lang in existing_word.languages)
    for language_code in unique_language_codes:
        language = get_or_create(session, Language, code=language_code)
        if language.id not in existing_language_ids:
            existing_word.languages.append(language)
            existing_language_ids.add(language.id)

    # Update other relationships (definitions, forms, etc.) as needed
    # Be careful to avoid duplicates

    session.add(existing_word)

def migrate_data(json_data):
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        if 'words' not in existing_tables:
            logger.warning("'words' table not found in the database. Creating tables...")
            create_tables()

        total_words = len(json_data)
        with tqdm(total=total_words, desc="Migrating words") as pbar:
            for word_key, word_data in json_data.items():
                migrate_word(session, word_data)
                pbar.update(1)
    except Exception as e:
        logger.error(f"Error occurred during migration: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    create_tables()
    logger.info("Database tables created successfully.")

    file_path = '../data/combined_filipino_dictionary.json'
    json_data = load_json_data(file_path)
    
    migrate_data(json_data)
    logger.info("Data migration completed successfully.")