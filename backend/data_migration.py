import json
import logging
import os
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
from models import (
    Word, Definition, Meaning, Source, Form,
    HeadTemplate, Etymology, EtymologyComponent, Language, Derivative, 
    Example, Hypernym, Hyponym, Meronym, Holonym, AssociatedWord, 
    AlternateForm, Inflection
)
from database import db_session, engine, Base
from sqlalchemy.orm.exc import FlushError
from tqdm import tqdm

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create handlers
success_handler = logging.FileHandler('logs/success.log')
error_handler = logging.FileHandler('logs/error.log')
overall_handler = logging.FileHandler('logs/overall.log')

# Set levels
success_handler.setLevel(logging.INFO)
error_handler.setLevel(logging.ERROR)
overall_handler.setLevel(logging.DEBUG)

# Create formatters and add to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
success_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
overall_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(success_handler)
logger.addHandler(error_handler)
logger.addHandler(overall_handler)

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

def get_or_create(session, model, create_new=True, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    elif create_new:
        instance = model(**kwargs)
        session.add(instance)
        return instance
    else:
        return None

def process_etymology(session, etymology_data, word):
    if not etymology_data:
        return

    try:
        etymology = Etymology(word=word, etymology_text=str(etymology_data))
        session.add(etymology)

        if isinstance(etymology_data, (list, dict)):
            components = []
            if isinstance(etymology_data, dict):
                components = etymology_data.get('etymology_components', [])
            elif isinstance(etymology_data, list):
                components = etymology_data

            etym_components = []
            for i, component in enumerate(components):
                etym_component = EtymologyComponent(
                    etymology=etymology,
                    component=str(component),
                    order=i
                )
                etym_components.append(etym_component)
            session.add_all(etym_components)
    except Exception as e:
        logger.error(f"Error processing etymology for word '{word.word}': {e}")
        raise

def process_definitions(session, definitions_data, word):
    try:
        definitions = []
        meanings = []
        examples = []

        for definition_data in definitions_data:
            definition = Definition(
                word=word,
                part_of_speech=definition_data.get('part_of_speech'),
                usage_notes=definition_data.get('usage_notes', []),
                tags=definition_data.get('tags', [])
            )
            definitions.append(definition)

            for source_name, meaning_texts in definition_data.get('meanings_by_source', {}).items():
                source = get_or_create(session, Source, source_name=source_name)
                for meaning_text in meaning_texts:
                    meaning = Meaning(definition=definition, source=source, meaning=meaning_text)
                    meanings.append(meaning)

            for example_text in definition_data.get('examples', []):
                example = Example(word=word, definition=definition, example=example_text)
                examples.append(example)

        session.add_all(definitions)
        session.add_all(meanings)
        session.add_all(examples)
    except Exception as e:
        logger.error(f"Error processing definitions for word '{word.word}': {e}")
        raise

def process_forms(session, forms_data, word):
    try:
        forms = []
        for form_data in forms_data:
            form = Form(
                word=word,
                form=form_data.get('form'),
                tags=form_data.get('tags', [])
            )
            forms.append(form)
        session.add_all(forms)
    except Exception as e:
        logger.error(f"Error processing forms for word '{word.word}': {e}")
        raise

def process_head_templates(session, templates_data, word):
    try:
        templates = []
        for template_data in templates_data:
            template = HeadTemplate(
                word=word,
                template_name=template_data.get('name'),
                args=template_data.get('args', {}),
                expansion=template_data.get('expansion', '')
            )
            templates.append(template)
        session.add_all(templates)
    except Exception as e:
        logger.error(f"Error processing head templates for word '{word.word}': {e}")
        raise

def process_associated_words(session, associated_words_data, word):
    try:
        associated_words = []
        for associated_word in associated_words_data:
            assoc = AssociatedWord(word=word, associated_word=associated_word)
            associated_words.append(assoc)
        session.add_all(associated_words)
    except Exception as e:
        logger.error(f"Error processing associated words for '{word.word}': {e}")
        raise

def process_alternate_forms(session, alternate_forms_data, word):
    try:
        alternate_forms = []
        for alternate_form in alternate_forms_data:
            alt_form = AlternateForm(word=word, alternate_form=alternate_form)
            alternate_forms.append(alt_form)
        session.add_all(alternate_forms)
    except Exception as e:
        logger.error(f"Error processing alternate forms for '{word.word}': {e}")
        raise

def process_semantic_relations(session, relations_data, word, relation_class):
    try:
        relations = []
        field_name = relation_class.__tablename__[:-1]  # Remove 's' from table name to get field name
        for relation in relations_data:
            rel = relation_class(word=word, **{field_name: relation})
            relations.append(rel)
        session.add_all(relations)
    except Exception as e:
        logger.error(f"Error processing {relation_class.__name__} for '{word.word}': {e}")
        raise

def process_inflections(session, inflections_data, word):
    try:
        inflections = []
        for inflection_data in inflections_data:
            inflection = Inflection(
                word=word,
                name=inflection_data.get('name'),
                args=inflection_data.get('args', {})
            )
            inflections.append(inflection)
        session.add_all(inflections)
    except Exception as e:
        logger.error(f"Error processing inflections for '{word.word}': {e}")
        raise

def migrate_word(session, word_data):
    try:
        # Try to get the word, or create if not exists
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
        derivatives = []
        for derivative in word_data.get('derivatives', []):
            if isinstance(derivative, str):
                deriv = Derivative(word=word, derivative=derivative)
                derivatives.append(deriv)
        session.add_all(derivatives)

        # Process examples
        examples = []
        for example in word_data.get('examples', []):
            ex = Example(word=word, example=example)
            examples.append(ex)
        session.add_all(examples)

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

        return True, None
    except Exception as e:
        logger.error(f"Error migrating word '{word_data.get('word')}': {e}")
        return False, str(e)
    
def update_word_data(existing_word, new_word_data):
    logger.info(f"Updating data for word: {existing_word.word}")
    
    # Update basic attributes
    basic_attrs = ['pronunciation', 'audio_pronunciation', 'tags', 'kaikki_etymology', 'variant', 'root_word']
    for attr in basic_attrs:
        new_value = new_word_data.get(attr)
        if new_value is not None:
            setattr(existing_word, attr, new_value)
            logger.info(f"Updated {attr} for word {existing_word.word}")

    # Update or append related entities
    related_entities = [
        ('definitions', 'definitions', Definition), ('forms', 'forms', Form),
        ('head_templates', 'head_templates', HeadTemplate), ('languages', 'language_codes', Language),
        ('derivatives', 'derivatives', Derivative), ('examples', 'examples', Example),
        ('associated_words', 'associated_words', AssociatedWord), 
        ('alternate_forms', 'alternate_forms', AlternateForm),
        ('hypernyms', 'hypernyms', Hypernym), ('hyponyms', 'hyponyms', Hyponym),
        ('meronyms', 'meronyms', Meronym), ('holonyms', 'holonyms', Holonym),
        ('inflections', 'inflections', Inflection), ('synonyms', 'synonyms', Word),
        ('antonyms', 'antonyms', Word), ('related_terms', 'related_terms', Word)
    ]

    for existing_attr, new_attr, model in related_entities:
        existing_items = getattr(existing_word, existing_attr)
        new_items = new_word_data.get(new_attr, [])
        
        for new_item in new_items:
            if isinstance(new_item, dict):
                # For complex objects like definitions
                existing_item = next((item for item in existing_items if item.matches(new_item)), None)
                if existing_item:
                    existing_item.update(new_item)
                    logger.info(f"Updated existing {existing_attr} for word {existing_word.word}")
                else:
                    new_obj = model(**new_item)
                    existing_items.append(new_obj)
                    logger.info(f"Added new {existing_attr} for word {existing_word.word}")
            else:
                # For simple objects like synonyms
                if new_item not in existing_items:
                    new_obj = model(word=new_item) if model == Word else model(**{existing_attr.rstrip('s'): new_item})
                    existing_items.append(new_obj)
                    logger.info(f"Added new {existing_attr} for word {existing_word.word}")

    logger.info(f"Finished updating data for word: {existing_word.word}")


def migrate_single_word(session, word_key, word_data):
    logger.info(f"Starting migration for word: {word_key}")
    if not isinstance(word_data, dict) or 'word' not in word_data:
        error_message = f"Invalid entry structure for key: {word_key}"
        logger.error(error_message)
        return False, error_message

    # Check if the word already exists in the database
    existing_word = session.query(Word).filter_by(word=word_data['word']).first()
    if existing_word:
        logger.info(f"Word '{word_data['word']}' already exists in the database")
        # Compare the existing word data with the new data
        if compare_word_data(existing_word, word_data):
            logger.info(f"Word '{word_data['word']}' has identical data. Skipping.")
            return True, None
        else:
            logger.info(f"Word '{word_data['word']}' has different data. Updating.")
            update_word_data(existing_word, word_data)
            return True, None

    logger.info(f"Migrating new word: {word_data['word']}")
    success, error_message = migrate_word(session, word_data)
    if success:
        logger.info(f"Successfully migrated word: {word_data['word']}")
    else:
        logger.error(f"Failed to migrate word '{word_data['word']}': {error_message}")
    return success, error_message


def compare_word_data(existing_word, new_word_data):
    logger.info(f"Comparing data for word: {existing_word.word}")
    
    # Compare basic attributes
    basic_attrs = ['pronunciation', 'audio_pronunciation', 'tags', 'kaikki_etymology', 'variant', 'root_word']
    for attr in basic_attrs:
        existing_value = getattr(existing_word, attr)
        new_value = new_word_data.get(attr)
        if existing_value != new_value:
            logger.info(f"Difference found in {attr}: Existing: {existing_value}, New: {new_value}")
            return False

    # Compare related entities
    related_entities = [
        ('definitions', 'definitions'), ('forms', 'forms'), ('head_templates', 'head_templates'),
        ('languages', 'language_codes'), ('derivatives', 'derivatives'), ('examples', 'examples'),
        ('associated_words', 'associated_words'), ('alternate_forms', 'alternate_forms'),
        ('hypernyms', 'hypernyms'), ('hyponyms', 'hyponyms'), ('meronyms', 'meronyms'),
        ('holonyms', 'holonyms'), ('inflections', 'inflections'), ('synonyms', 'synonyms'),
        ('antonyms', 'antonyms'), ('related_terms', 'related_terms')
    ]

    for existing_attr, new_attr in related_entities:
        existing_count = len(getattr(existing_word, existing_attr))
        new_count = len(new_word_data.get(new_attr, []))
        if existing_count != new_count:
            logger.info(f"Difference found in {existing_attr}: Existing count: {existing_count}, New count: {new_count}")
            return False

    logger.info(f"No differences found for word: {existing_word.word}")
    return True

def migrate_data(json_data):
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    if 'words' not in existing_tables:
        logger.warning("'words' table not found in the database. Creating tables...")
        Base.metadata.create_all(engine)

    total_words = len(json_data)
    successful_migrations = 0
    failed_migrations = []

    batch_size = 1000
    batch_counter = 0

    with tqdm(total=total_words, desc="Migrating words") as pbar:
        for word_key, word_data in json_data.items():
            success, error_message = migrate_single_word(db_session, word_key, word_data)
            if success:
                successful_migrations += 1
                logger.info(f"Successfully migrated word: {word_data.get('word')}")
                logger.info(f"Word '{word_data.get('word')}' migration details: {word_data}")
            else:
                failed_migrations.append((word_key, error_message))
                db_session.rollback()  # Rollback any partial changes for this word
                logger.error(f"Failed to migrate word '{word_key}': {error_message}")

            batch_counter += 1
            if batch_counter >= batch_size:
                try:
                    db_session.commit()
                    batch_counter = 0
                except Exception as e:
                    db_session.rollback()
                    logger.error(f"Error committing batch at word '{word_key}': {e}")
                    # Collect all words in this batch as failed
                    failed_migrations.extend([(word_key, str(e))])
                    batch_counter = 0  # Reset batch counter
            pbar.update(1)
    # Commit any remaining words
    if batch_counter > 0:
        try:
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            logger.error(f"Error committing final batch: {e}")

    # Log results
    logger.info(f"Migration completed. Successful: {successful_migrations}, Failed: {len(failed_migrations)}")

    if failed_migrations:
        logger.warning("The following entries failed to migrate:")
        for word_key, error_message in failed_migrations:
            logger.warning(f"  - {word_key}: {error_message}")

        # Write failed migrations to a file for later review
        with open('logs/failed_migrations.log', 'w', encoding='utf-8') as f:
            for word_key, error_message in failed_migrations:
                f.write(f"{word_key}: {error_message}\n")
        logger.info("Failed migrations have been logged to 'logs/failed_migrations.log'")

    db_session.remove()

if __name__ == "__main__":
    file_path = '../data/processed_filipino_dictionary.json'
    try:
        json_data = load_json_data(file_path)
        migrate_data(json_data)
        logger.info("Data migration process completed successfully.")
    except Exception as e:
        logger.error(f"Data migration process failed: {e}")
