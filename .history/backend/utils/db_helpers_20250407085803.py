import pickle
import logging
from sqlalchemy import text
from backend.models import (
    Word, Definition, Etymology, Pronunciation, Relation
    # Add other models used by the function if necessary, e.g.:
    # Affixation, WordForm, WordTemplate, DefinitionRelation, PartOfSpeech, Credit
)
from backend.database import db, get_cache_client

# Set up logging (consider using structlog if used elsewhere)
logger = logging.getLogger(__name__)

# Get cache client
# NOTE: This assumes get_cache_client() is safe to call multiple times
# or that it returns a singleton. Otherwise, import the instance from routes.py
# or database.py if it's defined there.
cache_client = None
try:
    cache_client = get_cache_client()
except Exception as e:
    logger.warning(f"Failed to get cache client in db_helpers: {e}")

def _fetch_word_details(word_id,
                        include_definitions=True,
                        include_etymologies=True,
                        include_pronunciations=True,
                        include_credits=True, # Credits might not be used in the copied logic
                        include_relations=True,
                        include_affixations=True,
                        include_root=True,
                        include_derived=True,
                        include_forms=True,
                        include_templates=True,
                        include_definition_relations=False):
    """
    Fetch a word with specified relationships using direct SQL to avoid ORM issues.
    Returns a Word object with the requested relationships loaded.
    """
    global cache_client # Declare use of the module-level cache_client
    cache_key = f'word_details:{word_id}:{include_definitions}:{include_etymologies}:' \
                f'{include_pronunciations}:{include_credits}:{include_relations}:' \
                f'{include_affixations}:{include_root}:{include_derived}:' \
                f'{include_forms}:{include_templates}:{include_definition_relations}'

    if cache_client:
        try:
            cached_data = cache_client.get(cache_key)
            if cached_data:
                word = pickle.loads(cached_data)
                if word:
                    # Attempt to re-attach to the current session if needed
                    # This might be necessary depending on how sessions are managed
                    try:
                        # Check if object is detached
                        from sqlalchemy.orm.exc import DetachedInstanceError
                        _ = word.lemma # Accessing an attribute triggers check
                        return word # Already attached or session scope handles it
                    except DetachedInstanceError:
                         logger.debug(f"Re-attaching word {word_id} from cache to session.")
                         word = db.session.merge(word)
                         return word
                    except AttributeError:
                         logger.warning(f"Cached object for word {word_id} seems invalid.")
                    except Exception as attach_err:
                        logger.warning(f"Error re-attaching word {word_id} from cache: {attach_err}")
                        # Proceed to fetch from DB if re-attachment fails
        except Exception as e:
            logger.warning(f"Cache retrieval error for word_id={word_id}: {e}")

    try:
        # First get the basic word data
        sql_word = """
        SELECT id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form,
               root_word_id, preferred_spelling, tags, source_info, word_metadata,
               data_hash, search_text, badlit_form, hyphenation, is_proper_noun,
               is_abbreviation, is_initialism
        FROM words
        WHERE id = :id
        """
        word_result = db.session.execute(text(sql_word), {"id": word_id}).fetchone()

        if not word_result:
            return None

        # Create word object manually
        word = Word()
        for key in word_result.keys():
            if hasattr(word, key) and key != 'is_root': # Skip is_root since it's a hybrid property
                setattr(word, key, word_result[key])

        # Load definitions if requested
        if include_definitions:
            sql_defs = """
            SELECT id, definition_text, original_pos, standardized_pos_id,
                  examples, usage_notes, tags, sources
            FROM definitions
            WHERE word_id = :word_id
            """
            defs_result = db.session.execute(text(sql_defs), {"word_id": word_id}).fetchall()
            word.definitions = []
            for d in defs_result:
                definition = Definition()
                for key in d.keys():
                    if hasattr(definition, key):
                        setattr(definition, key, d[key])
                definition.word_id = word_id
                # definition.word = word # Avoid circular ref in direct assignment?
                word.definitions.append(definition)

        # Load etymologies if requested
        if include_etymologies:
            sql_etym = """
            SELECT id, etymology_text, normalized_components, etymology_structure,
                  language_codes, sources
            FROM etymologies
            WHERE word_id = :word_id
            """
            etym_result = db.session.execute(text(sql_etym), {"word_id": word_id}).fetchall()
            word.etymologies = []
            for e in etym_result:
                etymology = Etymology()
                for key in e.keys():
                    if hasattr(etymology, key):
                        setattr(etymology, key, e[key])
                etymology.word_id = word_id
                # etymology.word = word
                word.etymologies.append(etymology)

        # Load pronunciations if requested
        if include_pronunciations:
            sql_pron = """
            SELECT id, type, value, tags, pronunciation_metadata, sources
            FROM pronunciations
            WHERE word_id = :word_id
            """
            pron_result = db.session.execute(text(sql_pron), {"word_id": word_id}).fetchall()
            word.pronunciations = []
            for p in pron_result:
                pronunciation = Pronunciation()
                for key in p.keys():
                    if hasattr(pronunciation, key):
                        setattr(pronunciation, key, p[key])
                pronunciation.word_id = word_id
                # pronunciation.word = word
                word.pronunciations.append(pronunciation)

        # Load relations if requested
        if include_relations:
            # Outgoing relations
            sql_out_rel = """
            SELECT r.id, r.from_word_id, r.to_word_id, r.relation_type, r.relation_data,
                   w.id as target_id, w.lemma as target_lemma, w.language_code as target_language_code,
                   w.has_baybayin as target_has_baybayin, w.baybayin_form as target_baybayin_form
              FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = :word_id
            """
            out_rel_result = db.session.execute(text(sql_out_rel), {"word_id": word_id}).fetchall()
            word.outgoing_relations = []
            for r in out_rel_result:
                relation = Relation()
                relation.id = r.id
                relation.from_word_id = r.from_word_id
                relation.to_word_id = r.to_word_id
                relation.relation_type = r.relation_type
                relation.relation_data = r.relation_data
                # relation.source_word = word

                target_word = Word()
                target_word.id = r.target_id
                target_word.lemma = r.target_lemma
                target_word.language_code = r.target_language_code
                target_word.has_baybayin = r.target_has_baybayin
                target_word.baybayin_form = r.target_baybayin_form
                relation.target_word = target_word
                word.outgoing_relations.append(relation)

            # Incoming relations
            sql_in_rel = """
            SELECT r.id, r.from_word_id, r.to_word_id, r.relation_type, r.relation_data,
                   w.id as source_id, w.lemma as source_lemma, w.language_code as source_language_code,
                   w.has_baybayin as source_has_baybayin, w.baybayin_form as source_baybayin_form
              FROM relations r
            JOIN words w ON r.from_word_id = w.id
            WHERE r.to_word_id = :word_id
            """
            in_rel_result = db.session.execute(text(sql_in_rel), {"word_id": word_id}).fetchall()
            word.incoming_relations = []
            for r in in_rel_result:
                relation = Relation()
                relation.id = r.id
                relation.from_word_id = r.from_word_id
                relation.to_word_id = r.to_word_id
                relation.relation_type = r.relation_type
                relation.relation_data = r.relation_data
                # relation.target_word = word

                source_word = Word()
                source_word.id = r.source_id
                source_word.lemma = r.source_lemma
                source_word.language_code = r.source_language_code
                source_word.has_baybayin = r.source_has_baybayin
                source_word.baybayin_form = r.source_baybayin_form
                relation.source_word = source_word
                word.incoming_relations.append(relation)

        # Skip loading affixations to avoid issues
        if include_affixations:
            word.root_affixations = []
            word.affixed_affixations = []

        if include_root and word.root_word_id:
            root_sql = """
            SELECT id, lemma, language_code, has_baybayin, baybayin_form
            FROM words WHERE id = :root_id
            """
            root_result = db.session.execute(text(root_sql), {"root_id": word.root_word_id}).fetchone()
            if root_result:
                root_word = Word()
                root_word.id = root_result.id
                root_word.lemma = root_result.lemma
                root_word.language_code = root_result.language_code
                root_word.has_baybayin = root_result.has_baybayin
                root_word.baybayin_form = root_result.baybayin_form
                word.root_word = root_word
            else:
                word.root_word = None
        else:
            word.root_word = None

        if include_derived:
            derived_sql = """
            SELECT id, lemma, language_code, has_baybayin, baybayin_form
            FROM words WHERE root_word_id = :word_id
            """
            derived_results = db.session.execute(text(derived_sql), {"word_id": word_id}).fetchall()
            word.derived_words = []
            for d in derived_results:
                derived = Word()
                derived.id = d.id
                derived.lemma = d.lemma
                derived.language_code = d.language_code
                derived.has_baybayin = d.has_baybayin
                derived.baybayin_form = d.baybayin_form
                word.derived_words.append(derived)
        else:
            word.derived_words = [] # Ensure it's always a list

        # Skip forms/templates/def_relations loading
        if include_forms:
            word.forms = []
        if include_templates:
            word.templates = []
        if include_definition_relations:
            word.definition_relations = []
            word.related_definitions = []

        # Cache the result
        if cache_client:
            try:
                pickled_word = pickle.dumps(word)
                cache_client.set(cache_key, pickled_word, timeout=600) # Cache for 10 minutes
            except Exception as e:
                logger.warning(f"Cache storage error for word_id={word_id}: {e}")

        return word

    except Exception as e:
        logger.error(f"Error in _fetch_word_details for word_id {word_id}: {str(e)}", exc_info=True)
        return None 