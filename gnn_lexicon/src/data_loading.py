"""
Data loading utilities for Philippine Lexicon GNN.
Supports PostgreSQL and CSV/JSON fallback for toy graphs.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import os
import json
import csv
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

def load_pg_connection(cfg: Dict[str, Any]) -> Optional["psycopg2.extensions.connection"]:
    """Connect to PostgreSQL using config dict."""
    if psycopg2 is None:
        print("[data_loading] psycopg2 not installed.")
        return None
    try:
        conn = psycopg2.connect(**cfg)
        return conn
    except Exception as e:
        print(f"[data_loading] PostgreSQL connection failed: {e}")
        return None

def fetch_graph_from_postgres(conn) -> Dict[str, Any]:
    """
    Extracts nodes and edges from the PostgreSQL DB using the full schema.
    Returns a dict with node/edge lists and features.
    """
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # --- Nodes ---
    # Words (now Tagalog and Cebuano)
    cur.execute("""
        SELECT id, lemma, normalized_lemma, language_code, is_proper_noun, is_abbreviation, 
               is_initialism, has_baybayin, baybayin_form, romanized_form, root_word_id, 
               preferred_spelling, tags, idioms, pronunciation_data, source_info, word_metadata, 
               badlit_form, hyphenation
        FROM words 
        WHERE language_code IN ('tl', 'ceb')
        LIMIT 20000;
    """)
    words = cur.fetchall()
    word_ids = [w['id'] for w in words]
    
    # Word forms
    cur.execute("""
        SELECT id, form, is_canonical, is_primary, word_id, tags
        FROM word_forms 
        WHERE word_id = ANY(%s);
    """, (word_ids,))
    forms = cur.fetchall()
    
    # Definitions
    cur.execute("""
        SELECT id, definition_text, original_pos, standardized_pos_id, examples, 
               usage_notes, tags, sources, definition_metadata, word_id
        FROM definitions 
        WHERE word_id = ANY(%s);
    """, (word_ids,))
    definitions = cur.fetchall()
    
    # Parts of speech
    cur.execute("SELECT id, code, name_en, name_tl, description FROM parts_of_speech;")
    parts_of_speech = cur.fetchall()
    
    # Etymologies
    cur.execute("""
        SELECT id, etymology_text, normalized_components, etymology_structure, 
               language_codes, sources, word_id
        FROM etymologies 
        WHERE word_id = ANY(%s);
    """, (word_ids,))
    etymologies = cur.fetchall()
    
    # Pronunciations
    cur.execute("""
        SELECT id, type, value, tags, pronunciation_metadata, word_id
        FROM pronunciations 
        WHERE word_id = ANY(%s);
    """, (word_ids,))
    pronunciations = cur.fetchall()
    
    # Word templates
    cur.execute("""
        SELECT id, template_name, args, expansion, sources, word_id
        FROM word_templates 
        WHERE word_id = ANY(%s);
    """, (word_ids,))
    word_templates = cur.fetchall()
    
    # Definition examples
    definition_ids = [d['id'] for d in definitions]
    cur.execute("""
        SELECT id, example_text, translation, example_type, reference, metadata, sources, definition_id
        FROM definition_examples 
        WHERE definition_id = ANY(%s);
    """, (definition_ids,))
    definition_examples = cur.fetchall()
    
    # Definition categories
    cur.execute("""
        SELECT id, category_name, category_kind, parents, sources, category_metadata, definition_id
        FROM definition_categories 
        WHERE definition_id = ANY(%s);
    """, (definition_ids,))
    definition_categories = cur.fetchall()
    
    # Definition links
    cur.execute("""
        SELECT id, link_text, tags, link_metadata, sources, definition_id
        FROM definition_links 
        WHERE definition_id = ANY(%s);
    """, (definition_ids,))
    definition_links = cur.fetchall()
    
    # Languages
    # cur.execute("SELECT id, code, name_en, name_tl, region, family, status FROM languages;")
    # languages = cur.fetchall()
    languages = []
    
    # --- Edges ---
    # Only root_of and synonym relations from relations table
    cur.execute("""
        SELECT from_word_id, to_word_id, relation_type, sources, metadata
        FROM relations 
        WHERE (from_word_id = ANY(%s) OR to_word_id = ANY(%s))
        AND relation_type IN ('synonym', 'root_of');
    """, (word_ids, word_ids))
    relations = cur.fetchall()
    
    # Word-Form relationships
    has_form = [{"word_id": f["word_id"], "form_id": f["id"]} for f in forms if f["word_id"] is not None]
    of_word = [{"form_id": f["id"], "word_id": f["word_id"]} for f in forms if f["word_id"] is not None]
    
    # Word-Definition relationships
    has_definition = [{"word_id": d["word_id"], "definition_id": d["id"]} for d in definitions if d["word_id"] is not None]
    
    # Word-PartOfSpeech relationships
    has_pos = [{"word_id": d["word_id"], "pos_id": d["standardized_pos_id"]} for d in definitions if d["standardized_pos_id"] is not None]
    
    # Word-Etymology relationships
    has_etymology = [{"word_id": e["word_id"], "etymology_id": e["id"]} for e in etymologies if e["word_id"] is not None]
    
    # Word-Pronunciation relationships
    has_pronunciation = [{"word_id": p["word_id"], "pronunciation_id": p["id"]} for p in pronunciations if p["word_id"] is not None]
    
    # Word-Template relationships
    has_template = [{"word_id": t["word_id"], "template_id": t["id"]} for t in word_templates if t["word_id"] is not None]
    
    # Definition-Example relationships
    has_example = [{"definition_id": ex["definition_id"], "example_id": ex["id"]} for ex in definition_examples if ex["definition_id"] is not None]
    
    # Definition-Category relationships
    has_category = [{"definition_id": cat["definition_id"], "category_id": cat["id"]} for cat in definition_categories if cat["definition_id"] is not None]
    
    # Definition-Link relationships
    has_link = [{"definition_id": link["definition_id"], "link_id": link["id"]} for link in definition_links if link["definition_id"] is not None]
    
    # Word-Language relationships
    in_language = [{"word_id": w["id"], "language_id": w["language_code"]} for w in words]
    
    # Root word relationships (self-referential)
    root_of = [{"root_id": w["root_word_id"], "word_id": w["id"]} for w in words if w["root_word_id"] is not None]
    
    # Create relation-specific edges
    synonym_of = [{"word1_id": r["from_word_id"], "word2_id": r["to_word_id"]} for r in relations if r["relation_type"] == "synonym"]
    antonym_of = [{"word1_id": r["from_word_id"], "word2_id": r["to_word_id"]} for r in relations if r["relation_type"] == "antonym"]
    related_to = [{"word1_id": r["from_word_id"], "word2_id": r["to_word_id"]} for r in relations if r["relation_type"] == "related"]
    derived_from = [{"word1_id": r["from_word_id"], "word2_id": r["to_word_id"]} for r in relations if r["relation_type"] == "derived_from"]
    cognate_of = [{"word1_id": r["from_word_id"], "word2_id": r["to_word_id"]} for r in relations if r["relation_type"] == "cognate_of"]
    translation_of = [{"word1_id": r["from_word_id"], "word2_id": r["to_word_id"]} for r in relations if r["relation_type"] == "translation_of"]
    see_also = [{"word1_id": r["from_word_id"], "word2_id": r["to_word_id"]} for r in relations if r["relation_type"] == "see_also"]
    variant_of = [{"word1_id": r["from_word_id"], "word2_id": r["to_word_id"]} for r in relations if r["relation_type"] == "variant"]
    doublet_of = [{"word1_id": r["from_word_id"], "word2_id": r["to_word_id"]} for r in relations if r["relation_type"] == "doublet_of"]
    
    return {
        # Nodes
        "words": words,
        "forms": forms,
        "definitions": definitions,
        "parts_of_speech": parts_of_speech,
        "etymologies": etymologies,
        "pronunciations": pronunciations,
        "word_templates": word_templates,
        "definition_examples": definition_examples,
        "definition_categories": definition_categories,
        "definition_links": definition_links,
        "languages": languages,
        
        # Edges
        "relations": relations,
        "has_form": has_form,
        "of_word": of_word,
        "has_definition": has_definition,
        "has_pos": has_pos,
        "has_etymology": has_etymology,
        "has_pronunciation": has_pronunciation,
        "has_template": has_template,
        "has_example": has_example,
        "has_category": has_category,
        "has_link": has_link,
        "in_language": in_language,
        "root_of": root_of,
        "synonym_of": synonym_of,
        "antonym_of": antonym_of,
        "related_to": related_to,
        "derived_from": derived_from,
        "cognate_of": cognate_of,
        "translation_of": translation_of,
        "see_also": see_also,
        "variant_of": variant_of,
        "doublet_of": doublet_of,
    }

def load_toy_graph(path: str) -> Dict[str, Any]:
    """Loads a toy graph from a JSON file for offline/demo use."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_graph_json(graph: Dict[str, Any], path: str) -> None:
    """Saves a graph dict to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

def create_toy_graph() -> Dict[str, Any]:
    """Creates a small toy graph for testing without DB."""
    return {
        "words": [
            {"id": 0, "lemma": "takbo", "language_code": "tl", "frequency": 100, "is_proper_noun": False},
            {"id": 1, "lemma": "tumakbo", "language_code": "tl", "frequency": 50, "is_proper_noun": False},
            {"id": 2, "lemma": "lakad", "language_code": "tl", "frequency": 80, "is_proper_noun": False},
            {"id": 3, "lemma": "kain", "language_code": "tl", "frequency": 90, "is_proper_noun": False},
            {"id": 4, "lemma": "kumain", "language_code": "tl", "frequency": 40, "is_proper_noun": False},
        ],
        "forms": [
            {"id": 0, "form": "tatakbo", "is_canonical": False, "is_primary": False, "word_id": 0},
            {"id": 1, "form": "nagtakbo", "is_canonical": False, "is_primary": False, "word_id": 0},
            {"id": 2, "form": "lalakad", "is_canonical": False, "is_primary": False, "word_id": 2},
        ],
        "morphemes": [
            {"id": 0, "morpheme_text": "um", "morpheme_type": "infix"},
            {"id": 1, "morpheme_text": "mag", "morpheme_type": "prefix"},
            {"id": 2, "morpheme_text": "in", "morpheme_type": "infix"},
        ],
        "senses": [
            {"id": 0, "definition_text": "to run", "original_pos": "verb", "word_id": 0},
            {"id": 1, "definition_text": "to walk", "original_pos": "verb", "word_id": 2},
            {"id": 2, "definition_text": "to eat", "original_pos": "verb", "word_id": 3},
        ],
        "languages": [
            {"id": 0, "language_code": "tl"},
            {"id": 1, "language_code": "en"},
        ],
        "relations": [
            {"from_word_id": 0, "to_word_id": 1, "relation_type": "derived_from"},
            {"from_word_id": 3, "to_word_id": 4, "relation_type": "derived_from"},
            {"from_word_id": 0, "to_word_id": 2, "relation_type": "related_motion"},
        ],
        "has_form": [
            {"word_id": 0, "form_id": 0},
            {"word_id": 0, "form_id": 1},
            {"word_id": 2, "form_id": 2},
        ],
        "of_word": [
            {"form_id": 0, "word_id": 0},
            {"form_id": 1, "word_id": 0},
            {"form_id": 2, "word_id": 2},
        ],
        "has_sense": [
            {"word_id": 0, "definition_id": 0},
            {"word_id": 2, "definition_id": 1},
            {"word_id": 3, "definition_id": 2},
        ],
        "has_affix": [
            {"word_id": 1, "morpheme_id": 0},  # tumakbo has um
            {"word_id": 4, "morpheme_id": 0},  # kumain has um
        ],
        "shares_phon": [
            {"word1_id": 0, "word2_id": 1},  # takbo, tumakbo
            {"word1_id": 3, "word2_id": 4},  # kain, kumain
        ],
        "shares_etym": [],
    } 