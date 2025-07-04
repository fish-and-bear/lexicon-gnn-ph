"""
Data loading utilities for Philippine Lexicon GNN.
Fetches all relevant data from PostgreSQL using the actual schema (see actual_schema.json).
"""

from typing import Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor

# --- DB Connection ---
def load_pg_connection(cfg: Dict[str, Any]) -> Optional["psycopg2.extensions.connection"]:
    try:
        conn = psycopg2.connect(**cfg)
        return conn
    except Exception as e:
        print(f"[data_loading] PostgreSQL connection failed: {e}")
        return None

# --- Main Graph Extraction ---
def fetch_graph_from_postgres(conn) -> Dict[str, Any]:
    """
    Extracts nodes and edges from the PostgreSQL DB using the actual schema.
    Returns a dict with node/edge lists and features.
    Only loads Tagalog (language_code = 'tl') data.
    """
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # --- Nodes ---
    # Words (Tagalog only)
    cur.execute("""
        SELECT id, lemma, normalized_lemma, language_code, root_word_id, is_proper_noun, is_abbreviation, is_initialism, has_baybayin, baybayin_form, romanized_form, preferred_spelling, tags, idioms, pronunciation_data, source_info, word_metadata, data_hash, badlit_form, hyphenation, search_text, created_at, updated_at
        FROM words
        WHERE language_code = 'tl';
    """)
    words = cur.fetchall()
    tagalog_word_ids = set(w['id'] for w in words)

    # Word forms (only for Tagalog words)
    cur.execute("""
        SELECT id, word_id, form, is_canonical, is_primary, tags, sources, created_at, updated_at
        FROM word_forms
        WHERE word_id = ANY(%s);
    """, (list(tagalog_word_ids),))
    forms = cur.fetchall()

    # Morphemes (from affixations and word_templates, only for Tagalog words)
    cur.execute("SELECT DISTINCT affix_type as morpheme_text FROM affixations WHERE affix_type IS NOT NULL AND affixed_word_id = ANY(%s);", (list(tagalog_word_ids),))
    affix_morphemes = [row['morpheme_text'] for row in cur.fetchall()]
    cur.execute("SELECT DISTINCT template_name as morpheme_text FROM word_templates WHERE template_name IS NOT NULL AND word_id = ANY(%s);", (list(tagalog_word_ids),))
    template_morphemes = [row['morpheme_text'] for row in cur.fetchall()]
    morphemes = list({m for m in affix_morphemes + template_morphemes if m})
    morpheme_map = {m: idx for idx, m in enumerate(morphemes)}

    # Senses (definitions) (only for Tagalog words)
    cur.execute("""
        SELECT id, word_id, definition_text, original_pos, standardized_pos_id, examples, usage_notes, tags, sources, definition_metadata, created_at, updated_at
        FROM definitions
        WHERE word_id = ANY(%s);
    """, (list(tagalog_word_ids),))
    senses = cur.fetchall()

    # Languages (all)
    cur.execute("SELECT id, code, name_en, name_tl, region, family, status, created_at, updated_at FROM languages;")
    langs = cur.fetchall()
    lang_map = {l['code']: l['id'] for l in langs if l.get('code') is not None}

    # Parts of Speech (all)
    cur.execute("SELECT id, code, name_en, name_tl, description, created_at, updated_at FROM parts_of_speech;")
    pos_list = cur.fetchall()
    pos_map = {p['id']: p for p in pos_list}

    # --- Edges ---
    # Relations (word-to-word, only between Tagalog words)
    cur.execute("SELECT from_word_id, to_word_id, relation_type FROM relations WHERE from_word_id = ANY(%s) AND to_word_id = ANY(%s);", (list(tagalog_word_ids), list(tagalog_word_ids)))
    rels = cur.fetchall()

    # Word-Form relationships
    has_form = [{"word_id": f["word_id"], "form_id": f["id"]} for f in forms if f["word_id"] is not None]
    of_word = [{"form_id": f["id"], "word_id": f["word_id"]} for f in forms if f["word_id"] is not None]

    # Word-Sense relationships
    has_sense = [{"word_id": s["word_id"], "definition_id": s["id"]} for s in senses if s["word_id"] is not None]

    # Word-Morpheme relationships (from affixations, only for Tagalog words)
    cur.execute("SELECT affixed_word_id as word_id, affix_type FROM affixations WHERE affix_type IS NOT NULL AND affixed_word_id = ANY(%s);", (list(tagalog_word_ids),))
    affix_rels = cur.fetchall()
    has_affix = [{"word_id": a["word_id"], "morpheme_id": morpheme_map.get(a["affix_type"], 0)}
                 for a in affix_rels if a["affix_type"] in morpheme_map]

    # Word-Morpheme relationships (from word_templates, only for Tagalog words)
    cur.execute("SELECT word_id, template_name FROM word_templates WHERE template_name IS NOT NULL AND word_id = ANY(%s);", (list(tagalog_word_ids),))
    template_rels = cur.fetchall()
    has_template = [{"word_id": t["word_id"], "morpheme_id": morpheme_map.get(t["template_name"], 0)}
                   for t in template_rels if t["template_name"] in morpheme_map]
    has_affix += has_template

    # Etymology relations (words with same language code in etymologies, only for Tagalog words)
    cur.execute("SELECT word_id, language_codes FROM etymologies WHERE language_codes IS NOT NULL AND word_id = ANY(%s);", (list(tagalog_word_ids),))
    etyms = cur.fetchall()
    shares_etym = []
    for e in etyms:
        codes = [c.strip() for c in (e["language_codes"] or '').split(',') if c.strip()]
        for code in codes:
            if code != 'tl':
                continue
            for w in words:
                if w["language_code"] == code and w["id"] != e["word_id"]:
                    shares_etym.append({"word1_id": e["word_id"], "word2_id": w["id"]})

    # Phonology relations (words sharing first 2 letters, for demo, only Tagalog words)
    shares_phon = []
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if i < j and w1["lemma"][:2] == w2["lemma"][:2]:
                shares_phon.append({"word1_id": w1["id"], "word2_id": w2["id"]})

    return {
        "words": words,
        "forms": forms,
        "morphemes": [{"id": idx, "morpheme_text": m} for m, idx in morpheme_map.items()],
        "senses": senses,
        "languages": langs,
        "parts_of_speech": pos_list,
        "relations": rels,
        "has_form": has_form,
        "of_word": of_word,
        "has_sense": has_sense,
        "has_affix": has_affix,
        "shares_phon": shares_phon,
        "shares_etym": shares_etym
    }