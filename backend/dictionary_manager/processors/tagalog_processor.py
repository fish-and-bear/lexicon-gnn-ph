#!/usr/bin/env python3
"""
Processor for Tagalog dictionary files (specifically tagalog-words.json format).
"""

import json
import logging
import os
import re
from typing import Tuple, Dict, List, Any, Optional

# Third-party imports (ensure these are installed)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # Optional dependency

# Project-specific imports (using absolute paths)
from backend.dictionary_manager.db_helpers import (
    get_or_create_word_id,
    insert_definition,
    insert_etymology,
    insert_pronunciation,
    insert_definition_example,
    insert_relation, # Needed for inline relation processing
    process_relations_batch, # <-- ADDED import
    get_standardized_pos_id,
    with_transaction,
)
from backend.dictionary_manager.text_helpers import (
    normalize_lemma, 
    SourceStandardization, 
    remove_trailing_numbers,
    get_standard_code,  # ADDED: Import get_standard_code
    standardize_entry_pos, # ADDED: Import standardize_entry_pos (handles lists)
    get_language_code, # Added: Import get_language_code
)
from backend.dictionary_manager.enums import DEFAULT_LANGUAGE_CODE # Ensure DEFAULT_LANGUAGE_CODE is imported

logger = logging.getLogger(__name__)

# Define known affixes (can be expanded)
KNOWN_PREFIXES = {
    "mag", "nag", "pag", "nagpa", "pagpa", "pa", "ma", "na", "ipa", "ipag", "ika", "ikapang", "mala", "mapa", "tag", 
    "maka", "naka", "paka", "pinaka", "sing", "kasing", "magkasing", "magsing", "ka",
    # Added common prefixes:
    "pang", "pinag", "nakipag", "makipag", "pakiki", "taga", "tig", "mang", "nang", "pam", "pan", "mai", "maipa", "makapa",
    "nga" # Added 'nga'
}
KNOWN_SUFFIXES = {"in", "hin", "an", "han", "on", "hon"} # Added 'hon' just in case
KNOWN_INFIXES = {"um", "in"} # Less common to see explicitly with '+' but useful

# --- ADDED: POS Mapping for tagalog-words.json specific POS tags (DATA-DRIVEN) ---
TAGALOG_WORDS_POS_MAP = {
    # Based on unique values extracted from data/tagalog-words.json
    "daglat": "daglat",            # Abbreviation
    "pdd": "pandamdam",          # Interjection (e.g., Abá!)
    "pnb": "pangngalan",         # Noun (likely Pambalana/Common or general)
    "pnd": "pandiwa",            # Verb (Corrected from previous assumption)
    "png": "pangngalan",         # Noun (General)
    "pnh": "panghalip",          # Pronoun
    "pnk": "pang-angkop",        # Linker / Ligature (e.g., -ng, na, -g)
    "pnl": "pandiwa",            # Verb (General or specific type like Pangnilikha)
    "pnr": "pang-uri",           # Adjective
    "pnt": "pangatnig",          # Conjunction
    "pnu": "pang-ukol",          # Preposition
    "ptg": "pangatnig",          # Conjunction (variant/alternative for pnt)
    "ptk": "pantukoy",           # Determiner / Article
    "symbol": "simbolo",         # Symbol

    # --- Previous, more expansive map kept for reference or if other sources use these ---
    # Common Tagalog abbreviations (keys are what might be IN THE JSON)
    # "pnb.": "pangngalan",         # Pangngalan (Noun)
    # "pn.": "pangngalan",          # Pangngalan (Noun)
    # "pnl.": "pandiwa",            # Pandiwa (Verb)
    # "pdw.": "pandiwa",           # Pandiwa (Verb)
    # "pd.": "pandiwa",             # Pandiwa (Verb)
    # "pu.": "pang-uri",            # Pang-uri (Adjective)
    # "png-uri": "pang-uri",        # Pang-uri (Adjective)
    # "pnbb.": "pang-abay",         # Pang-abay (Adverb)
    # "pang abay": "pang-abay",     # Pang-abay (Adverb)
    # "pnh.": "panghalip",          # Panghalip (Pronoun)
    # "pnu.": "pang-ukol",          # Pang-ukol (Preposition)
    # "puuk.": "pang-ukol",         # Pang-ukol (Preposition)
    # "pnt.": "pangatnig",          # Pangatnig (Conjunction)
    # "pngtn.": "pangatnig",        # Pangatnig (Conjunction)
    # "pndm.": "pandamdam",         # Pandamdam (Interjection)
    # "pdm.": "pandamdam",          # Pandamdam (Interjection)
    # "pntk.": "pantukoy",          # Pantukoy (Determiner/Article)
    # "ptk.": "pantukoy",           # Pantukoy (Determiner/Article)
    # "pnm.": "pamilang",           # Pamilang (Numeral)
    # "pml.": "pamilang",           # Pamilang (Numeral)
    # "prl.": "parirala",           # Parirala (Phrase/Expression)
    # "dag.": "daglat",             # Daglat (Abbreviation)
    # "sl.": "salitang lansangan",  # Salitang Lansangan (Slang)
    # "s.u.": "salitang ugat",      # Salitang Ugat (Root word)
    # "panl.": "panlapi",           # Panlapi (Affix)
    # "unl.": "unlapi",             # Unlapi (Prefix)
    # "gitl.": "gitlapi",           # Gitlapi (Infix)
    # "gtl.": "gitlapi",            # Gitlapi (Infix)
    # "hul.": "hulapi",             # Hulapi (Suffix)

    # Full Tagalog terms (values are the canonical Tagalog POS name)
    # "pangngalan": "pangngalan",
    # "pangalan": "pangngalan",      # Common variant/misspelling of Pangngalan
    # "pandiwa": "pandiwa",
    # "pang-uri": "pang-uri",
    # "pang-abay": "pang-abay",
    # "panghalip": "panghalip",
    # "pang-ukol": "pang-ukol",
    # "pangatnig": "pangatnig",
    # "pandamdam": "pandamdam",
    # "pantukoy": "pantukoy",
    # "pamilang": "pamilang",
    # "parirala": "parirala",
    # "balbal": "balbal",            # Standard Tagalog term for Slang
    # "salitang lansangan": "balbal", # Maps to the standard term "balbal"
    # "salitang-ugat": "salitang ugat",
    # "salitang ugat": "salitang ugat",
    # "panlapi": "panlapi",
    # "unlapi": "unlapi",
    # "gitlapi": "gitlapi",
    # "hulapi": "hulapi",

    # English terms that might be IN THE JSON (mapping them to Tagalog equivalents)
    # "noun": "pangngalan",
    # "verb": "pandiwa",
    # "adjective": "pang-uri",
    # "adj.": "pang-uri",
    # "adverb": "pang-abay",
    # "adv.": "pang-abay",
    # "pronoun": "panghalip",
    # "prn.": "panghalip",
    # "preposition": "pang-ukol",
    # "prep.": "pang-ukol",
    # "conjunction": "pangatnig",
    # "conj.": "pangatnig",
    # "interjection": "pandamdam",
    # "interj.": "pandamdam",
    # "determiner": "pantukoy",
    # "det.": "pantukoy",
    # "article": "pantukoy", 
    # "art.": "pantukoy",
    # "numeral": "pamilang",
    # "num.": "pamilang",
    # "phrase": "parirala",
    # "phr.": "parirala",
    # "abbreviation": "daglat",
    # "abbr.": "daglat",
    # "root": "salitang ugat",
    # "root word": "salitang ugat",
    # "affix": "panlapi",
    # "prefix": "unlapi",
    # "infix": "gitlapi",
    # "suffix": "hulapi",
}
# --- END ADDED: POS Mapping ---

def clean_and_extract_pronunciation_guide(text: str, is_already_pronunciation_field: bool = False) -> Tuple[str, Optional[str]]:
    """
    Cleans a string by removing middle dots (·) unless it's already a pronunciation field.
    Returns the cleaned string and the original string if it contained dots (as a guide),
    or None for the guide if no dots were present or if it's already a pronunciation field
    and dots should be preserved.

    Args:
        text: The input string.
        is_already_pronunciation_field: If True, dots are preserved in the "cleaned" output,
                                        and the guide will be None (as it's already the guide).

    Returns:
        A tuple: (cleaned_text: str, pronunciation_guide: Optional[str])
    """
    if not text:
        return "", None

    # For fields explicitly marked as pronunciation, we preserve the dots in the "cleaned" version
    # and do not offer a separate guide, as the input is already the guide.
    if is_already_pronunciation_field:
        # We assume if it's a pronunciation field, it's already in the desired format.
        # No cleaning of dots is done, and no separate guide is returned.
        return text, None # text is returned as is, no separate guide

    # For other fields, we clean the dots.
    # If dots were present, the original text becomes the pronunciation guide.
    if '·' in text:
        cleaned_text = text.replace('·', '')
        pronunciation_guide = text # The original text with dots is the guide
        return cleaned_text, pronunciation_guide
    else:
        # No dots found, so no cleaning needed, and no guide to extract.
        return text, None

# Moved from dictionary_manager.py (originally around line 97 in old structure)
def process_definition_relations(cur, word_id: int, definition: str, source: str):
    """Process and create relationships from definition text."""
    synonym_patterns = [
        r"ka(singkahulugan|tulad|patulad|tumbas) ng\\s+(\\w+)",
        r"singkahulugan:\\s+(\\w+)",
    ]
    antonym_patterns = [
        r"(kasalungat|kabaligtaran) ng\\s+(\\w+)",
        r"kabaligtaran:\\s+(\\w+)",
    ]

    for pattern in synonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            syn = match.group(2).strip()
            syn_id = get_or_create_word_id(cur, syn, "tl")
            insert_relation(cur, word_id, syn_id, "synonym", source_identifier=source) # Use source_identifier

    for pattern in antonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            ant = match.group(2).strip()
            ant_id = get_or_create_word_id(cur, ant, "tl")
            insert_relation(cur, word_id, ant_id, "antonym", source_identifier=source) # Use source_identifier

# Moved from dictionary_manager.py (originally around line 852)
def _process_single_tagalog_word_entry(
    cur,  # <-- ADDED: Explicit cursor is now required
    entry_data: dict,
    word_key: str,
    source_identifier: str,
    language_code: str,  # Should be 'tl' for this processor
    add_etymology: bool,
    word_id_cache: dict,
    stats: dict,
    error_types: dict,  # Pass error_types to log specific errors
    table_column_map: dict,
    relations_batch: list,
    required_columns: dict,
):
    """
    Processes a single word entry from tagalog-words.json format using the provided cursor.
    Assumes it's running within a transaction managed by the caller.
    Raises exceptions on failure, which should be caught by the caller.
    Handles structure observed: top-level word, pronunciation, etymology obj, domains, senses list.
    Sense structure includes: definition, part_of_speech, example obj (raw, examples), references.
    """
    original_lemma_from_json = entry_data.get("word", word_key).strip()
    if not original_lemma_from_json:
        logger.warning(
            f"Skipping entry with missing/empty lemma (original key: '{word_key}')"
        )
        stats["skipped"] += 1
        return

    # Clean the lemma and extract pronunciation guide
    cleaned_lemma, pron_guide_for_lemma = clean_and_extract_pronunciation_guide(original_lemma_from_json)

    if not cleaned_lemma: # If cleaning results in an empty lemma
        logger.warning(
            f"Skipping entry because lemma '{original_lemma_from_json}' became empty after cleaning (original key: '{word_key}')"
        )
        stats["skipped"] += 1
        return

    # --- Determine Language Code based on Etymology (NEW LOGIC) ---
    determined_language_code = "tl" # Default to Tagalog
    etymology_obj = entry_data.get("etymology", {})
    if isinstance(etymology_obj, dict) and etymology_obj:
        etymology_languages = etymology_obj.get("languages", [])
        if isinstance(etymology_languages, list) and len(etymology_languages) == 1:
            single_lang = etymology_languages[0]
            if isinstance(single_lang, str) and single_lang.strip().upper() != "ST":
                # Standardize the found language code
                standard_code, _ = get_language_code(single_lang.strip()) # Uses text_helpers.get_language_code
                # Use the standardized code only if it's valid and not the default 'tl' itself
                if standard_code and standard_code != "unc":
                    determined_language_code = standard_code
                    if determined_language_code != "tl": # Log only if it changed from default
                        logger.debug(f"Using language code '{determined_language_code}' based on single etymology language '{single_lang}' for lemma '{cleaned_lemma}'")
                else:
                    logger.debug(f"Single etymology language '{single_lang}' for lemma '{cleaned_lemma}' standardized to invalid/default '{standard_code}'. Keeping default 'tl'.")

    # --- Extract Top-Level Word Information ---
    top_level_pos_code = None
    top_level_pos_list = entry_data.get("part_of_speech", [])
    # --- ADDED: Set of known invalid codes mistakenly used as POS ---
    INVALID_POS_CODES = {'tl', 'en', 'es', 'la'} # Add others if found

    if isinstance(top_level_pos_list, list) and top_level_pos_list:
        # Expects format like [["pnl"]] or [["png"]]
        if isinstance(top_level_pos_list[0], list) and top_level_pos_list[0]:
            if (
                isinstance(top_level_pos_list[0][0], str)
                and top_level_pos_list[0][0].strip()
            ):
                potential_pos_code_raw = top_level_pos_list[0][0].strip()
                # --- ADDED: Check if extracted code is invalid ---
                if potential_pos_code_raw.lower() not in INVALID_POS_CODES:
                    # First, try to map using our source-specific TAGALOG_WORDS_POS_MAP
                    top_level_pos_code = TAGALOG_WORDS_POS_MAP.get(potential_pos_code_raw.lower(), potential_pos_code_raw)
                else:
                    logger.warning(f"[Tagalog:{cleaned_lemma}] Found invalid code '{potential_pos_code_raw}' in top-level part_of_speech field. Treating as missing POS.")

    # --- Initial Word Insertion/Retrieval ---
    # Collect potential metadata to pass to get_or_create_word_id or update later
    word_creation_metadata = {"source_file": source_identifier}
    top_domains = entry_data.get("domains", [])  # Extract domains here for later use
    # REMOVED: Option 1 (adding domains to metadata) was here.

    # Add other top-level flags if they exist in the JSON (e.g., is_proper, is_abbr)
    # word_creation_metadata['is_proper_noun'] = entry_data.get('is_proper', False)

    # Note: Simplified call for now, only passing metadata. Add other args as needed.
    word_id = get_or_create_word_id(
        cur,
        cleaned_lemma, # Use the cleaned lemma
        determined_language_code,
        source_identifier,
        word_metadata=word_creation_metadata, # Pass dict directly, Json() is handled internally
        # Pass other flags if extracted: has_baybayin, baybayin_form, etc.
    )
    if not word_id:
        # Error logged by get_or_create_word_id
        stats["errors"] += 1
        error_types["WordCreationFailed"] = error_types.get("WordCreationFailed", 0) + 1
        raise ValueError(f"Failed to get or create word_id for cleaned lemma: {cleaned_lemma} (original: {original_lemma_from_json})")

    # --- Add pronunciation guide for the main lemma if extracted ---
    if pron_guide_for_lemma:
        pron_id_lemma = insert_pronunciation(
            cur,
            word_id,
            "respelling_guide",
            pron_guide_for_lemma,
            source_identifier=source_identifier
        )
        if pron_id_lemma:
            stats["pronunciations_added_from_lemma"] += 1

    normalized_lemma_cache_key = normalize_lemma(cleaned_lemma)
    # Use determined_language_code for the cache key, as it's used for word_id creation.
    cache_key = f"{normalized_lemma_cache_key}|{determined_language_code}"
    word_id_cache[cache_key] = word_id

    # --- Store Top-Level Domains in Tags Column (Option 2) ---
    if isinstance(top_domains, list) and top_domains:
        cleaned_domains = [
            str(d).strip() for d in top_domains if d and isinstance(d, str)
        ]
        cleaned_domains = [
            d for d in cleaned_domains if d
        ]  # Ensure not empty after stripping
        if cleaned_domains:
            # Append domains to existing tags (comma-separated)
            domains_str = ",".join(cleaned_domains)
            try:
                # Use COALESCE to handle NULL tags and append with a comma
                # Assume 'tags' column exists based on schema definition
                cur.execute(
                    """UPDATE words
                        SET tags = COALESCE(tags || ',', '') || %s
                        WHERE id = %s""",
                    (domains_str, word_id),
                )
                logger.debug(
                    f"Appended domains {cleaned_domains} to tags for word ID {word_id}"
                )

            except Exception as tag_update_err:
                logger.warning(
                    f"Failed to update tags with domains for word ID {word_id}: {tag_update_err}"
                )

    # --- Process Top-Level Pronunciations ---
    # Uses 'pronunciation' and 'alternate_pronunciation' keys
    main_pron = entry_data.get("pronunciation")
    alt_pron = entry_data.get("alternate_pronunciation")

    if main_pron and isinstance(main_pron, str) and main_pron.strip():
        # Assume simple IPA string for now, could be enhanced to parse structure if needed
        # Use dict format for insert_pronunciation consistency
        pron_id = insert_pronunciation(
            cur, word_id, "ipa", main_pron.strip(), source_identifier=source_identifier
        )
        if pron_id:
            stats["pronunciations_added"] += 1

    if alt_pron and isinstance(alt_pron, str) and alt_pron.strip():
        # Treat alternate as a separate entry, maybe tag it?
        pron_id = insert_pronunciation(
            cur, word_id, "ipa", alt_pron.strip(), tags=["alternate"], source_identifier=source_identifier
        )
        if pron_id:
            stats["pronunciations_added"] += 1

    # --- Process Etymology ---
    etymology_obj = entry_data.get("etymology", {})
    if add_etymology and isinstance(etymology_obj, dict) and etymology_obj:
        etymology_text = etymology_obj.get("raw", "").strip()
        lang_codes = etymology_obj.get("languages", [])  # List like ["Esp"]
        # Convert list of codes to comma-separated string if needed by insert_etymology
        lang_codes_str = (
            ",".join(lc for lc in lang_codes if lc and isinstance(lc, str))
            if lang_codes
            else None
        )

        if etymology_text:
            ety_id = insert_etymology(
                cur,
                word_id,
                etymology_text,
                source_identifier,
                language_codes=lang_codes_str,
            )
            if ety_id:
                stats["etymologies_added"] += 1

        # --- NEW: Call improved etymology term processor ---
        _process_etymology_terms_improved(
            cur, word_id, etymology_obj, relations_batch, source_identifier
        )

    # --- Process Top-Level Derivative ---
    derivative_raw = entry_data.get("derivative")
    if derivative_raw and isinstance(derivative_raw, str) and derivative_raw.strip():
        derived_parts_raw = [part.strip() for part in derivative_raw.split(',') if part.strip()]
        TRAILING_TAGS = {"Med", "Fig", "Lit", "Bot", "Zool", "Chem"}

        for part_raw in derived_parts_raw:
            original_part_for_log = part_raw
            cleaned_part_for_relation = part_raw
            identified_tag = None
            pron_guide_for_derivative = None # Initialize

            potential_tag_match = re.search(r"\s([A-Z][a-zA-Z]{1,3})$", cleaned_part_for_relation)
            if potential_tag_match:
                potential_tag = potential_tag_match.group(1)
                if potential_tag in TRAILING_TAGS:
                    identified_tag = potential_tag
                    cleaned_part_for_relation = cleaned_part_for_relation[:potential_tag_match.start()].strip()
            
            # Clean middle dots and extract guide for the part meant for relation
            cleaned_part_for_relation, pron_guide_for_derivative = clean_and_extract_pronunciation_guide(cleaned_part_for_relation)

            if '·' in cleaned_part_for_relation: # This check might be redundant if helper works perfectly
                cleaned_part_for_relation = cleaned_part_for_relation.replace('·', '')
                # If pron_guide_for_derivative wasn't set by helper, use original_part_for_log if it had dots
                if not pron_guide_for_derivative and '·' in original_part_for_log:
                    pron_guide_for_derivative = original_part_for_log

            if cleaned_part_for_relation and (re.search(r"[a-z-]", cleaned_part_for_relation) or len(cleaned_part_for_relation) > 3):
                relation_metadata = {"context": "derivative"}
                if identified_tag:
                    relation_metadata["tag"] = identified_tag
                if pron_guide_for_derivative:
                    relation_metadata["pronunciation_guide_for_to_word"] = pron_guide_for_derivative

                relations_batch.append({
                            "from_word": word_id,
                    "to_word": cleaned_part_for_relation,
                    "relation_type": "root_of",
                            "source": source_identifier,
                    "metadata": relation_metadata,
                            "def_id": None,
                })
            else:
                logger.debug(f"Ignoring non-word derivative part '{original_part_for_log}' (cleaned: '{cleaned_part_for_relation}') for word ID {word_id}")

    # --- Process Senses / Definitions ---
    senses = entry_data.get("senses", [])
    if isinstance(senses, list):
        for sense_idx, sense in enumerate(senses):
            if not isinstance(sense, dict):
                logger.warning(
                    f"Skipping non-dict sense at index {sense_idx} for '{cleaned_lemma}'"
                )
                continue

            # --- Definition Text ---
            definition_text = sense.get("definition", "").strip()
            if not definition_text:
                logger.debug(
                    f"Skipping sense {sense_idx} for '{cleaned_lemma}' due to missing/empty definition text."
                )
                continue

            # --- Part of Speech ---
            sense_pos_code = None
            sense_pos_list = sense.get("part_of_speech", [])
            if isinstance(sense_pos_list, list) and sense_pos_list:
                if isinstance(sense_pos_list[0], list) and sense_pos_list[0]:
                    if (
                        isinstance(sense_pos_list[0][0], str)
                        and sense_pos_list[0][0].strip()
                    ):
                        potential_sense_pos_raw = sense_pos_list[0][0].strip()
                        # --- ADDED: Check if extracted code is invalid ---
                        if potential_sense_pos_raw.lower() not in INVALID_POS_CODES:
                            # First, try to map using our source-specific TAGALOG_WORDS_POS_MAP
                            sense_pos_code = TAGALOG_WORDS_POS_MAP.get(potential_sense_pos_raw.lower(), potential_sense_pos_raw)
                        else:
                            logger.warning(f"[Tagalog:{cleaned_lemma}] Found invalid code '{potential_sense_pos_raw}' in sense {sense_idx} part_of_speech field. Treating as missing POS for this sense.")

            # Use sense POS if available, otherwise fallback to top-level POS
            pos_code_to_use = sense_pos_code if sense_pos_code else top_level_pos_code
            if not pos_code_to_use:
                logger.warning(
                    f"Missing POS for sense {sense_idx} of '{cleaned_lemma}' (and no top-level POS). Skipping definition."
                )
                continue # Must have a POS to insert definition

            # --- Apply POS Mapping using text_helpers ---
            # pos_code_to_use will be a string like "pnl." or "pangngalan"
            # standardize_entry_pos can handle lists, but here pos_code_to_use is a string
            mapped_pos_code = get_standard_code(pos_code_to_use)
            if pos_code_to_use and mapped_pos_code == "unc" and pos_code_to_use.lower() != "unc":
                logger.debug(f"POS '{pos_code_to_use}' for '{cleaned_lemma}' (sense {sense_idx}) mapped to 'unc' by get_standard_code.")
            # --- End POS Mapping ---

            # Get standardized POS ID (this function now internally calls get_standard_code if needed, 
            # but we've already standardized it, so it should be a direct lookup or simple validation)
            standardized_pos_id = get_standardized_pos_id(cur, mapped_pos_code)

            # --- Extract Usage Notes ---
            sense_notes = sense.get("notes") # Could be list or string
            usage_notes_str = None
            if isinstance(sense_notes, list):
                 usage_notes_str = "; ".join(str(n) for n in sense_notes if n)
            elif isinstance(sense_notes, str):
                 usage_notes_str = sense_notes.strip()

            # Create metadata dictionary
            metadata_dict = {}
            if mapped_pos_code:
                metadata_dict["original_pos"] = mapped_pos_code
            if usage_notes_str:
                 metadata_dict["usage_notes"] = usage_notes_str

            # --- NEW: Process Sense-Level Etymology ---
            sense_etymology_obj = sense.get("etymology")
            if isinstance(sense_etymology_obj, dict) and sense_etymology_obj:
                sense_ety_raw = sense_etymology_obj.get("raw", "").strip()
                sense_ety_langs = sense_etymology_obj.get("languages", [])
                if sense_ety_raw:
                    metadata_dict["sense_etymology_raw"] = sense_ety_raw
                if sense_ety_langs and isinstance(sense_ety_langs, list):
                     # Store list directly, can be processed later if needed
                    metadata_dict["sense_etymology_langs"] = [str(l).strip() for l in sense_ety_langs if l and isinstance(l, str)]
                # Extract first language for sense_language_origin (optional, keep existing?)
                if metadata_dict.get("sense_etymology_langs"):
                    first_lang = metadata_dict["sense_etymology_langs"][0]
                    if first_lang:
                         metadata_dict["sense_language_origin"] = first_lang # Overwrites if already set by terms

            # Pass sense data to insert_definition
            definition_id = insert_definition(
                cur,
                word_id,
                definition_text,
                part_of_speech=mapped_pos_code, # Use the mapped POS code from get_standard_code
                usage_notes=usage_notes_str if usage_notes_str else None,
                metadata=metadata_dict if metadata_dict else None,
                sources=source_identifier,  # Changed from source_identifier to sources
            )

            if not definition_id:
                # Error logged by insert_definition
                stats["errors"] += 1
                error_types["DefinitionInsertionFailed"] = error_types.get("DefinitionInsertionFailed", 0) + 1
                logger.error(f"Failed to insert definition for sense {sense_idx} of '{cleaned_lemma}'. Skipping related data.")
                continue # Skip examples/references if definition failed

            stats["definitions_added"] += 1

            # --- Process Definition Examples ---
            # Examples structure: {"raw": "...", "examples": ["...", "..."]}
            examples_obj = sense.get("examples", {})
            if isinstance(examples_obj, dict):
                example_texts = []
                raw_example = examples_obj.get("raw")
                if raw_example and isinstance(raw_example, str) and raw_example.strip():
                    example_texts.append(raw_example.strip())

                # Append examples from the list if they exist
                example_list = examples_obj.get("examples", [])
                if isinstance(example_list, list):
                     for ex_item in example_list:
                          if ex_item and isinstance(ex_item, str) and ex_item.strip():
                               example_texts.append(ex_item.strip())

                for example_text in example_texts:
                    example_id = insert_definition_example(
                        cur,
                        definition_id,
                        {"text": example_text}, # Simple format for now
                        source_identifier,
                    )
                    if example_id:
                        stats["examples_added"] += 1

            # --- Process Definition References ---
            references_raw = sense.get("references", [])
            if isinstance(references_raw, list):
                for ref_text_raw in references_raw:
                    if ref_text_raw and isinstance(ref_text_raw, str) and ref_text_raw.strip():
                        cleaned_ref_text, pron_guide_for_ref = clean_and_extract_pronunciation_guide(ref_text_raw.strip())
                        if cleaned_ref_text:
                            ref_metadata = {"context": "reference", "definition_id": definition_id}
                            if pron_guide_for_ref:
                                ref_metadata["pronunciation_guide_for_to_word"] = pron_guide_for_ref
                            relations_batch.append({
                                "from_word": word_id,
                                "to_word": cleaned_ref_text,
                                "relation_type": "related",
                                "source": source_identifier,
                                "metadata": ref_metadata,
                                "def_id": definition_id,
                            })

            # --- NEW: Process Definition Synonyms ---
            synonyms_raw = sense.get("synonyms", [])
            if isinstance(synonyms_raw, list):
                for syn_text_raw in synonyms_raw:
                    if syn_text_raw and isinstance(syn_text_raw, str) and syn_text_raw.strip():
                        cleaned_syn_for_relation = remove_trailing_numbers(syn_text_raw.strip())
                        cleaned_syn_for_relation, pron_guide_for_syn = clean_and_extract_pronunciation_guide(cleaned_syn_for_relation)
                        
                        if cleaned_syn_for_relation:
                            syn_metadata = {"context": "definition_sense", "definition_id": definition_id}
                            if pron_guide_for_syn:
                                syn_metadata["pronunciation_guide_for_to_word"] = pron_guide_for_syn
                        relations_batch.append({
                            "from_word": word_id,
                                "to_word": cleaned_syn_for_relation,
                            "relation_type": "synonym",
                            "source": source_identifier,
                                "metadata": syn_metadata,
                            "def_id": definition_id,
                        })
                        else:
                            logger.debug(f"Synonym '{syn_text_raw}' became empty after cleaning for word ID {word_id}, sense {sense_idx}.")

            # --- NEW: Process Definition Affix Forms ---
            affix_forms_raw = sense.get("affix_forms", [])
            if isinstance(affix_forms_raw, list):
                for idx, form_text_raw_from_list in enumerate(affix_forms_raw):
                    if form_text_raw_from_list and isinstance(form_text_raw_from_list, str):
                        form_text_for_processing = form_text_raw_from_list.strip().rstrip('.?!,;')
                        cleaned_form_text, pron_guide_for_affix = clean_and_extract_pronunciation_guide(form_text_for_processing)

                        if cleaned_form_text: 
                        form_metadata = {"context": "definition_sense_affix", "definition_id": definition_id}
                            if pron_guide_for_affix:
                                form_metadata["pronunciation_guide_for_to_word"] = pron_guide_for_affix

                        relations_batch.append({
                                "from_word": word_id,
                                "to_word": cleaned_form_text,
                                "relation_type": "derived",
                            "source": source_identifier,
                            "metadata": form_metadata,
                                "def_id": definition_id,
                        })
                            # Logging of potential pronunciation guide already handled by the previous edit for affix_forms
                            # We just ensure it's correctly added to metadata here.
                        else:
                             logger.debug(f"Affix form '{form_text_raw_from_list}' became empty after cleaning for word ID {word_id}, sense {sense_idx}.")

            # --- Process Sense Domains ---
            # Add sense domains to definition tags or metadata if needed
            # sense_domains = sense.get("domains", [])
            # if sense_domains and isinstance(sense_domains, list):
            #     # Logic to add domains to definition record (e.g., update metadata JSONB)

    else:
        logger.warning(
            f"Expected list for 'senses' in '{cleaned_lemma}', got {type(senses)}. Skipping definitions."
        )

    # --- Process Word Forms and Templates (if keys exist) ---
    # Add logic similar to Kaikki processor here if 'forms' or 'inflection_templates' keys are found
    # Example:
    # if "forms" in entry_data and isinstance(entry_data["forms"], list):
    #    for form_data in entry_data["forms"]:
    #        # Extract form_text, metadata
    #        # Call insert_word_form(cur, word_id, form_text, metadata, source_identifier)
    # if "inflection_templates" in entry_data and ...:
    #    # Call insert_word_template(...)

    # Note: No explicit return, function mutates stats, error_types, relations_batch
    # Exceptions are raised on critical errors to be caught by the caller

# --- MOVED & UPDATED: Helper function for root finding from '+' parts ---
def find_root_from_parts(parts_raw: List[str], cur, determined_language_code: str, source_identifier: str, stats: Optional[Dict] = None):
    """Applies the heuristic to find the root from parts split by '+'.
    It now also attempts to extract and add pronunciation for the identified root if its original part contained dots.
    Args:
        parts_raw: List of original string parts (before dot cleaning).
        cur: Database cursor.
        determined_language_code: Language code for word creation/lookup.
        source_identifier: Source identifier for new entries.
        stats: Optional statistics dictionary to update.
    Returns:
        Tuple (root_word_cleaned, affixes_cleaned, remaining_for_log)
    """
    if stats is None:
        stats = {} # Initialize if not provided, to avoid errors on stats.get

    # Clean each part and store its potential pronunciation guide
    cleaned_parts_with_guides = []
    for p_raw in parts_raw:
        cleaned_p, guide_p = clean_and_extract_pronunciation_guide(p_raw)
        cleaned_parts_with_guides.append((cleaned_p if cleaned_p else p_raw, guide_p)) # Use raw if cleaning makes it empty
    
    # Work with cleaned parts for root/affix identification logic
    potential_root_parts_tuples = list(cleaned_parts_with_guides) # List of (cleaned_part, guide)
    
    identified_prefixes_cleaned = []
    identified_suffixes_cleaned = []

    # Identify prefixes (from cleaned parts)
    while potential_root_parts_tuples and potential_root_parts_tuples[0][0] in KNOWN_PREFIXES:
        prefix_tuple = potential_root_parts_tuples.pop(0)
        identified_prefixes_cleaned.append(prefix_tuple[0])
        # Pronunciation for affixes themselves is not typically stored as they aren't full words.

    # Identify suffixes (from cleaned parts)
    while potential_root_parts_tuples and potential_root_parts_tuples[-1][0] in KNOWN_SUFFIXES:
        suffix_tuple = potential_root_parts_tuples.pop(-1)
        identified_suffixes_cleaned.insert(0, suffix_tuple[0])

    root_word_cleaned = None
    pron_guide_for_root = None
    remaining_for_log_cleaned = []
    num_remaining = len(potential_root_parts_tuples)

    if num_remaining == 1:
        root_word_cleaned, pron_guide_for_root = potential_root_parts_tuples[0]
    elif num_remaining == 2:
        # Assume second part is root (prefix + root pattern)
        # The first part is an assumed prefix, its guide is not directly used for the root.
        assumed_prefix_cleaned, _ = potential_root_parts_tuples[0]
        root_word_cleaned, pron_guide_for_root = potential_root_parts_tuples[1]
        remaining_for_log_cleaned = [assumed_prefix_cleaned]
        logger.debug(f"Assuming second part '{root_word_cleaned}' is root, remaining: {remaining_for_log_cleaned}")
    elif num_remaining > 2:
        # Sort by length of cleaned part to find longest as potential root
        potential_root_parts_tuples.sort(key=lambda x: len(x[0]), reverse=True)
        root_word_cleaned, pron_guide_for_root = potential_root_parts_tuples[0]
        remaining_for_log_cleaned = [pt[0] for pt in potential_root_parts_tuples[1:]]
        logger.warning(f"Multiple ({num_remaining}) potential root parts remain. Assuming longest '{root_word_cleaned}' is root. Remaining parts: {remaining_for_log_cleaned}")

    if root_word_cleaned and pron_guide_for_root:
        try:
            # Get/Create ID for the cleaned root word
            root_word_id = get_or_create_word_id(cur, root_word_cleaned, determined_language_code, source_identifier)
            if root_word_id:
                pron_id = insert_pronunciation(cur, root_word_id, "respelling_guide", pron_guide_for_root, source_identifier=source_identifier)
                if pron_id:
                    stats["pronunciations_added_from_etym_root"] = stats.get("pronunciations_added_from_etym_root", 0) + 1
                    logger.debug(f"Added pronunciation guide '{pron_guide_for_root}' for etymological root '{root_word_cleaned}' (ID: {root_word_id})")
        except Exception as e:
            logger.warning(f"Could not add pronunciation guide for etymological root '{root_word_cleaned}': {e}")

    affixes_cleaned = identified_prefixes_cleaned + identified_suffixes_cleaned
    return root_word_cleaned, affixes_cleaned, remaining_for_log_cleaned

# Note: This function replaces the previous etymology term processing logic within _process_single_tagalog_word_entry
def _process_etymology_terms_improved(
    cur, word_id, etymology_obj, relations_batch, source_identifier
):
    """
    Improved logic to process etymology terms from tagalog-words.json,
    handling both '+' separated terms and the ['root'], ['affix'] structure.
    Also extracts potential pronunciation guides from dot-formatted terms.
    """
    if not isinstance(etymology_obj, dict):
        return

    terms_list = etymology_obj.get("terms", [])
    if not isinstance(terms_list, list):
        return

    for term_data in terms_list:
        if not isinstance(term_data, dict):
            continue

        term_str_raw = term_data.get("term", "").strip()
        term_lang_raw = term_data.get("language", "").strip()
        term_alt_raw = term_data.get("alt", "").strip()

        # Clean the main term and extract potential pronunciation guide
        term_str_cleaned, pron_guide_for_term = clean_and_extract_pronunciation_guide(term_str_raw)

        if not term_str_cleaned:
            continue # Skip if term becomes empty after cleaning

        # Standardize language code for the term
        term_lang_std, _ = get_language_code(term_lang_raw if term_lang_raw else "tl") # Default to 'tl' if lang missing

        relation_metadata = {"context": "etymology_term"}
        if term_lang_raw:
            relation_metadata["original_term_language"] = term_lang_raw
        if term_alt_raw:
            relation_metadata["alternate_form"] = term_alt_raw # Store alt form if present
        if pron_guide_for_term:
            relation_metadata["pronunciation_guide_for_to_word"] = pron_guide_for_term

        # If term_str_cleaned contains '+', it indicates a compound or affixed form
        if "+" in term_str_cleaned: 
            parts = [p.strip() for p in term_str_cleaned.split('+') if p.strip()]
            if len(parts) >= 2:
                # Pass the original parts (before dot cleaning) to find_root_from_parts
                # to allow it to extract pronunciation guides for the root/affixes if it can.
                # This assumes find_root_from_parts is modified to handle this.
                original_parts_for_root_finding = [p.strip() for p in term_str_raw.split('+') if p.strip()]
                root_word_cleaned, affixes_cleaned, _ = find_root_from_parts(original_parts_for_root_finding, cur, determined_language_code=term_lang_std, source_identifier=source_identifier, stats=None) # Pass cur, lang, source, stats is optional here

                if root_word_cleaned:
                    # Add relation to the identified root
                    root_relation_metadata = {**relation_metadata, "derivation_type": "compound/affixed_etymology"}
                    # The pron_guide_for_term (from the full term_str_raw) applies to the *target* of this relation (root_word_cleaned)
                    # if the root_word_cleaned is what term_str_cleaned resolves to. This is complex.
                    # For now, let pron_guide_for_term from the full string be handled by process_relations_batch
                    # if the full term_str_cleaned is added as a relation.

                    relations_batch.append({
                        "from_word": word_id, # The word whose etymology we are processing
                        "to_word": root_word_cleaned, # The identified root from the etymological term
                        "relation_type": "etymological_origin",
                        "source": source_identifier,
                        "metadata": root_relation_metadata,
                        "language_of_to_word": term_lang_std # Language of the etymological root
                    })
                    # Pronunciation for root_word_cleaned itself (if its original part had dots) is handled by find_root_from_parts

                # Add relations for affixes if needed (affixes are strings, not word IDs directly)
                # This part would require affixes to be actual words or a different handling mechanism.
                # For now, affixes are primarily for identifying the root.
            else: # Not enough parts after splitting by '+'
                relations_batch.append({
                    "from_word": word_id,
                    "to_word": term_str_cleaned, # Use the full cleaned term if not splittable into root/affix
                    "relation_type": "etymological_origin",
                    "source": source_identifier,
                    "metadata": relation_metadata,
                    "language_of_to_word": term_lang_std
                })
        else: # Not a '+' separated term, treat as a direct etymological origin
            relations_batch.append({
                "from_word": word_id,
                "to_word": term_str_cleaned,
                "relation_type": "etymological_origin",
                "source": source_identifier,
                "metadata": relation_metadata,
                "language_of_to_word": term_lang_std
            })

    # Process 'root' and 'affix' keys if they exist (less common in tagalog-words.json but good for robustness)
    etym_root_raw = etymology_obj.get("root")
    etym_affixes_raw = etymology_obj.get("affixes", []) # Ensure it's a list

    if etym_root_raw and isinstance(etym_root_raw, str):
        cleaned_etym_root, pron_guide_for_etym_root = clean_and_extract_pronunciation_guide(etym_root_raw.strip())
        if cleaned_etym_root:
            root_metadata = {"context": "etymology_root_key"}
            if pron_guide_for_etym_root:
                root_metadata["pronunciation_guide_for_to_word"] = pron_guide_for_etym_root
            
            relations_batch.append({
                "from_word": word_id,
                "to_word": cleaned_etym_root,
                "relation_type": "etymological_origin", # Or potentially "etymological_root"
                "source": source_identifier,
                "metadata": root_metadata,
                 # Assuming root language is same as main etymology if not specified, or default to tl
                "language_of_to_word": get_language_code(etymology_obj.get("language", "tl"))[0]
            })
    
    if isinstance(etym_affixes_raw, list):
        for affix_raw in etym_affixes_raw:
            if affix_raw and isinstance(affix_raw, str):
                cleaned_affix, pron_guide_for_affix = clean_and_extract_pronunciation_guide(affix_raw.strip())
                if cleaned_affix:
                    # How to relate an affix? It's not usually a word entry itself.
                    # Option 1: Add as a note/metadata to the main word's etymology entry (if ety_id was created)
                    # Option 2: Create a specific relation type if affixes can be target words.
                    # For now, logging it. If affixes are important relations, this needs more thought.
                    logger.debug(f"Found etymological affix: '{cleaned_affix}' (guide: {pron_guide_for_affix}) for word ID {word_id}. Handling TBD.")


# Moved from dictionary_manager.py (originally around line 1311)
def process_tagalog_words(
    cur, filename: str, add_etymology: bool = True
) -> Dict:
    """
    Process a JSON file containing Tagalog words (tagalog-words.json format)
    using the provided cursor and iterative JSON parsing.
    Assumes transaction is managed by the caller.

    Args:
        cur: Active database cursor provided by the caller.
        filename: Path to the JSON file.
        add_etymology: Whether to add etymology data.

    Returns:
        Dictionary with processing statistics.
    """
    logger.info(f"Processing Tagalog words file: {filename} using provided cursor.")

    # --- Initialize Statistics ---
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "definitions_added": 0,
        "pronunciations_added": 0,
        "pronunciations_added_from_lemma": 0, # For guides from main lemma
        "pronunciations_added_from_etym_root": 0, # For guides from etymological roots
        "pronunciations_added_from_relations_metadata": 0, # For guides from relations batch
        "etymologies_added": 0,
        "examples_added": 0,
        "relations_processed": 0,
        "relations_inserted": 0,
        "relations_failed": 0,
    }
    error_types = {}
    word_id_cache = {} # Cache word IDs to reduce lookups during relation processing
    relations_batch = [] # Batch relations for later processing

    # --- Source Identifier ---
    raw_source_identifier = os.path.basename(filename)
    source_identifier = SourceStandardization.standardize_sources(raw_source_identifier)
    if not source_identifier: # Fallback
        source_identifier = "diksiyonaryo.ph" # Default name for this source
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    # --- Required Columns Check (Simplified for this processor) ---
    # Assuming standard schema, no need for column mapping/checking here unless customizing
    table_column_map = {} # Placeholder if needed by _process_single_tagalog_word_entry
    required_columns = {} # Placeholder if needed by _process_single_tagalog_word_entry

    # --- Load JSON Data ---
    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        stats["errors"] += 1
        error_types["FileNotFound"] = 1
        return stats
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        stats["errors"] += 1
        error_types["JSONDecodeError"] = 1
        return stats # Cannot proceed
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        stats["errors"] += 1
        error_types["FileReadError"] = 1
        return stats # Cannot proceed

    if not isinstance(data, dict):
        logger.error(f"File {filename} does not contain a top-level dictionary.")
        # Return stats indicating format error; caller should handle transaction
        stats["errors"] += 1
        error_types["InvalidFormat"] = 1
        return {**stats, "error_details": error_types}

    # --- Optional: Extract top-level metadata if needed ---
    dictionary_metadata = data.pop("__metadata__", {}) # Remove metadata if key exists
    entries_in_file = len(data) # Count actual word entries

    if entries_in_file == 0:
        logger.info(f"Found 0 word entries in {filename}. Skipping file.")
        # No commit needed, just return stats
        return stats # Return empty stats

    logger.info(f"Found {entries_in_file} word entries in {filename}")

    # --- Get connection for savepoints ---
    conn = cur.connection
    if conn.closed:
        logger.error("Connection is closed. Cannot proceed with processing.")
        stats["errors"] += 1
        error_types["ConnectionClosed"] = 1
        return {**stats, "error_details": error_types}

    # --- Process Entries ---
    iterator = data.items()
    pbar = None
    if tqdm:
        pbar = tqdm(iterator, total=entries_in_file, desc=f"Processing {source_identifier}", unit="entry", leave=False)

    try:
        # --- Iterate using data.items() directly, handle tqdm update manually ---
        for entry_index, (word_key, entry_data) in enumerate(data.items()):
            # --- Savepoint logic ---
            savepoint_name = f"tagalog_entry_{entry_index}"
            lemma_for_log = entry_data.get("word", word_key) # Get original lemma for logging
            cleaned_lemma = None # Initialize cleaned_lemma

            try:
                # Create savepoint for this entry
                cur.execute(f"SAVEPOINT {savepoint_name}")

                # Call the single entry processor
                _process_single_tagalog_word_entry(
                    cur,
                    entry_data,
                    word_key,
                    source_identifier,
                    "tl", # Hardcoded language code for this processor
                    add_etymology,
                    word_id_cache,
                    stats, # Pass stats dict for updates
                    error_types, # Pass error_types for updates
                    table_column_map,
                    relations_batch,
                    required_columns,
                )
                # If _process_single_tagalog_word_entry doesn't raise an exception, release savepoint
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                stats["processed"] += 1 # Increment processed count only on success

                # Commit every 100 entries to avoid transaction bloat
                if stats["processed"] % 100 == 0:
                    conn.commit()
                    logger.debug(f"Committed after processing {stats['processed']} entries")

            except Exception as entry_error:
                # Use original lemma for logging if cleaned_lemma is not yet defined or caused the error
                log_lemma_ref = lemma_for_log
                # Try to get cleaned_lemma if available in this scope for more specific error logging
                try:
                    # This is a bit of a hack to get the cleaned_lemma if the error happened after its definition
                    # It assumes 'cleaned_lemma' would be in the local scope if defined.
                    # This won't work if the error is *during* cleaned_lemma definition.
                    if 'cleaned_lemma' in locals() and cleaned_lemma:
                         log_lemma_ref = f"cleaned: {cleaned_lemma} (original: {lemma_for_log})"
                except NameError:
                    pass # cleaned_lemma not defined, stick to lemma_for_log

                logger.error(f"Error processing entry \'{log_lemma_ref}\' (key: {word_key}, index: {entry_index}): {entry_error}", exc_info=True)
                stats["errors"] += 1
                error_key = f"EntryProcessingError: {type(entry_error).__name__}"
                error_types[error_key] = error_types.get(error_key, 0) + 1

                # Rollback to the savepoint to discard changes for this entry
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                except Exception as rb_error:
                    logger.error(f"Failed to rollback/release savepoint {savepoint_name} for entry \'{log_lemma_ref}\': {rb_error}")
                    # If we can't rollback to savepoint, try to rollback the whole transaction
                    try:
                        conn.rollback()
                        logger.info("Performed full transaction rollback after savepoint failure")
                    except Exception as full_rb_error:
                        logger.critical(f"CRITICAL: Failed both savepoint and full rollback: {full_rb_error}")
                        raise # Re-raise to stop processing

            finally:
                # --- Update tqdm manually inside the loop ---
                if pbar:
                    pbar.update(1)

        # --- NEW: Process pronunciation guides collected in relations_batch metadata ---
        if relations_batch:
            logger.info(f"Attempting to add {sum(1 for item in relations_batch if item.get('metadata', {}).get('pronunciation_guide_for_to_word'))} pronunciation guides from relations metadata...")
            pron_guides_processed_count = 0
            for rel_item_idx, rel_item in enumerate(relations_batch):
                metadata = rel_item.get("metadata", {})
                pron_guide = metadata.get("pronunciation_guide_for_to_word")
                to_word_cleaned = rel_item.get("to_word")
                item_source = rel_item.get("source")
                item_lang = rel_item.get("language_of_to_word", DEFAULT_LANGUAGE_CODE)

                if pron_guide and to_word_cleaned and item_source:
                    try:
                        normalized_to_word_for_pron = normalize_lemma(to_word_cleaned)
                        cache_key_for_pron = f"{normalized_to_word_for_pron}|{item_lang}"
                        to_word_id_for_pron = word_id_cache.get(cache_key_for_pron)

                        if not to_word_id_for_pron:
                            # logger.debug(f"Cache miss for '{to_word_cleaned}' (lang: {item_lang}) for pronunciation guide. Querying DB.")
                            to_word_id_for_pron = get_or_create_word_id(cur, to_word_cleaned, item_lang, source_identifier=item_source)
                            if to_word_id_for_pron:
                                word_id_cache[cache_key_for_pron] = to_word_id_for_pron
                            
                            if to_word_id_for_pron:
                                pron_id = insert_pronunciation(
                                    cur, 
                                    to_word_id_for_pron, 
                                    "respelling_guide", 
                                    pron_guide, 
                                    source_identifier=item_source
                                )
                                if pron_id:
                                    stats["pronunciations_added_from_relations_metadata"] += 1
                                    pron_guides_processed_count +=1
                                    # No need to pop from metadata, as process_relations_batch doesn't use it
                            else:
                                logger.warning(f"Could not get/create ID for '{to_word_cleaned}' (lang: {item_lang}) to add pronunciation guide '{pron_guide}'.")
                    except Exception as e:
                        logger.error(f"Error adding pronunciation guide '{pron_guide}' for '{to_word_cleaned}' from relations_batch item {rel_item_idx}: {e}")
            
            if pron_guides_processed_count > 0:
                try:
                    conn.commit()
                    logger.info(f"Committed {pron_guides_processed_count} pronunciation guides from relations metadata.")
                except Exception as commit_err:
                    logger.error(f"Error committing pronunciation guides from relations metadata: {commit_err}")
                    conn.rollback() # Rollback this specific commit if it fails

        # Process the relations batch *after* iterating through all entries and adding metadata pron guides
        if relations_batch:
            try:
                process_relations_batch(cur, relations_batch, stats, word_id_cache)
                conn.commit()
                logger.info("Successfully processed and committed relations batch")
            except Exception as batch_error:
                logger.error(f"Error processing relations batch: {batch_error}", exc_info=True)
                error_types["RelationsBatchError"] = error_types.get("RelationsBatchError", 0) + 1
                try:
                    conn.rollback()
                    logger.info("Rolled back failed relations batch")
                except Exception as rb_error:
                    logger.critical(f"CRITICAL: Failed to rollback relations batch: {rb_error}")
                    raise # Re-raise to stop processing

    except Exception as e:
        logger.critical(f"Critical error during processing: {e}", exc_info=True)
        error_types["CriticalProcessingError"] = error_types.get("CriticalProcessingError", 0) + 1
        try:
            conn.rollback()
            logger.info("Rolled back transaction after critical error")
        except Exception as rb_error:
            logger.critical(f"CRITICAL: Failed to rollback after critical error: {rb_error}")
    finally:
        if pbar:
            pbar.close()
            
    # --- Final Stats ---
    logger.info(f"Finished processing {source_identifier}. Processed: {stats['processed']}, Errors: {stats['errors']}, Skipped: {stats['skipped']}")
    if error_types:
        stats["error_details"] = error_types
    return stats