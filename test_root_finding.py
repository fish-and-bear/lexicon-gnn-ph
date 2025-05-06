import json
import random
import logging
import re
from pathlib import Path

# --- Configuration ---
JSON_FILE_PATH = Path("data/tagalog-words.json")
NUM_SAMPLES = 100000 # Number of random entries to test
LOG_LEVEL = logging.INFO # Change to logging.DEBUG for more detail

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Root Finding Logic (Copied and adapted from tagalog_processor.py) ---
# --- MODIFIED: Expanded affix lists (matching processor) ---
KNOWN_PREFIXES = {
    "mag", "nag", "pag", "nagpa", "pagpa", "pa", "ma", "na", "ipa", "ipag", "ika", "ikapang", "mala", "mapa", "tag",
    "maka", "naka", "paka", "pinaka", "sing", "kasing", "magkasing", "magsing", "ka",
    "pang", "pinag", "nakipag", "makipag", "pakiki", "taga", "tig", "mang", "nang", "pam", "pan", "mai", "maipa", "makapa",
    "nga"
}
KNOWN_SUFFIXES = {"in", "hin", "an", "han", "on", "hon"}
# KNOWN_INFIXES = {"um", "in"} # Not currently used in '+' logic

def find_root_from_parts(parts):
    """Applies the heuristic to find the root from parts split by '+'"""
    identified_prefixes = []
    identified_suffixes = []
    potential_root_parts = list(parts) # Copy parts to modify

    # Identify prefixes
    while potential_root_parts and potential_root_parts[0] in KNOWN_PREFIXES:
        identified_prefixes.append(potential_root_parts.pop(0))

    # Identify suffixes
    while potential_root_parts and potential_root_parts[-1] in KNOWN_SUFFIXES:
        identified_suffixes.insert(0, potential_root_parts.pop(-1)) # Insert at beginning to maintain order

    # --- Determine Root ---
    root_word = None
    remaining_for_log = []
    num_remaining = len(potential_root_parts)

    if num_remaining == 1:
        # Common case: one part remaining is the root
        root_word = potential_root_parts[0]
    elif num_remaining == 2:
        # --- MODIFIED: Assume second part is root if two remain (prefix + root pattern) ---
        root_word = potential_root_parts[1]
        remaining_for_log = [potential_root_parts[0]] # Log the assumed prefix
        logger.debug(f"Assuming second part '{root_word}' is root, remaining: {remaining_for_log}")
    elif num_remaining > 2:
        # --- Keep longest part heuristic, but log clearer warning ---
        potential_root_parts.sort(key=len, reverse=True)
        root_word = potential_root_parts[0] # Take the longest
        remaining_for_log = potential_root_parts[1:]
        logger.warning(f"Multiple ({num_remaining}) potential root parts remain. Assuming longest '{root_word}' is root. Remaining parts: {remaining_for_log}")
    # Else: num_remaining is 0, root_word remains None

    affixes = identified_prefixes + identified_suffixes
    return root_word, affixes, remaining_for_log

# --- MODIFIED: Updated test_etymology_logic to match processor's improved logic ---
def test_etymology_logic(word_key, etymology_obj):
    """Tests the root finding logic for a single etymology object."""
    results = []
    if not isinstance(etymology_obj, dict):
        return results

    terms = etymology_obj.get("terms", [])
    other_terms = etymology_obj.get("other_terms", [])
    processed_terms = set()

    # --- Handle Separated Root/Affixes Case (Matching processor logic) ---
    if len(terms) == 1 and other_terms and isinstance(terms[0], str):
        potential_root_term = terms[0].strip()
        potential_affix_terms = [str(ot).strip() for ot in other_terms if ot and isinstance(ot, str)]
        potential_affix_terms = [a for a in potential_affix_terms if a] # Ensure not empty

        identified_root = None
        identified_affixes = []
        result_type = "separated_default"

        term_is_suffix = potential_root_term in KNOWN_SUFFIXES
        other_terms_contain_non_affix = any(p_aff not in KNOWN_PREFIXES and p_aff not in KNOWN_SUFFIXES for p_aff in potential_affix_terms)

        if term_is_suffix and other_terms_contain_non_affix and len(potential_affix_terms) == 1:
             identified_root = potential_affix_terms[0]
             identified_affixes = [potential_root_term]
             result_type = "separated_swapped_simple"
        elif term_is_suffix and other_terms_contain_non_affix and len(potential_affix_terms) > 1:
             non_affix_others = [ot for ot in potential_affix_terms if ot not in KNOWN_PREFIXES and ot not in KNOWN_SUFFIXES]
             if non_affix_others:
                 non_affix_others.sort(key=len, reverse=True)
                 identified_root = non_affix_others[0]
                 identified_affixes = [potential_root_term] + [ot for ot in potential_affix_terms if ot != identified_root]
                 result_type = "separated_swapped_complex"

        if identified_root is None and potential_root_term and potential_affix_terms:
            identified_root = potential_root_term
            identified_affixes = potential_affix_terms
            result_type = "separated_default"

        if identified_root:
            results.append({
                "type": result_type, # Use the determined type
                "original_term": identified_root, # Show identified root here for consistency
                "root": identified_root,
                "affixes": identified_affixes,
                "remaining": []
            })
            processed_terms.add(identified_root)
            if potential_root_term != identified_root:
                processed_terms.add(potential_root_term)

    # --- Process Remaining Terms (including '+' notation) ---
    all_terms_to_process = [t for t in terms if t not in processed_terms]

    for term_data in all_terms_to_process:
        if isinstance(term_data, str):
            term_str_cleaned = term_data.strip()
            if "+" in term_str_cleaned:
                parts = [p.strip() for p in term_str_cleaned.split("+") if p.strip()]
                if len(parts) >= 2:
                    root, affixes, remaining = find_root_from_parts(parts)
                    results.append({
                        "type": "plus_notation",
                        "original_term": term_str_cleaned,
                        "root": root,
                        "affixes": affixes,
                        "remaining": remaining
                    })
                else:
                     results.append({ # Log invalid '+' term
                        "type": "invalid_plus",
                        "original_term": term_str_cleaned,
                        "root": None,
                        "affixes": [],
                        "remaining": []
                    })
            # else: # Simple term, maybe log as 'related'? Not the focus here.
            #    pass
        # elif isinstance(term_data, dict): # Ignore dict terms for this test
        #    pass

    return results

# --- Main Script Logic ---
def main():
    if not JSON_FILE_PATH.is_file():
        logger.error(f"JSON file not found: {JSON_FILE_PATH}")
        return

    logger.info(f"Loading JSON file: {JSON_FILE_PATH}...")
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return

    if not isinstance(data, dict):
        logger.error("JSON data is not a dictionary (object).")
        return

    logger.info(f"Loaded {len(data)} entries.")

    all_keys = list(data.keys())
    if len(all_keys) < NUM_SAMPLES:
        logger.warning(f"Requested {NUM_SAMPLES} samples, but only {len(all_keys)} entries exist. Testing all.")
        sample_keys = all_keys
    else:
        sample_keys = random.sample(all_keys, NUM_SAMPLES)

    logger.info(f"--- Testing Root Finding on {len(sample_keys)} Random Entries ---")

    for key in sample_keys:
        entry = data.get(key)
        if not entry or not isinstance(entry, dict):
            continue

        etymology_obj = entry.get("etymology")
        if etymology_obj:
            print(f"\n--- Word: {key} ---")
            print(f"Etymology Object: {etymology_obj}")
            test_results = test_etymology_logic(key, etymology_obj)
            if test_results:
                print("Analysis Results:")
                for result in test_results:
                    print(f"  - Type: {result['type']}")
                    print(f"    Original: '{result['original_term']}'")
                    print(f"    Identified Root: '{result['root']}'")
                    print(f"    Identified Affixes: {result['affixes']}")
                    if result['remaining']:
                        print(f"    Remaining Parts: {result['remaining']}")
            else:
                print("Analysis Results: No relevant terms found for root analysis.")

if __name__ == "__main__":
    main() 