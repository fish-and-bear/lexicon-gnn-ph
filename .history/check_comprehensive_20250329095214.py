import json
import re
from collections import Counter

# Load the JSON file
with open('data/gay-slang.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

# 1. Check for consistent field structure
field_counts = Counter()
for entry in data:
    for field in entry.keys():
        field_counts[field] += 1

print("\nField distribution:")
for field, count in field_counts.most_common():
    percentage = count / len(data) * 100
    print(f"{field}: {count}/{len(data)} ({percentage:.1f}%)")

# 2. Check for examples with "meaning" instead of "example"
meaning_instead_of_example = []
for i, entry in enumerate(data):
    if "examples" in entry and entry["examples"]:
        for j, example in enumerate(entry["examples"]):
            if "meaning" in example and "example" not in example:
                meaning_instead_of_example.append((i, entry["headword"], j, example))

if meaning_instead_of_example:
    print(f"\nFound {len(meaning_instead_of_example)} examples using 'meaning' instead of 'example':")
    for idx, headword, ex_idx, example in meaning_instead_of_example[:5]:  # Show only first 5
        print(f"Entry {idx} ('{headword}'), example {ex_idx}: {example}")
    if len(meaning_instead_of_example) > 5:
        print(f"... and {len(meaning_instead_of_example) - 5} more")

# 3. Check for empty definitions or examples
empty_content = []
for i, entry in enumerate(data):
    # Check definitions
    if "definitions" in entry:
        for j, definition in enumerate(entry["definitions"]):
            if "meaning" not in definition or definition["meaning"] is None or definition["meaning"].strip() == "":
                empty_content.append((i, entry["headword"], "definition", j, "empty meaning"))
    
    # Check examples
    if "examples" in entry:
        for j, example in enumerate(entry["examples"]):
            if "example" in example and (example["example"] is None or example["example"].strip() == ""):
                empty_content.append((i, entry["headword"], "example", j, "empty example"))
            elif "meaning" in example and (example["meaning"] is None or example["meaning"].strip() == ""):
                empty_content.append((i, entry["headword"], "example", j, "empty meaning field (should be example)"))

if empty_content:
    print(f"\nFound {len(empty_content)} empty content fields:")
    for idx, headword, field_type, j, issue in empty_content[:10]:
        print(f"Entry {idx} ('{headword}'): {field_type} {j} has {issue}")
    if len(empty_content) > 10:
        print(f"... and {len(empty_content) - 10} more")

# 4. Check for inconsistent end punctuation in examples
missing_punctuation = []
extra_punctuation = []

for i, entry in enumerate(data):
    if "examples" in entry and entry["examples"]:
        for j, example in enumerate(entry["examples"]):
            example_text = example.get("example", example.get("meaning", "")).strip()
            
            # Check for ending punctuation
            if example_text and not example_text.endswith((".", "?", "!", ":")):
                missing_punctuation.append((i, entry["headword"], j, example_text))
            
            # Check for Filipino examples with periods but English translations without
            if j > 0 and "language" in example and example["language"] == "english":
                prev_example = entry["examples"][j-1]
                if "language" in prev_example and prev_example["language"] == "filipino":
                    filipino_text = prev_example.get("example", prev_example.get("meaning", "")).strip()
                    english_text = example_text
                    
                    if filipino_text.endswith(".") and not english_text.endswith("."):
                        missing_punctuation.append((i, entry["headword"], j, f"English: '{english_text}' missing period, while Filipino has it"))
                    elif not filipino_text.endswith(".") and english_text.endswith("."):
                        extra_punctuation.append((i, entry["headword"], j, f"English: '{english_text}' has period, while Filipino does not"))

if missing_punctuation:
    print(f"\nFound {len(missing_punctuation)} examples with missing punctuation:")
    for idx, headword, ex_idx, text in missing_punctuation[:5]:
        print(f"Entry {idx} ('{headword}'), example {ex_idx}: {text}")
    if len(missing_punctuation) > 5:
        print(f"... and {len(missing_punctuation) - 5} more")

if extra_punctuation:
    print(f"\nFound {len(extra_punctuation)} examples with inconsistent punctuation between languages:")
    for idx, headword, ex_idx, text in extra_punctuation[:5]:
        print(f"Entry {idx} ('{headword}'), example {ex_idx}: {text}")
    if len(extra_punctuation) > 5:
        print(f"... and {len(extra_punctuation) - 5} more")

# 5. Check for duplicate headwords (entries with the same headword)
headword_indices = {}
for i, entry in enumerate(data):
    headword = entry.get("headword", "")
    if headword:
        if headword in headword_indices:
            headword_indices[headword].append(i)
        else:
            headword_indices[headword] = [i]

duplicates = {hw: indices for hw, indices in headword_indices.items() if len(indices) > 1}
if duplicates:
    print(f"\nFound {len(duplicates)} duplicate headwords:")
    for headword, indices in list(duplicates.items())[:10]:
        print(f"Headword '{headword}' appears at indices: {indices}")
    if len(duplicates) > 10:
        print(f"... and {len(duplicates) - 10} more")

# 6. Check for inconsistent synonym format
invalid_synonyms = []
for i, entry in enumerate(data):
    if "synonym" in entry:
        synonym = entry["synonym"]
        if synonym is None:
            invalid_synonyms.append((i, entry["headword"], "null synonym"))
        elif isinstance(synonym, str):
            invalid_synonyms.append((i, entry["headword"], f"string instead of list: '{synonym}'"))
        elif not isinstance(synonym, list):
            invalid_synonyms.append((i, entry["headword"], f"unexpected type: {type(synonym).__name__}"))

if invalid_synonyms:
    print(f"\nFound {len(invalid_synonyms)} entries with invalid synonym format:")
    for idx, headword, issue in invalid_synonyms:
        print(f"Entry {idx} ('{headword}'): {issue}")

print("\nCheck complete.") 