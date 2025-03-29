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

# 3. Check for empty or null values in important fields
empty_or_null_values = []
for i, entry in enumerate(data):
    # Check definitions
    if "definitions" in entry:
        for j, definition in enumerate(entry["definitions"]):
            if "meaning" not in definition:
                empty_or_null_values.append((i, entry["headword"], "definition", j, "missing meaning field"))
            elif definition["meaning"] is None:
                empty_or_null_values.append((i, entry["headword"], "definition", j, "null meaning"))
            elif isinstance(definition["meaning"], str) and definition["meaning"].strip() == "":
                empty_or_null_values.append((i, entry["headword"], "definition", j, "empty meaning"))
    else:
        empty_or_null_values.append((i, entry["headword"], "entry", 0, "missing definitions array"))
    
    # Check headword
    if "headword" not in entry or entry["headword"] is None or (isinstance(entry["headword"], str) and entry["headword"].strip() == ""):
        empty_or_null_values.append((i, "UNKNOWN", "entry", 0, "missing or empty headword"))
    
    # Check partOfSpeech
    if "partOfSpeech" not in entry:
        empty_or_null_values.append((i, entry.get("headword", "UNKNOWN"), "entry", 0, "missing partOfSpeech"))
    elif entry["partOfSpeech"] is None:
        empty_or_null_values.append((i, entry["headword"], "entry", 0, "null partOfSpeech"))
    
    # Check other fields with null values
    for field in ["synonym", "sangkahulugan", "etymology", "usageLabels"]:
        if field in entry and entry[field] is None:
            empty_or_null_values.append((i, entry["headword"], "entry", 0, f"null {field}"))

if empty_or_null_values:
    print(f"\nFound {len(empty_or_null_values)} instances of empty or null values:")
    for idx, headword, section, j, issue in empty_or_null_values[:10]:
        print(f"Entry {idx} ('{headword}'): {section} {j} has {issue}")
    if len(empty_or_null_values) > 10:
        print(f"... and {len(empty_or_null_values) - 10} more")

# 4. Check for inconsistent array formats
inconsistent_arrays = []
array_fields = ["variations", "usageLabels", "synonym"]
for i, entry in enumerate(data):
    for field in array_fields:
        if field in entry and entry[field] is not None:
            if not isinstance(entry[field], list):
                inconsistent_arrays.append((i, entry["headword"], field, f"expected list but got {type(entry[field]).__name__}"))

if inconsistent_arrays:
    print(f"\nFound {len(inconsistent_arrays)} instances of inconsistent array formats:")
    for idx, headword, field, issue in inconsistent_arrays:
        print(f"Entry {idx} ('{headword}'): {field} - {issue}")

# 5. Check for language inconsistencies in definitions and examples
language_issues = []
for i, entry in enumerate(data):
    # Check definitions
    if "definitions" in entry:
        has_filipino = False
        has_english = False
        for j, definition in enumerate(entry["definitions"]):
            if "language" not in definition:
                language_issues.append((i, entry["headword"], "definition", j, "missing language field"))
            elif definition["language"] not in ["filipino", "english"]:
                language_issues.append((i, entry["headword"], "definition", j, f"invalid language: '{definition['language']}'"))
            elif definition["language"] == "filipino":
                has_filipino = True
            elif definition["language"] == "english":
                has_english = True
        
        # Check if both languages are present
        if not has_filipino:
            language_issues.append((i, entry["headword"], "definitions", 0, "missing Filipino definition"))
        if not has_english:
            language_issues.append((i, entry["headword"], "definitions", 0, "missing English definition"))
    
    # Check examples
    if "examples" in entry and entry["examples"]:
        has_filipino_example = False
        has_english_example = False
        for j, example in enumerate(entry["examples"]):
            if "language" not in example:
                language_issues.append((i, entry["headword"], "example", j, "missing language field"))
            elif example["language"] not in ["filipino", "english"]:
                language_issues.append((i, entry["headword"], "example", j, f"invalid language: '{example['language']}'"))
            elif example["language"] == "filipino":
                has_filipino_example = True
            elif example["language"] == "english":
                has_english_example = True
        
        # Check if both languages are present in examples
        if not has_filipino_example:
            language_issues.append((i, entry["headword"], "examples", 0, "missing Filipino example"))
        if not has_english_example:
            language_issues.append((i, entry["headword"], "examples", 0, "missing English example"))

if language_issues:
    print(f"\nFound {len(language_issues)} language-related issues:")
    for idx, headword, section, j, issue in language_issues[:10]:
        print(f"Entry {idx} ('{headword}'): {section} {j} has {issue}")
    if len(language_issues) > 10:
        print(f"... and {len(language_issues) - 10} more")

# 6. Check for duplicate headwords
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

print("\nCheck complete.") 