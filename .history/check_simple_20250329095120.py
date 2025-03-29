import json

# Load the JSON file
with open('data/gay-slang.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

# Check for examples with "meaning" instead of "example"
meaning_instead_of_example = []
for i, entry in enumerate(data):
    if "examples" in entry and entry["examples"]:
        for j, example in enumerate(entry["examples"]):
            if "meaning" in example and "language" in example and example["language"] == "english":
                meaning_instead_of_example.append((i, entry["headword"], example))

if meaning_instead_of_example:
    print(f"\nFound {len(meaning_instead_of_example)} examples using 'meaning' instead of 'example':")
    for idx, headword, example in meaning_instead_of_example[:5]:  # Show only first 5
        print(f"Entry {idx} ('{headword}'): {example}")
    if len(meaning_instead_of_example) > 5:
        print(f"... and {len(meaning_instead_of_example) - 5} more")

# Check for empty fields
empty_fields = []
for i, entry in enumerate(data):
    for field in ["headword", "partOfSpeech", "definitions", "examples"]:
        if field not in entry or entry[field] is None or (isinstance(entry[field], str) and entry[field].strip() == ""):
            empty_fields.append((i, entry.get("headword", "UNKNOWN"), field))

if empty_fields:
    print(f"\nFound {len(empty_fields)} empty required fields:")
    for idx, headword, field in empty_fields:
        print(f"Entry {idx} ('{headword}'): Missing or empty '{field}'")

# Check for syntax consistency in examples (periods, etc.)
example_punctuation = []
for i, entry in enumerate(data):
    if "examples" in entry and entry["examples"]:
        for j, example in enumerate(entry["examples"]):
            if "example" in example and isinstance(example["example"], str):
                text = example["example"]
                # Check if ends with period
                if not text.endswith(".") and not text.endswith("?") and not text.endswith("!"):
                    example_punctuation.append((i, entry["headword"], "Missing end punctuation", text))

if example_punctuation:
    print(f"\nFound {len(example_punctuation)} examples with punctuation issues:")
    for idx, headword, issue, text in example_punctuation[:10]:  # Show only first 10
        print(f"Entry {idx} ('{headword}'): {issue} in '{text}'")
    if len(example_punctuation) > 10:
        print(f"... and {len(example_punctuation) - 10} more")

print("\nCheck complete.") 