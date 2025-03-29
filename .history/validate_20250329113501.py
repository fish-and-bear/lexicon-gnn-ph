import json

def validate_fixed_file(file_path):
    print(f"Validating file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    # Check for remaining issues
    meaning_in_examples = 0
    null_values = 0
    missing_punctuation = 0
    
    for entry in data:
        # Check for null values
        for field in ["synonym", "sangkahulugan", "etymology", "usageLabels"]:
            if field in entry and entry[field] is None:
                null_values += 1
                print(f"Null value in '{entry['headword']}': {field}")
        
        # Check examples
        if "examples" in entry and entry["examples"]:
            for example in entry["examples"]:
                if "meaning" in example:
                    meaning_in_examples += 1
                    print(f"'Meaning' found in '{entry['headword']}' example")
                
                if "example" in example and example["example"]:
                    text = example["example"].strip()
                    if text and not text.endswith((".", "!", "?", ":")):
                        missing_punctuation += 1
                        print(f"Missing punctuation in '{entry['headword']}' example: {text}")
    
    # Print summary
    print("\nValidation Summary:")
    print(f"- Examples using 'meaning' instead of 'example': {meaning_in_examples}")
    print(f"- Null values found: {null_values}")
    print(f"- Examples with missing punctuation: {missing_punctuation}")
    
    if meaning_in_examples == 0 and null_values == 0 and missing_punctuation == 0:
        print("\n✓ All issues have been fixed!")
    else:
        print("\n✗ Some issues remain unfixed.")

if __name__ == "__main__":
    fixed_file = "data/gay-slang-fixed.json"
    validate_fixed_file(fixed_file) 