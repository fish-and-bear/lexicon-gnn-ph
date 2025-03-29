import json
import sys

def validate_fixed_file(file_path):
    print(f"Validating file: {file_path}")
    
    # Check if file exists
    import os
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} does not exist!")
        return -1, -1, -1
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Successfully loaded JSON data")
    except Exception as e:
        print(f"ERROR loading JSON: {str(e)}")
        return -1, -1, -1
    
    print(f"Total entries: {len(data)}")
    
    # Check for remaining issues
    meaning_in_examples = []
    null_values = []
    missing_punctuation = []
    
    print("Checking for issues...")
    
    for i, entry in enumerate(data):
        # Check for null values
        for field in ["synonym", "sangkahulugan", "etymology", "usageLabels"]:
            if field in entry and entry[field] is None:
                null_values.append((i, entry["headword"], field))
        
        # Check examples
        if "examples" in entry and entry["examples"]:
            for j, example in enumerate(entry["examples"]):
                if "meaning" in example:
                    meaning_in_examples.append((i, entry["headword"], j, example))
                
                if "example" in example and example["example"]:
                    text = example["example"].strip()
                    if text and not text.endswith((".", "!", "?", ":")):
                        missing_punctuation.append((i, entry["headword"], j, text))
    
    print("Done checking. Generating report...")
    
    # Print detailed results
    if meaning_in_examples:
        print("\nFound examples still using 'meaning' instead of 'example':")
        for i, headword, j, example in meaning_in_examples:
            print(f"- Entry {i} ('{headword}'), example {j}: {example}")
    else:
        print("\nNo examples using 'meaning' instead of 'example' found.")
    
    if null_values:
        print("\nFound null values:")
        for i, headword, field in null_values:
            print(f"- Entry {i} ('{headword}'): {field} is null")
    else:
        print("\nNo null values found.")
    
    if missing_punctuation:
        print("\nFound examples with missing punctuation:")
        for i, headword, j, text in missing_punctuation:
            print(f"- Entry {i} ('{headword}'), example {j}: '{text}'")
    else:
        print("\nNo examples with missing punctuation found.")
    
    # Print summary
    print("\nValidation Summary:")
    print(f"- Examples using 'meaning' instead of 'example': {len(meaning_in_examples)}")
    print(f"- Null values found: {len(null_values)}")
    print(f"- Examples with missing punctuation: {len(missing_punctuation)}")
    
    if not meaning_in_examples and not null_values and not missing_punctuation:
        print("\n✓ All issues have been fixed!")
    else:
        print("\n✗ Some issues remain unfixed.")
    
    return len(meaning_in_examples), len(null_values), len(missing_punctuation)

if __name__ == "__main__":
    fixed_file = "data/gay-slang-fixed.json"
    print(f"Starting validation of {fixed_file}")
    
    try:
        meaning, null, punct = validate_fixed_file(fixed_file)
        print(f"Validation complete. Issues found: meaning={meaning}, null={null}, punct={punct}")
    except Exception as e:
        print(f"ERROR during validation: {e}")
        import traceback
        traceback.print_exc() 