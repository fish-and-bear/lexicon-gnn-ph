import json

def quick_check(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    meaning_in_examples = []
    null_values = []
    missing_punctuation = []
    
    for i, entry in enumerate(data):
        # Check for null values
        for field in ["synonym", "sangkahulugan", "etymology", "usageLabels"]:
            if field in entry and entry[field] is None:
                null_values.append((i, entry["headword"], field))
        
        # Check examples for "meaning" field
        if "examples" in entry and entry["examples"]:
            for j, example in enumerate(entry["examples"]):
                if "meaning" in example:
                    meaning_in_examples.append((i, entry["headword"], j, example))
                
                # Check for missing punctuation
                if "example" in example and example["example"]:
                    text = example["example"].strip()
                    if text and not text.endswith((".", "!", "?", ":")):
                        missing_punctuation.append((i, entry["headword"], j, text))
    
    # Print results
    print(f"Checking file: {json_file}")
    print(f"Total entries: {len(data)}")
    
    if meaning_in_examples:
        print(f"\nFound {len(meaning_in_examples)} examples still using 'meaning' instead of 'example':")
        for idx, headword, ex_idx, example in meaning_in_examples[:5]:
            print(f"Entry {idx} ('{headword}'), example {ex_idx}: {example}")
    else:
        print("\nNo examples using 'meaning' instead of 'example' found.")
    
    if null_values:
        print(f"\nFound {len(null_values)} null values:")
        for idx, headword, field in null_values[:5]:
            print(f"Entry {idx} ('{headword}'): {field} is null")
        if len(null_values) > 5:
            print(f"... and {len(null_values) - 5} more")
    else:
        print("\nNo null values found.")
    
    if missing_punctuation:
        print(f"\nFound {len(missing_punctuation)} examples with missing punctuation:")
        for idx, headword, ex_idx, text in missing_punctuation[:5]:
            print(f"Entry {idx} ('{headword}'), example {ex_idx}: {text}")
        if len(missing_punctuation) > 5:
            print(f"... and {len(missing_punctuation) - 5} more")
    else:
        print("\nNo examples with missing punctuation found.")
    
    return len(meaning_in_examples) == 0 and len(null_values) == 0 and len(missing_punctuation) == 0

if __name__ == "__main__":
    original_file = "data/gay-slang.json"
    fixed_file = "data/gay-slang-fixed.json"
    
    print("=== CHECKING ORIGINAL FILE ===")
    quick_check(original_file)
    
    print("\n=== CHECKING FIXED FILE ===")
    all_fixed = quick_check(fixed_file)
    
    if all_fixed:
        print("\nAll critical issues have been fixed!")
    else:
        print("\nSome issues remain unfixed.") 