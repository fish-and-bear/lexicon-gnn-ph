import json
import os
import sys
import traceback

def quick_check(json_file):
    try:
        print(f"Checking file: {json_file}")
        
        # Check if file exists
        if not os.path.exists(json_file):
            print(f"ERROR: File {json_file} not found!")
            return False
            
        # Load JSON data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON: {e}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to read file: {e}")
            return False
            
        print(f"Total entries: {len(data)}")
        
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
        if meaning_in_examples:
            print(f"\nFound {len(meaning_in_examples)} examples still using 'meaning' instead of 'example':")
            for idx, headword, ex_idx, example in meaning_in_examples[:5]:
                print(f"Entry {idx} ('{headword}'), example {ex_idx}: {example}")
            if len(meaning_in_examples) > 5:
                print(f"... and {len(meaning_in_examples) - 5} more")
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
        
        result = len(meaning_in_examples) == 0 and len(null_values) == 0 and len(missing_punctuation) == 0
        
        if result:
            print("\nAll critical issues have been fixed!")
        else:
            print("\nSome issues remain unfixed.")
        
        return result
    
    except Exception as e:
        print(f"ERROR during check: {e}")
        print(f"Type: {type(e).__name__}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        original_file = "data/gay-slang.json"
        fixed_file = "data/gay-slang-fixed.json"
        
        print("=== CHECKING ORIGINAL FILE ===")
        quick_check(original_file)
        
        print("\n=== CHECKING FIXED FILE ===")
        if os.path.exists(fixed_file):
            all_fixed = quick_check(fixed_file)
            print(f"\nVerification result: {'✓ All issues fixed!' if all_fixed else '✗ Some issues remain unfixed'}")
        else:
            print(f"Fixed file {fixed_file} not found. Run fix_json_inconsistencies.py first.")
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc() 