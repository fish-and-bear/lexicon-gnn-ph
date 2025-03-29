import json
import sys

def check_json_consistency(file_path):
    """Check for inconsistencies in the JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} entries from {file_path}")
        
        # Check for required fields
        required_fields = ["headword", "partOfSpeech", "definitions", "examples"]
        field_counts = {}
        missing_required = []
        
        for i, entry in enumerate(data):
            for field in entry.keys():
                field_counts[field] = field_counts.get(field, 0) + 1
            
            # Check for missing required fields
            for req_field in required_fields:
                if req_field not in entry:
                    missing_required.append((i, entry["headword"], req_field))
        
        # Print field statistics
        print("\nField occurrence counts:")
        for field, count in sorted(field_counts.items()):
            print(f"{field}: {count}/{len(data)} ({count/len(data)*100:.1f}%)")
        
        if missing_required:
            print("\nEntries missing required fields:")
            for idx, headword, field in missing_required:
                print(f"Entry {idx} ('{headword}'): Missing '{field}'")
        
        # Check for inconsistencies in definitions
        definition_issues = []
        for i, entry in enumerate(data):
            if "definitions" in entry:
                for j, definition in enumerate(entry["definitions"]):
                    if "language" not in definition:
                        definition_issues.append((i, entry["headword"], j, "Missing 'language' field"))
                    elif definition["language"] not in ["filipino", "english"]:
                        definition_issues.append((i, entry["headword"], j, f"Invalid language: '{definition['language']}'"))
                    
                    if "meaning" not in definition:
                        definition_issues.append((i, entry["headword"], j, "Missing 'meaning' field"))
                    elif definition["meaning"] is None or definition["meaning"].strip() == "":
                        definition_issues.append((i, entry["headword"], j, "Empty meaning"))
        
        if definition_issues:
            print("\nDefinition inconsistencies:")
            for idx, headword, def_idx, issue in definition_issues:
                print(f"Entry {idx} ('{headword}'), definition {def_idx}: {issue}")
        
        # Check for inconsistencies in examples
        example_issues = []
        for i, entry in enumerate(data):
            if "examples" in entry and entry["examples"]:
                for j, example in enumerate(entry["examples"]):
                    if "language" not in example:
                        example_issues.append((i, entry["headword"], j, "Missing 'language' field"))
                    elif example["language"] not in ["filipino", "english"]:
                        example_issues.append((i, entry["headword"], j, f"Invalid language: '{example['language']}'"))
                    
                    example_field = "example" if "example" in example else "meaning" if "meaning" in example else None
                    if example_field is None:
                        example_issues.append((i, entry["headword"], j, "Missing both 'example' and 'meaning' fields"))
                    elif example_field == "meaning":
                        example_issues.append((i, entry["headword"], j, "Has 'meaning' field instead of 'example'"))
                    elif example[example_field] is None or example[example_field].strip() == "":
                        example_issues.append((i, entry["headword"], j, f"Empty {example_field}"))
        
        if example_issues:
            print("\nExample inconsistencies:")
            for idx, headword, ex_idx, issue in example_issues:
                print(f"Entry {idx} ('{headword}'), example {ex_idx}: {issue}")
        
        # Check for duplicate headwords
        headwords = {}
        for i, entry in enumerate(data):
            headword = entry["headword"]
            if headword in headwords:
                headwords[headword].append(i)
            else:
                headwords[headword] = [i]
        
        duplicates = {hw: indices for hw, indices in headwords.items() if len(indices) > 1}
        if duplicates:
            print("\nDuplicate headwords:")
            for headword, indices in duplicates.items():
                print(f"'{headword}' appears at indices: {indices}")
        
        if not (missing_required or definition_issues or example_issues or duplicates):
            print("\nNo inconsistencies found!")
    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    file_path = "data/gay-slang.json"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    check_json_consistency(file_path) 