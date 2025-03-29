import json
import copy
import os

def fix_inconsistencies(json_file, output_file=None):
    # If no output file is specified, create a backup and overwrite the original
    if output_file is None:
        backup_file = json_file + ".backup"
        output_file = json_file
        print(f"Creating backup of original file to {backup_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            with open(backup_file, 'w', encoding='utf-8') as backup:
                backup.write(f.read())
    
    # Load the JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a deep copy to work with
    fixed_data = copy.deepcopy(data)
    
    # Track changes
    changes = {
        "meaning_to_example": 0,
        "added_punctuation": 0,
        "replaced_null_values": 0
    }
    
    # Process each entry
    for i, entry in enumerate(fixed_data):
        # 1. Fix null values
        for field in ["synonym", "sangkahulugan"]:
            if field in entry and entry[field] is None:
                entry[field] = []
                changes["replaced_null_values"] += 1
        
        if "etymology" in entry and entry["etymology"] is None:
            entry["etymology"] = ""
            changes["replaced_null_values"] += 1
        
        # 2. Fix examples section - replace "meaning" with "example"
        if "examples" in entry and entry["examples"]:
            for example in entry["examples"]:
                if "meaning" in example and "example" not in example:
                    example["example"] = example["meaning"]
                    del example["meaning"]
                    changes["meaning_to_example"] += 1
                
                # 3. Add end punctuation if missing
                if "example" in example and example["example"]:
                    text = example["example"].strip()
                    if text and not text.endswith((".", "!", "?", ":")):
                        example["example"] = text + "."
                        changes["added_punctuation"] += 1
    
    # Save the fixed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    # Print a summary of changes
    print(f"Fixed JSON saved to {output_file}")
    print("\nSummary of changes:")
    print(f"- Replaced 'meaning' with 'example' in examples: {changes['meaning_to_example']}")
    print(f"- Added missing end punctuation: {changes['added_punctuation']}")
    print(f"- Replaced null values with empty lists or strings: {changes['replaced_null_values']}")
    
    return changes

if __name__ == "__main__":
    json_file = "data/gay-slang.json"
    
    # Create a fixed version without overwriting the original
    fixed_file = "data/gay-slang-fixed.json"
    fix_inconsistencies(json_file, fixed_file)
    
    print("\nTo verify the fixes worked correctly, run the check scripts on the fixed file:")
    print(f"python check_comprehensive.py {fixed_file}") 