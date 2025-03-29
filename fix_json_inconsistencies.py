import json
import copy
import os
import sys

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
        "replaced_null_values": 0,
        "special_fixes": 0
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
        
        if "usageLabels" in entry and entry["usageLabels"] is None:
            entry["usageLabels"] = []
            changes["replaced_null_values"] += 1
        
        # 2. Fix examples section - replace "meaning" with "example"
        if "examples" in entry and entry["examples"]:
            for j, example in enumerate(entry["examples"]):
                # Use a direct check for the specific problematic examples we found
                if "meaning" in example:
                    # Store the original meaning value
                    meaning_value = example["meaning"]
                    
                    # Create a new example object with the correct structure
                    new_example = {
                        "language": example["language"],
                        "example": meaning_value
                    }
                    
                    # Replace the original example with the fixed one
                    entry["examples"][j] = new_example
                    changes["meaning_to_example"] += 1
                
                # 3. Add end punctuation if missing
                if "example" in example and example["example"]:
                    text = example["example"].strip()
                    if text and not text.endswith((".", "!", "?", ":")):
                        example["example"] = text + "."
                        changes["added_punctuation"] += 1
    
    # Now apply very specific direct fixes for the three examples we know are problematic
    print("Applying direct fixes to specific entries...")
    
    # Direct fix for "1-2-3"
    for entry in fixed_data:
        if entry["headword"] == "1-2-3":
            if "examples" in entry and len(entry["examples"]) > 1:
                entry["examples"][1]["example"] = "We can just do a 1-2-3 on the jeepney."
                print(f"Direct fix applied to '1-2-3' example")
                changes["special_fixes"] += 1
    
    # Direct fix for "alicia mayer"
    for entry in fixed_data:
        if entry["headword"] == "alicia mayer":
            if "examples" in entry and len(entry["examples"]) > 1:
                entry["examples"][1]["example"] = "The party's over. Let's alicia mayer."
                print(f"Direct fix applied to 'alicia mayer' example")
                changes["special_fixes"] += 1
    
    # Direct fix for "barnakol"
    for entry in fixed_data:
        if entry["headword"] == "barnakol":
            if "examples" in entry and len(entry["examples"]) > 1:
                entry["examples"][1]["example"] = "She has no conscience. That's why she has barnakol."
                print(f"Direct fix applied to 'barnakol' example")
                changes["special_fixes"] += 1
    
    # Save the fixed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    # Print a summary of changes
    print(f"Fixed JSON saved to {output_file}")
    print("\nSummary of changes:")
    print(f"- Replaced 'meaning' with 'example' in examples: {changes['meaning_to_example']}")
    print(f"- Added missing end punctuation: {changes['added_punctuation']}")
    print(f"- Replaced null values with empty lists or strings: {changes['replaced_null_values']}")
    print(f"- Applied special fixes: {changes['special_fixes']}")
    
    return changes

if __name__ == "__main__":
    # Allow command-line arguments for input and output files
    if len(sys.argv) > 2:
        json_file = sys.argv[1]
        fixed_file = sys.argv[2]
    else:
        json_file = "data/gay-slang.json"
        # Create a fixed version without overwriting the original
        fixed_file = "data/gay-slang-fixed.json"
    
    changes = fix_inconsistencies(json_file, fixed_file)
    
    print("\nTo verify the fixes worked correctly, run the validation script:")
    print("python validate.py") 