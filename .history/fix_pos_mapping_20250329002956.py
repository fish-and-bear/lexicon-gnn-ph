#!/usr/bin/env python

"""
Script to fix the POS_MAPPING dictionary in the dictionary_manager.py file.
This script moves the POS_MAPPING dictionary from inside the migrate_relationships_to_new_system
function to the module level, so it can be accessed by the get_standard_code function.
"""

import re

def fix_pos_mapping():
    # File paths
    file_path = 'backend/dictionary_manager.py'
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    # Find the POS_MAPPING dictionary inside the migrate_relationships_to_new_system function
    pos_mapping_start = None
    pos_mapping_end = None
    inside_migrate_func = False
    
    for i, line in enumerate(content):
        if "def migrate_relationships_to_new_system" in line:
            inside_migrate_func = True
        elif inside_migrate_func and "POS_MAPPING =" in line:
            pos_mapping_start = i
        elif pos_mapping_start is not None and pos_mapping_end is None and "}" in line:
            pos_mapping_end = i
            break
    
    if pos_mapping_start is None or pos_mapping_end is None:
        print("Could not find POS_MAPPING dictionary in the file")
        return
    
    # Extract the POS_MAPPING dictionary
    pos_mapping_lines = content[pos_mapping_start:pos_mapping_end+1]
    
    # Find where to insert it at module level - just before get_standard_code
    insert_position = None
    for i, line in enumerate(content):
        if "def get_standard_code" in line:
            # Look for a comment line before the function
            comment_line = None
            for j in range(i-1, max(0, i-5), -1):
                if content[j].strip().startswith('#'):
                    comment_line = j
                    break
            
            if comment_line is not None:
                insert_position = comment_line
            else:
                # Insert before the function
                insert_position = i
            break
    
    if insert_position is None:
        print("Could not find suitable insert position for POS_MAPPING")
        return
    
    # Insert the POS_MAPPING at module level and remove it from the function
    new_content = []
    for i, line in enumerate(content):
        if i == insert_position:
            new_content.extend(pos_mapping_lines)
            new_content.append("\n")
        elif pos_mapping_start <= i <= pos_mapping_end:
            # Skip these lines as we're moving them
            continue
        else:
            new_content.append(line)
    
    # Replace function POS_MAPPING reference with a note
    for i, line in enumerate(new_content):
        if "migrate_relationships_to_new_system" in line:
            # Find where to insert the note
            for j in range(i+1, len(new_content)):
                if "# 5. get_uncategorized_pos_id" in new_content[j]:
                    note_position = j + 1
                    new_content.insert(note_position, "    # Note: POS_MAPPING dictionary is now defined at module level\n")
                    break
    
    # Write the modified content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)
    
    print(f"Fixed POS_MAPPING dictionary in {file_path}")

if __name__ == "__main__":
    fix_pos_mapping() 