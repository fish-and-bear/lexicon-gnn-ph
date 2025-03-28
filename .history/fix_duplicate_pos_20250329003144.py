with open('backend/dictionary_manager.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the indented POS_MAPPING
indented_pos_mapping_start = None
for i, line in enumerate(lines):
    if line.strip().startswith('POS_MAPPING =') and line.startswith('    '):
        indented_pos_mapping_start = i
        break

# Remove the indented POS_MAPPING if found
if indented_pos_mapping_start:
    end_line = indented_pos_mapping_start
    while end_line < len(lines) and not lines[end_line].strip().endswith('}'):
        end_line += 1
    
    if end_line < len(lines) and lines[end_line].strip().endswith('}'):
        # Remove the duplicate POS_MAPPING
        del lines[indented_pos_mapping_start:end_line+1]
        
        # Write the changes back to the file
        with open('backend/dictionary_manager.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print('Removed duplicate POS_MAPPING dictionary')
    else:
        print('Could not find end of indented POS_MAPPING')
else:
    print('No indented POS_MAPPING found') 