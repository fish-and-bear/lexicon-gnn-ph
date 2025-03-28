import re
import os

def remove_duplicate_functions(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into lines for analysis
    lines = content.splitlines()
    
    # Function to identify function definitions
    def_pattern = re.compile(r'^\s*def\s+([a-zA-Z0-9_]+)\s*\(')
    
    # Store function names and their first occurrence
    function_names = {}
    duplicate_ranges = []
    
    # Analyze the file line by line
    i = 0
    while i < len(lines):
        line = lines[i]
        match = def_pattern.match(line)
        
        if match:
            # Found a function definition
            function_name = match.group(1)
            print(f"Found function: {function_name} at line {i+1}")
            
            # Find the end of the function
            start_idx = i
            i += 1
            
            # Count indentation level of the function body
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(' ') or lines[i].startswith('\t')):
                i += 1
            
            end_idx = i - 1
            
            # Check if this is a duplicate
            if function_name in function_names:
                print(f"  Duplicate of previous definition at line {function_names[function_name]+1}")
                duplicate_ranges.append((start_idx, end_idx))
            else:
                function_names[function_name] = start_idx
        else:
            i += 1
    
    # Also check for POS_MAPPING duplicates (special case)
    pos_mapping_pattern = re.compile(r'^\s*POS_MAPPING\s*=\s*{')
    pos_mapping_indices = []
    
    for i, line in enumerate(lines):
        if pos_mapping_pattern.match(line):
            pos_mapping_indices.append(i)
            print(f"Found POS_MAPPING at line {i+1}")
    
    if len(pos_mapping_indices) > 1:
        # Find the end of each POS_MAPPING definition
        for start_idx in pos_mapping_indices[1:]:  # Skip the first one
            end_idx = start_idx
            while end_idx < len(lines) and not lines[end_idx].strip().endswith('}'):
                end_idx += 1
            if end_idx < len(lines):
                duplicate_ranges.append((start_idx, end_idx))
                print(f"  Duplicate POS_MAPPING from line {start_idx+1} to {end_idx+1}")
    
    # Sort the duplicate ranges in reverse order to remove from bottom to top
    duplicate_ranges.sort(reverse=True)
    
    # Remove the duplicates
    for start, end in duplicate_ranges:
        print(f"Removing lines {start+1} to {end+1}")
        del lines[start:end+1]
    
    # Write the modified content back
    output_path = file_path + '.fixed'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Removed {len(duplicate_ranges)} duplicate function definitions")
    print(f"Saved cleaned file to {output_path}")
    
    return output_path

if __name__ == "__main__":
    file_path = os.path.join('backend', 'dictionary_manager.py')
    cleaned_file = remove_duplicate_functions(file_path)
    
    # Print stats
    with open(file_path, 'r', encoding='utf-8') as f:
        original_lines = len(f.readlines())
    
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        new_lines = len(f.readlines())
    
    print(f"Original file: {original_lines} lines")
    print(f"Cleaned file: {new_lines} lines")
    print(f"Removed {original_lines - new_lines} lines") 