import re
import os

def remove_duplicate_functions(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into lines for analysis
    lines = content.splitlines()
    
    # Function to identify function definitions
    def_pattern = re.compile(r'^def\s+([a-zA-Z0-9_]+)\s*\(')
    
    # Store function names and their first occurrence
    function_names = {}
    duplicate_ranges = []
    
    # First pass: identify all function definitions
    in_function = False
    current_function = None
    function_start = None
    
    for i, line in enumerate(lines):
        # Check if line defines a function
        match = def_pattern.match(line)
        if match and not in_function:
            function_name = match.group(1)
            in_function = True
            current_function = function_name
            function_start = i
        
        # Check if we're exiting a function definition
        elif in_function and line.strip() == "" and not any(lines[j].strip().startswith((' ', '\t')) for j in range(i+1, min(i+3, len(lines)))):
            # We've reached the end of a function definition
            if current_function in function_names:
                # This is a duplicate
                print(f"Found duplicate function: {current_function} at line {i+1}")
                duplicate_ranges.append((function_start, i))
            else:
                # This is the first occurrence
                function_names[current_function] = function_start
            
            in_function = False
            current_function = None
    
    # Handle the case where a function extends to the end of the file
    if in_function and current_function in function_names:
        duplicate_ranges.append((function_start, len(lines)-1))
    
    # Also check for POS_MAPPING duplicates (special case)
    pos_mapping_pattern = re.compile(r'^POS_MAPPING\s*=\s*{')
    pos_mapping_indices = []
    
    for i, line in enumerate(lines):
        if pos_mapping_pattern.match(line.strip()):
            pos_mapping_indices.append(i)
    
    if len(pos_mapping_indices) > 1:
        # Find the end of each POS_MAPPING definition
        for start_idx in pos_mapping_indices[1:]:  # Skip the first one
            end_idx = start_idx
            while end_idx < len(lines) and not lines[end_idx].strip().endswith('}'):
                end_idx += 1
            if end_idx < len(lines):
                duplicate_ranges.append((start_idx, end_idx))
    
    # Sort the duplicate ranges in reverse order to remove from bottom to top
    duplicate_ranges.sort(reverse=True)
    
    # Remove the duplicates
    for start, end in duplicate_ranges:
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