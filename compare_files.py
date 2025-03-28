import re
import difflib
from collections import defaultdict

def analyze_removed_functions():
    # Read files
    with open('backend/dictionary_manager.py', 'r', encoding='utf-8') as f:
        orig_lines = f.readlines()
    
    with open('backend/dictionary_manager.py.fixed', 'r', encoding='utf-8') as f:
        fixed_lines = f.readlines()
    
    # Get basic stats
    print(f"Original file: {len(orig_lines)} lines")
    print(f"Fixed file: {len(fixed_lines)} lines")
    print(f"Removed: {len(orig_lines) - len(fixed_lines)} lines\n")
    
    # Find function definitions in both files
    def_pattern = re.compile(r'^\s*def\s+([a-zA-Z0-9_]+)')
    
    orig_funcs = {}
    for i, line in enumerate(orig_lines):
        match = def_pattern.match(line)
        if match:
            func_name = match.group(1)
            orig_funcs.setdefault(func_name, []).append(i)
    
    fixed_funcs = {}
    for i, line in enumerate(fixed_lines):
        match = def_pattern.match(line)
        if match:
            func_name = match.group(1)
            fixed_funcs.setdefault(func_name, []).append(i)
    
    # Find functions with multiple definitions in original but only one in fixed
    print("Functions with duplicates removed:")
    for func_name, positions in orig_funcs.items():
        if len(positions) > 1 and func_name in fixed_funcs and len(fixed_funcs[func_name]) == 1:
            print(f"  {func_name}: had {len(positions)} occurrences, now has 1")
            
            # Show the starting line of each duplicate
            for i, pos in enumerate(positions):
                status = "Kept" if i == 0 else "Removed"
                print(f"    - {status} at line {pos+1}: {orig_lines[pos].strip()}")
    
    # Find any POS_MAPPING duplicates
    pos_mapping_pattern = re.compile(r'^\s*POS_MAPPING\s*=\s*{')
    
    pos_mapping_lines_orig = []
    for i, line in enumerate(orig_lines):
        if pos_mapping_pattern.match(line):
            pos_mapping_lines_orig.append(i)
    
    pos_mapping_lines_fixed = []
    for i, line in enumerate(fixed_lines):
        if pos_mapping_pattern.match(line):
            pos_mapping_lines_fixed.append(i)
    
    if len(pos_mapping_lines_orig) > len(pos_mapping_lines_fixed):
        print("\nPOS_MAPPING definitions:")
        print(f"  Original had {len(pos_mapping_lines_orig)}, fixed has {len(pos_mapping_lines_fixed)}")
        for i, line_num in enumerate(pos_mapping_lines_orig):
            status = "Kept" if i < len(pos_mapping_lines_fixed) else "Removed"
            print(f"    - {status} at line {line_num+1}")
    
    # Show a summary of all file differences
    print("\nSummary of removed content:")
    d = difflib.Differ()
    diff = list(d.compare(orig_lines, fixed_lines))
    
    removed_blocks = []
    current_block = []
    
    for line in diff:
        if line.startswith('- '):
            current_block.append(line[2:])
        elif current_block:
            if len(current_block) > 0:
                removed_blocks.append(current_block)
            current_block = []
    
    if current_block:
        removed_blocks.append(current_block)
    
    for i, block in enumerate(removed_blocks):
        if len(block) > 10:
            # Just show beginning and end for large blocks
            print(f"\nRemoved block #{i+1} ({len(block)} lines):")
            print("".join(block[:3]))
            print("...")
            print("".join(block[-3:]))
        else:
            print(f"\nRemoved block #{i+1}:")
            print("".join(block))

if __name__ == "__main__":
    analyze_removed_functions() 