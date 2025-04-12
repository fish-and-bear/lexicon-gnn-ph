#!/usr/bin/env python
"""
Script to fix syntax errors in routes.py
"""

def main():
    # Read the original file
    with open('backend/routes.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the position where the file gets corrupted
    # We know it's around line 3510 where there's a triple-quoted string without closure
    start_position = content.find('"""', 3500)
    
    # Find where we can continue from - at the end of the function around line 7000
    end_position = content.find('        # Load credits if requested', 7000)
    
    # Create fixed content by joining the uncorrupted start with a proper except block
    # and then appending the end of the file from where the function continues
    fixed_content = (
        content[:start_position] + 
        'derived.baybayin_form = d.baybayin_form\n                    word.derived_words.append(derived)\n' +
        '            except Exception as e:\n' +
        '                logger.error(f"Error loading derived words for word {word_id}: {e}", exc_info=False)\n' +
        '                word.derived_words = []\n' +
        '        else:\n' +
        '            word.derived_words = []\n\n' +
        content[end_position:]
    )
    
    # Write the fixed content to a new file
    with open('backend/routes.fixed.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed file created at backend/routes.fixed.py")
    print("Review the fixed file and if it looks good, replace the original with:")
    print("copy backend\\routes.fixed.py backend\\routes.py")

if __name__ == "__main__":
    main() 