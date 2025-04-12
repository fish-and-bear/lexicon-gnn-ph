#!/usr/bin/env python
"""
Script to fix syntax errors in routes.py by extracting just the clean parts
"""

def main():
    # Read content of the file
    with open('backend/routes.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract the parts we need: the first part up to where corruption starts,
    # and the second part after where the duplication ends
    first_part = lines[:3500]  # First part up to where corruption starts
    last_part = lines[6000:]   # Last part after corrupted section
    
    # Insert the missing parts to fix the derived_words section
    patch = [
        "                    derived.baybayin_form = d.baybayin_form\n",
        "                    word.derived_words.append(derived)\n",
        "            except Exception as e:\n",
        "                logger.error(f\"Error loading derived words for word {word_id}: {e}\", exc_info=False)\n",
        "                word.derived_words = []\n",
        "        else:\n",
        "            word.derived_words = []\n",
        "\n"
    ]
    
    # Find the start of the load credits section
    credit_start = None
    for i, line in enumerate(last_part):
        if "# Load credits if requested" in line:
            credit_start = i
            break
    
    if credit_start is None:
        print("Could not find the 'Load credits if requested' section in the file.")
        return
    
    # Create the fixed content
    fixed_content = first_part + patch + last_part[credit_start:]
    
    # Write to a new file
    with open('backend/routes.fixed2.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_content)
    
    print("Fixed file created at backend/routes.fixed2.py")
    print("Review the fixed file and if it looks good, replace the original with:")
    print("copy backend\\routes.fixed2.py backend\\routes.py")

if __name__ == "__main__":
    main() 