import re
from models import Word
from backend.database import db_session

# List of language codes to ignore (case-sensitive)
IGNORE_CODES = ('Esp', 'Ing', 'Tag', 'Hil', 'Seb', 'War', 'Kap', 'Bik')

def extract_language_codes(etymology):
    # Create a regex pattern to match and extract the language codes
    pattern = r'\b(?:' + '|'.join(re.escape(code) for code in IGNORE_CODES) + r')\b'
    # Find all matches of the language codes
    matches = re.findall(pattern, etymology)
    # Remove the language codes from the etymology
    cleaned_etymology = re.sub(pattern, '', etymology).strip()
    return matches, cleaned_etymology

def update_language_codes():
    words = db_session.query(Word).all()  # Fetch all words
    total = len(words)
    
    for i, word_entry in enumerate(words, 1):
        try:
            # Extract language codes and clean etymology
            language_codes, cleaned_etymology = extract_language_codes(word_entry.etymology)
            
            # Update the word entry with the cleaned data
            word_entry.etymology = cleaned_etymology
            word_entry.language_codes = ', '.join(language_codes)
            
            if i % 100 == 0:  # Commit every 100 words to avoid huge transactions
                db_session.commit()
                print(f"Updated {i}/{total} words")

        except Exception as e:
            print(f"Error updating word '{word_entry.word}': {str(e)}")
            db_session.rollback()

    # Final commit
    db_session.commit()
    print("Update completed")

if __name__ == '__main__':
    update_language_codes()
