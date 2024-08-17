import json
from models import Word, Definition, Meaning, Source, AssociatedWord
from database import db_session, init_db
import re

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

def migrate_data():
    init_db()  # Initialize the database and create tables

    with open('../data/standardized_filipino_dictionary.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    for i, (word, entry) in enumerate(data.items(), 1):
        try:
            # Extract language codes and clean etymology
            language_codes, cleaned_etymology = extract_language_codes(entry.get('etymology', ''))

            # Create the Word entry
            word_entry = Word(
                word=word,
                pronunciation=entry.get('pronunciation', ''),
                etymology=cleaned_etymology,
                language_codes=', '.join(language_codes),  # Store language codes separately
                derivatives=json.dumps(entry.get('derivatives', {})),  # Convert dict to JSON string
                root_word=entry.get('root_word', '')
            )
            db_session.add(word_entry)
            
            # Add definitions
            for definition in entry.get('definitions', []):
                def_entry = Definition(
                    word=word_entry,
                    part_of_speech=definition.get('part_of_speech', '')
                )
                db_session.add(def_entry)
                
                # Add meanings
                for meaning in definition.get('meanings', []):
                    m_entry = Meaning(definition=def_entry, meaning=meaning)
                    db_session.add(m_entry)
                
                # Add sources
                for source in definition.get('sources', []):
                    s_entry = Source(definition=def_entry, source=source)
                    db_session.add(s_entry)
            
            # Add associated words
            for associated_word in entry.get('associated_words', []):
                aw_entry = AssociatedWord(word=word_entry, associated_word=associated_word)
                db_session.add(aw_entry)

            # Commit every 100 words to avoid huge transactions
            if i % 100 == 0:
                db_session.commit()
                print(f"Processed {i}/{total} words")

        except Exception as e:
            print(f"Error processing word '{word}': {str(e)}")
            db_session.rollback()

    # Final commit
    db_session.commit()
    print("Data migration completed")

if __name__ == '__main__':
    migrate_data()
