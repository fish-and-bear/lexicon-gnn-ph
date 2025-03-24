"""
Simple migration script to add any missing columns to the database.
"""

from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variables
db_url = os.getenv('DATABASE_URL')
if not db_url:
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'dictionary_dev')
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Connect to the database
engine = create_engine(db_url)

def column_exists(table, column):
    """Check if a column exists in a table."""
    query = text(f"""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name = '{table}' AND column_name = '{column}'
        );
    """)
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()[0]
    return result

def add_missing_columns():
    """Add missing columns from the models to the database tables."""
    missing_columns = []
    
    # Check and add root_word_id column
    if not column_exists('words', 'root_word_id'):
        missing_columns.append(("words", "root_word_id", "INT REFERENCES words(id)"))
    
    # Check and add preferred_spelling column
    if not column_exists('words', 'preferred_spelling'):
        missing_columns.append(("words", "preferred_spelling", "VARCHAR(255)"))
    
    # Check and add source_info column
    if not column_exists('words', 'source_info'):
        missing_columns.append(("words", "source_info", "JSONB DEFAULT '{}'"))
    
    # Check and add data_hash column
    if not column_exists('words', 'data_hash'):
        missing_columns.append(("words", "data_hash", "TEXT"))
    
    # Check and add idioms column
    if not column_exists('words', 'idioms'):
        missing_columns.append(("words", "idioms", "JSONB DEFAULT '[]'"))
    
    # Add any missing columns
    for table, column, definition in missing_columns:
        print(f"Adding {column} to {table}...")
        query = text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {definition};")
        with engine.connect() as conn:
            conn.execute(query)
            conn.commit()
        print(f"Added {column} to {table}")

if __name__ == "__main__":
    print("Checking for missing columns...")
    add_missing_columns()
    print("Migration complete.") 