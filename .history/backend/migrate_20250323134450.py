"""
Simple migration script to add any missing columns to the database.
"""

from sqlalchemy import create_engine, text
import os
import sys
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
    db_name = os.getenv('DB_NAME', 'fil_dict_db')
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

def table_exists(table):
    """Check if a table exists in the database."""
    query = text(f"""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_name = '{table}'
        );
    """)
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()[0]
    return result

def recreate_database():
    """Drop all tables and recreate them using the setup_db.sql script."""
    try:
        # Connect to the database
        print("Connecting to PostgreSQL server...")
        
        # Execute the setup_db.sql script
        print("Recreating database schema...")
        
        # Read the SQL file
        with open('setup_db.sql', 'r') as f:
            sql_script = f.read()
        
        # Connect to PostgreSQL (not the target database, as we need to drop it)
        postgres_url = db_url.rsplit('/', 1)[0] + '/postgres'
        postgres_engine = create_engine(postgres_url)
        
        # Execute the SQL commands in the script
        with postgres_engine.connect() as conn:
            conn.execute(text("COMMIT"))  # End any existing transaction
            
            # Execute each statement separately
            for statement in sql_script.split(';'):
                if statement.strip():
                    try:
                        conn.execute(text(statement))
                        conn.commit()
                    except Exception as e:
                        print(f"Warning: {e}")
                        # Continue with other statements
        
        print("Database schema recreated successfully")
        return True
    except Exception as e:
        print(f"Error recreating database: {e}")
        return False

def add_missing_columns():
    """Add missing columns from the models to the database tables."""
    if not table_exists('words'):
        print("Words table doesn't exist. Database needs to be recreated.")
        return False
    
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
    
    return len(missing_columns) == 0  # Return True if no missing columns

if __name__ == "__main__":
    print("Checking database structure...")
    
    # Check if the --recreate flag is present
    recreate = "--recreate" in sys.argv
    
    if recreate:
        print("Recreating database (--recreate flag provided)...")
        success = recreate_database()
    else:
        # Try adding missing columns first
        success = add_missing_columns()
        
        # If missing columns couldn't be added, recreate the database
        if not success:
            print("Database schema is missing tables or has issues. Recreating database...")
            success = recreate_database()
    
    if success:
        print("Migration complete. Database is ready.")
    else:
        print("Migration failed. Please check the error messages above.")
        sys.exit(1) 