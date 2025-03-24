#!/usr/bin/env python3
"""
Fix foreign key constraint issues in the dictionary database.
This script will scan the database for broken references and fix them.
"""

import os
import sys
import psycopg2
import logging
import argparse
from tqdm import tqdm
import functools
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_connection():
    """Get a database connection using the same configuration as dictionary_manager.py."""
    try:
        # Import the get_connection function from dictionary_manager
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from dictionary_manager import get_connection as get_dm_connection
            return get_dm_connection()
        except ImportError:
            logger.warning("Could not import get_connection from dictionary_manager, using fallback config")
            
            # Fallback to default database configuration if import fails
            db_host = os.environ.get('POSTGRES_HOST', 'localhost')
            db_port = os.environ.get('POSTGRES_PORT', '5432')
            db_name = os.environ.get('POSTGRES_DB', 'fil_relex')  # Updated default database name
            db_user = os.environ.get('POSTGRES_USER', 'postgres')
            db_password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
            
            # Create connection
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                dbname=db_name,
                user=db_user,
                password=db_password
            )
            
            return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def with_transaction(func):
    """Decorator for transaction management."""
    @functools.wraps(func)
    def wrapper(conn, *args, **kwargs):
        with conn:
            with conn.cursor() as cur:
                return func(cur, *args, **kwargs)
    return wrapper

@with_transaction
def find_broken_references(cur):
    """Find broken foreign key references in the database."""
    logger.info("Checking for broken foreign key references...")
    
    # Check for definitions with non-existent word_ids
    cur.execute("""
        SELECT d.id, d.word_id, d.definition_text
        FROM definitions d
        LEFT JOIN words w ON d.word_id = w.id
        WHERE w.id IS NULL
    """)
    broken_definitions = cur.fetchall()
    
    if broken_definitions:
        logger.info(f"Found {len(broken_definitions)} definitions with broken word references")
        # Print a sample of broken definitions
        for i, (def_id, word_id, text) in enumerate(broken_definitions[:10]):
            logger.info(f"  Broken definition {i+1}: ID={def_id}, word_id={word_id}, text={text[:50]}...")
    else:
        logger.info("No broken word references found in definitions")
    
    # Check for relations with non-existent word IDs
    cur.execute("""
        SELECT r.id, r.from_word_id, r.to_word_id, r.relation_type
        FROM relations r
        LEFT JOIN words w1 ON r.from_word_id = w1.id
        LEFT JOIN words w2 ON r.to_word_id = w2.id
        WHERE w1.id IS NULL OR w2.id IS NULL
    """)
    broken_relations = cur.fetchall()
    
    if broken_relations:
        logger.info(f"Found {len(broken_relations)} relations with broken word references")
        # Print a sample of broken relations
        for i, (rel_id, from_id, to_id, rel_type) in enumerate(broken_relations[:10]):
            logger.info(f"  Broken relation {i+1}: ID={rel_id}, from={from_id}, to={to_id}, type={rel_type}")
    else:
        logger.info("No broken word references found in relations")
    
    # Check for etymologies with non-existent word IDs
    cur.execute("""
        SELECT e.id, e.word_id, e.etymology_text
        FROM etymologies e
        LEFT JOIN words w ON e.word_id = w.id
        WHERE w.id IS NULL
    """)
    broken_etymologies = cur.fetchall()
    
    if broken_etymologies:
        logger.info(f"Found {len(broken_etymologies)} etymologies with broken word references")
        # Print a sample of broken etymologies
        for i, (etym_id, word_id, text) in enumerate(broken_etymologies[:10]):
            logger.info(f"  Broken etymology {i+1}: ID={etym_id}, word_id={word_id}, text={text[:50]}...")
    else:
        logger.info("No broken word references found in etymologies")
    
    return broken_definitions, broken_relations, broken_etymologies

@with_transaction
def fix_broken_definitions(cur, broken_definitions):
    """Fix or delete definitions with broken word references."""
    if not broken_definitions:
        logger.info("No broken definitions to fix")
        return
    
    logger.info(f"Fixing {len(broken_definitions)} broken definitions...")
    
    deleted_count = 0
    with tqdm(total=len(broken_definitions), desc="Processing broken definitions") as pbar:
        for def_id, word_id, text in broken_definitions:
            try:
                # Option 1: Try to find the word by normalizing the text from the definition
                # Extract potential word from definition text - simplified approach
                words = text.split()
                if words:
                    potential_word = words[0].strip('.,;:!?()"\'').lower()
                    if potential_word:
                        # Check if this word exists in the database
                        cur.execute("""
                            SELECT id FROM words
                            WHERE normalized_lemma LIKE %s
                            LIMIT 1
                        """, (potential_word + '%',))
                        result = cur.fetchone()
                        
                        if result:
                            # Update the definition to use a valid word_id
                            new_word_id = result[0]
                            cur.execute("""
                                UPDATE definitions
                                SET word_id = %s
                                WHERE id = %s
                            """, (new_word_id, def_id))
                            logger.debug(f"Updated definition ID={def_id} to use word_id={new_word_id}")
                            pbar.update(1)
                            continue
                
                # Option 2: If no valid word found, delete the definition
                cur.execute("DELETE FROM definitions WHERE id = %s", (def_id,))
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error fixing definition ID={def_id}: {str(e)}")
            finally:
                pbar.update(1)
    
    logger.info(f"Fixed or deleted {len(broken_definitions)} broken definitions (deleted: {deleted_count})")

@with_transaction
def fix_broken_relations(cur, broken_relations):
    """Fix or delete relations with broken word references."""
    if not broken_relations:
        logger.info("No broken relations to fix")
        return
    
    logger.info(f"Deleting {len(broken_relations)} broken relations...")
    
    deleted_count = 0
    with tqdm(total=len(broken_relations), desc="Processing broken relations") as pbar:
        for rel_id, from_id, to_id, rel_type in broken_relations:
            try:
                # For relations, we'll just delete them
                cur.execute("DELETE FROM relations WHERE id = %s", (rel_id,))
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error fixing relation ID={rel_id}: {str(e)}")
            finally:
                pbar.update(1)
    
    logger.info(f"Deleted {deleted_count} broken relations")

@with_transaction
def fix_broken_etymologies(cur, broken_etymologies):
    """Fix or delete etymologies with broken word references."""
    if not broken_etymologies:
        logger.info("No broken etymologies to fix")
        return
    
    logger.info(f"Deleting {len(broken_etymologies)} broken etymologies...")
    
    deleted_count = 0
    with tqdm(total=len(broken_etymologies), desc="Processing broken etymologies") as pbar:
        for etym_id, word_id, text in broken_etymologies:
            try:
                # For etymologies, we'll just delete them
                cur.execute("DELETE FROM etymologies WHERE id = %s", (etym_id,))
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error fixing etymology ID={etym_id}: {str(e)}")
            finally:
                pbar.update(1)
    
    logger.info(f"Deleted {deleted_count} broken etymologies")

@with_transaction
def verify_fixes(cur):
    """Verify that all foreign key constraints are now satisfied."""
    logger.info("Verifying database integrity...")
    
    # Check for remaining definitions with non-existent word_ids
    cur.execute("""
        SELECT COUNT(*)
        FROM definitions d
        LEFT JOIN words w ON d.word_id = w.id
        WHERE w.id IS NULL
    """)
    broken_definitions_count = cur.fetchone()[0]
    
    # Check for remaining relations with non-existent word IDs
    cur.execute("""
        SELECT COUNT(*)
        FROM relations r
        LEFT JOIN words w1 ON r.from_word_id = w1.id
        LEFT JOIN words w2 ON r.to_word_id = w2.id
        WHERE w1.id IS NULL OR w2.id IS NULL
    """)
    broken_relations_count = cur.fetchone()[0]
    
    # Check for remaining etymologies with non-existent word IDs
    cur.execute("""
        SELECT COUNT(*)
        FROM etymologies e
        LEFT JOIN words w ON e.word_id = w.id
        WHERE w.id IS NULL
    """)
    broken_etymologies_count = cur.fetchone()[0]
    
    total_issues = broken_definitions_count + broken_relations_count + broken_etymologies_count
    if total_issues == 0:
        logger.info("Database integrity verified. All foreign key constraints are satisfied.")
        return True
    else:
        logger.warning(f"Database still has {total_issues} integrity issues:")
        logger.warning(f"  {broken_definitions_count} broken definitions")
        logger.warning(f"  {broken_relations_count} broken relations")
        logger.warning(f"  {broken_etymologies_count} broken etymologies")
        return False

def main():
    """Main function to run the database repair."""
    parser = argparse.ArgumentParser(description="Fix foreign key constraint issues in the dictionary database")
    parser.add_argument("--dry-run", action="store_true", help="Only check for issues without fixing them")
    args = parser.parse_args()
    
    try:
        # Connect to the database
        conn = get_connection()
        
        # Find broken references
        broken_definitions, broken_relations, broken_etymologies = find_broken_references(conn)
        
        if args.dry_run:
            logger.info("Dry run complete. No changes made to the database.")
            return
        
        # Fix broken references if any were found
        total_issues = len(broken_definitions) + len(broken_relations) + len(broken_etymologies)
        if total_issues > 0:
            logger.info(f"Found {total_issues} issues to fix. Starting repair process...")
            
            fix_broken_definitions(conn, broken_definitions)
            fix_broken_relations(conn, broken_relations)
            fix_broken_etymologies(conn, broken_etymologies)
            
            # Verify fixes
            verify_fixes(conn)
        else:
            logger.info("No issues found. Database appears to be in good shape.")
        
    except Exception as e:
        logger.error(f"Error repairing database: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()
    
    logger.info("Database repair completed successfully.")

if __name__ == "__main__":
    main() 