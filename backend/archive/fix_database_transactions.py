#!/usr/bin/env python
"""
Database Transaction Fixer for Filipino Dictionary

This script attempts to repair transaction errors and clean up the database.
"""

import os
import sys
import logging
import psycopg2
from rich.console import Console
from rich.progress import Progress

# Add the parent directory to the path for importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from dictionary_manager
import dictionary_manager as dm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('database_fix.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_connection():
    """Establish a connection to the database."""
    try:
        conn = dm.get_connection()
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        sys.exit(1)

def fix_active_transactions(conn):
    """Cancel any active transactions that might be blocking operations."""
    console = Console()
    console.print("[bold yellow]Checking for active transactions...[/]")
    
    try:
        # First, check if there are any active transactions
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pid, usename, application_name, state, query_start, query 
                FROM pg_stat_activity 
                WHERE state = 'active' AND pid <> pg_backend_pid()
            """)
            active_transactions = cur.fetchall()
            
            if not active_transactions:
                console.print("[green]No active transactions found.[/]")
                return
            
            console.print(f"[yellow]Found {len(active_transactions)} active transactions.[/]")
            
            # Ask for confirmation before proceeding
            console.print("[bold red]WARNING: Terminating active transactions may cause data loss.[/]")
            confirmation = input("Do you want to proceed with terminating transactions? (y/n): ")
            
            if confirmation.lower() != 'y':
                console.print("[yellow]Operation cancelled.[/]")
                return
            
            # Terminate active transactions
            for pid, username, app, state, start, query in active_transactions:
                try:
                    console.print(f"Terminating transaction by {username} (PID: {pid})")
                    cur.execute(f"SELECT pg_terminate_backend({pid})")
                except Exception as e:
                    logger.error(f"Failed to terminate transaction {pid}: {str(e)}")
            
            console.print("[green]Transaction cleanup completed.[/]")
    except Exception as e:
        logger.error(f"Error checking active transactions: {str(e)}")

def repair_table_sequences(conn):
    """Repair table sequences to ensure they point to correct next values."""
    console = Console()
    console.print("[bold yellow]Repairing table sequences...[/]")
    
    tables_with_sequences = [
        ("words", "id", "words_id_seq"),
        ("parts_of_speech", "id", "parts_of_speech_id_seq"),
        ("definitions", "id", "definitions_id_seq"),
        ("word_relations", "id", "word_relations_id_seq"),
        ("definition_relations", "id", "definition_relations_id_seq"),
        ("etymologies", "id", "etymologies_id_seq"),
        ("affixations", "id", "affixations_id_seq")
    ]
    
    try:
        with conn.cursor() as cur:
            for table, id_column, sequence in tables_with_sequences:
                try:
                    # Get max ID
                    cur.execute(f"SELECT MAX({id_column}) FROM {table}")
                    max_id = cur.fetchone()[0]
                    
                    if max_id is None:
                        console.print(f"[yellow]No entries found in {table}, skipping sequence repair.[/]")
                        continue
                    
                    # Set sequence to the next value
                    next_id = max_id + 1
                    cur.execute(f"SELECT setval('{sequence}', {next_id}, false)")
                    console.print(f"[green]Repaired sequence for {table}. Next ID will be {next_id}.[/]")
                except Exception as e:
                    logger.error(f"Failed to repair sequence for {table}: {str(e)}")
    except Exception as e:
        logger.error(f"Error repairing table sequences: {str(e)}")

def fix_constraint_violations(conn):
    """Find and fix constraint violations in the database."""
    console = Console()
    console.print("[bold yellow]Checking for constraint violations...[/]")
    
    try:
        with conn.cursor() as cur:
            # Check Baybayin form validity
            console.print("[yellow]Checking Baybayin form constraint violations...[/]")
            cur.execute("""
                UPDATE words 
                SET has_baybayin = false, baybayin_form = NULL, romanized_form = lemma
                WHERE has_baybayin = true AND 
                      (baybayin_form IS NULL OR 
                       baybayin_form = '' OR 
                       NOT baybayin_form ~ '^[\u1700-\u171F\u1735\u1736\s]+$')
            """)
            affected_rows = cur.rowcount
            console.print(f"[green]Fixed {affected_rows} invalid Baybayin entries.[/]")
            
            # Fix broken foreign key references
            console.print("[yellow]Checking for broken foreign key references...[/]")
            
            # Check definitions with invalid word_id
            cur.execute("""
                DELETE FROM definitions
                WHERE word_id NOT IN (SELECT id FROM words)
                RETURNING id
            """)
            deleted_defs = cur.fetchall()
            console.print(f"[green]Removed {len(deleted_defs)} definitions with invalid word references.[/]")
            
            # Check word relations with invalid word IDs
            cur.execute("""
                DELETE FROM word_relations
                WHERE from_word_id NOT IN (SELECT id FROM words)
                   OR to_word_id NOT IN (SELECT id FROM words)
                RETURNING id
            """)
            deleted_rels = cur.fetchall()
            console.print(f"[green]Removed {len(deleted_rels)} word relations with invalid word references.[/]")
            
            # Check definition relations with invalid references
            cur.execute("""
                DELETE FROM definition_relations
                WHERE definition_id NOT IN (SELECT id FROM definitions)
                   OR word_id NOT IN (SELECT id FROM words)
                RETURNING id
            """)
            deleted_def_rels = cur.fetchall()
            console.print(f"[green]Removed {len(deleted_def_rels)} definition relations with invalid references.[/]")
            
            # Check etymologies with invalid word IDs
            cur.execute("""
                DELETE FROM etymologies
                WHERE word_id NOT IN (SELECT id FROM words)
                RETURNING id
            """)
            deleted_etyms = cur.fetchall()
            console.print(f"[green]Removed {len(deleted_etyms)} etymologies with invalid word references.[/]")
            
            # Check affixations with invalid word IDs
            cur.execute("""
                DELETE FROM affixations
                WHERE root_id NOT IN (SELECT id FROM words)
                   OR affixed_id NOT IN (SELECT id FROM words)
                RETURNING id
            """)
            deleted_affixs = cur.fetchall()
            console.print(f"[green]Removed {len(deleted_affixs)} affixations with invalid word references.[/]")
            
    except Exception as e:
        logger.error(f"Error fixing constraint violations: {str(e)}")

def vacuum_database(conn):
    """Perform a VACUUM ANALYZE to reclaim space and update statistics."""
    console = Console()
    console.print("[bold yellow]Performing database vacuum...[/]")
    
    try:
        # We need to be in autocommit mode for VACUUM
        old_isolation = conn.isolation_level
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            console.print("[yellow]Running VACUUM ANALYZE...[/]")
            cur.execute("VACUUM ANALYZE")
            console.print("[green]Database vacuum completed successfully.[/]")
        
        # Restore previous isolation level
        conn.set_isolation_level(old_isolation)
    except Exception as e:
        logger.error(f"Error during database vacuum: {str(e)}")

def main():
    """Main function to repair the database."""
    console = Console()
    console.print("[bold blue]===== Filipino Dictionary Database Repair Tool =====[/]")
    
    # Connect to the database
    conn = setup_connection()
    
    try:
        # Fix active transactions
        fix_active_transactions(conn)
        
        # Fix constraint violations
        fix_constraint_violations(conn)
        
        # Repair sequences
        repair_table_sequences(conn)
        
        # Run VACUUM
        vacuum_database(conn)
        
        console.print("[bold green]Database repair completed successfully![/]")
    except Exception as e:
        logger.error(f"Unhandled error during database repair: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main() 