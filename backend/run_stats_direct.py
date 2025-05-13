#!/usr/bin/env python3

"""
Direct statistics script that accesses the functions from dictionary_manager.py
without going through the package import.
"""

import os
import sys
import logging
import argparse
import psycopg2
import psycopg2.extras
import json
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_connection_params():
    """
    Get database connection parameters from environment variables, 
    falling back to defaults.
    """
    # Check for .env file in the root or current directory
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
        os.path.join(os.getcwd(), '.env'),
    ]
    
    # Load from .env file if it exists
    env_vars = {}
    for env_path in env_paths:
        if os.path.exists(env_path):
            logger.info(f"Reading environment variables from {env_path}")
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"\'')
    
    # Get params with this priority: command line args > env vars > defaults
    return {
        'dbname': os.environ.get('DB_NAME', env_vars.get('DB_NAME', 'fil_dict_db')),
        'host': os.environ.get('DB_HOST', env_vars.get('DB_HOST', 'localhost')),
        'port': os.environ.get('DB_PORT', env_vars.get('DB_PORT', '5432')),
        'user': os.environ.get('DB_USER', env_vars.get('DB_USER', 'postgres')),
        'password': os.environ.get('DB_PASSWORD', env_vars.get('DB_PASSWORD', '')),
    }

def get_connection(db_params=None):
    """Get a database connection."""
    if db_params is None:
        db_params = get_connection_params()
    
    try:
        # Try with the password included
        try:
            conn = psycopg2.connect(**db_params)
            logger.info(f"Successfully connected to database {db_params['dbname']} at {db_params['host']}")
            return conn
        except psycopg2.OperationalError as e:
            if "password" in str(e).lower():
                # If password authentication failed, try to connect with trust auth
                logger.warning("Password authentication failed, trying trust authentication")
                # Create a copy of the parameters without password
                trust_params = db_params.copy()
                if 'password' in trust_params:
                    del trust_params['password']
                conn = psycopg2.connect(**trust_params)
                logger.info("Connected using trust authentication")
                return conn
            else:
                raise  # Re-raise if it's not a password issue
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run dictionary stats directly without import issues")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--db-name", type=str, help="Database name (overrides environment variable)")
    parser.add_argument("--db-host", type=str, help="Database host (overrides environment variable)")
    parser.add_argument("--db-port", type=str, help="Database port (overrides environment variable)")
    parser.add_argument("--db-user", type=str, help="Database user (overrides environment variable)")
    parser.add_argument("--db-password", type=str, help="Database password (overrides environment variable)")
    
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get connection parameters from env vars and command line args
    db_params = get_connection_params()
    
    # Update params with command line args if provided
    if args.db_name:
        db_params['dbname'] = args.db_name
    if args.db_host:
        db_params['host'] = args.db_host
    if args.db_port:
        db_params['port'] = args.db_port
    if args.db_user:
        db_params['user'] = args.db_user
    if args.db_password:
        db_params['password'] = args.db_password
    
    # Create copy of params for display (hide password)
    display_params = db_params.copy()
    if 'password' in display_params:
        display_params['password'] = '******' if display_params['password'] else '(none)'
    
    console = Console()
    console.print("[bold blue]Running Dictionary Stats Directly[/]")
    console.print("[yellow]This will only read from the database, not modify it.[/]")
    console.print(f"[dim]Connection parameters: {display_params}[/]")
    
    conn = None
    try:
        conn = get_connection(db_params)
        with conn.cursor() as cur:
            # Instead of importing the function, we'll copy the basic code pattern
            # from the display_dictionary_stats function to show table counts
            
            console.print("\n[bold]Dictionary Statistics[/]")
            
            # Create a nice table for table counts
            count_table = Table(title="Table Counts", box=ROUNDED)
            count_table.add_column("Table", style="cyan")
            count_table.add_column("Count", justify="right", style="green")
            
            # Get some basic table counts
            tables = [
                "words",
                "definitions", 
                "relations",
                "etymologies",
                "pronunciations",
                "credits",
                "word_forms",
                "definition_examples",
                "definition_links", 
                "definition_categories"
            ]
            
            for table in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    count_table.add_row(table, f"{count:,}")
                except Exception as e:
                    count_table.add_row(table, f"[red]Error: {str(e)}[/]")
            
            console.print(count_table)
            
            # Get language statistics
            console.print("\n[bold]Language Distribution:[/]")
            
            lang_table = Table(box=ROUNDED)
            lang_table.add_column("Language", style="cyan")
            lang_table.add_column("Count", justify="right", style="green")
            
            try:
                cur.execute("""
                    SELECT language_code, COUNT(*) as count
                    FROM words
                    GROUP BY language_code
                    ORDER BY count DESC
                """)
                
                results = cur.fetchall()
                if results:
                    for lang_code, count in results:
                        lang_table.add_row(lang_code or "Unknown", f"{count:,}")
                    console.print(lang_table)
                else:
                    console.print("[yellow]No language data available[/]")
            except Exception as e:
                console.print(f"[red]Error getting language statistics: {str(e)}[/]")
            
            # Get parts of speech statistics
            console.print("\n[bold]Parts of Speech Distribution:[/]")
            
            pos_table = Table(box=ROUNDED)
            pos_table.add_column("Part of Speech", style="cyan")
            pos_table.add_column("Count", justify="right", style="green")
            
            try:
                # Try to get POS stats from different POS columns depending on schema
                try:
                    cur.execute("""
                        SELECT p.name_tl, COUNT(*) as count
                        FROM definitions d
                        JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                        GROUP BY p.name_tl
                        ORDER BY count DESC
                    """)
                except:
                    # Fallback to original_pos if the standardized_pos_id join fails
                    cur.execute("""
                        SELECT original_pos, COUNT(*) as count
                        FROM definitions
                        GROUP BY original_pos
                        ORDER BY count DESC
                    """)
                
                results = cur.fetchall()
                if results:
                    for pos, count in results:
                        pos_table.add_row(pos or "Uncategorized", f"{count:,}")
                    console.print(pos_table)
                else:
                    console.print("[yellow]No part of speech data available[/]")
            except Exception as e:
                console.print(f"[red]Error getting part of speech statistics: {str(e)}[/]")
            
            console.print("\n[dim]This is a simplified statistics report that only reads data without modifying the database.[/]")
            
    except Exception as e:
        logger.error(f"Error running stats: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error: {str(e)}[/]")
    finally:
        if conn and not conn.closed:
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main() 