#!/usr/bin/env python3

"""
Script to run dictionary stats without modifying the database.
This script connects to the database and displays statistics without modifying any data.
"""

import logging
import argparse
import psycopg2
import psycopg2.extras
from rich.console import Console

# Import directly from the dictionary_manager.py file
import sys
import os

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import from the local file directly
from dictionary_manager import display_dictionary_stats, setup_logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_connection():
    """Get a database connection."""
    try:
        # Try to get database parameters from environment variables
        db_name = os.environ.get('DB_NAME', 'fil_dict_db')
        db_host = os.environ.get('DB_HOST', 'localhost')
        db_port = os.environ.get('DB_PORT', '5432')
        db_user = os.environ.get('DB_USER', 'postgres')
        db_pass = os.environ.get('DB_PASSWORD', '')
        
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_pass
        )
        logger.info(f"Successfully connected to database {db_name} at {db_host}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

def main():
    setup_logging()  # Set up logging from dictionary_manager
    
    parser = argparse.ArgumentParser(description="Run dictionary stats")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--db-name", type=str, help="Database name")
    parser.add_argument("--db-host", type=str, help="Database host")
    parser.add_argument("--db-port", type=str, help="Database port")
    parser.add_argument("--db-user", type=str, help="Database user")
    parser.add_argument("--db-password", type=str, help="Database password")
    
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set environment variables from command line args if provided
    if args.db_name:
        os.environ['DB_NAME'] = args.db_name
    if args.db_host:
        os.environ['DB_HOST'] = args.db_host
    if args.db_port:
        os.environ['DB_PORT'] = args.db_port
    if args.db_user:
        os.environ['DB_USER'] = args.db_user
    if args.db_password:
        os.environ['DB_PASSWORD'] = args.db_password
    
    console = Console()
    console.print("[bold blue]Running Dictionary Stats[/]")
    console.print("[yellow]This will only read from the database, not modify it.[/]")
    console.print(f"[dim]Using database: {os.environ.get('DB_NAME', 'fil_dict_db')} on {os.environ.get('DB_HOST', 'localhost')}[/]")
    
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # Call the display_dictionary_stats function with the cursor
            # Use the function directly from the file
            display_dictionary_stats(cur)
            
    except Exception as e:
        logger.error(f"Error running stats: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error: {str(e)}[/]")
    finally:
        if conn and not conn.closed:
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main() 