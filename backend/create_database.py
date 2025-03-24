#!/usr/bin/env python3
"""
create_database.py

A helper script to create the required PostgreSQL database for the Filipino Dictionary.
This script will create the database if it doesn't exist.

Usage:
  python create_database.py
"""

import psycopg2
import os
import sys
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

# Database Configuration (same as in dictionary_manager.py)
DB_NAME = os.getenv("DB_NAME", "fil_dict")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

console = Console()

def create_database():
    """Create the database if it doesn't exist."""
    # Connect to default 'postgres' database first
    try:
        console.print(f"[bold]Connecting to PostgreSQL server...[/]")
        conn = psycopg2.connect(
            dbname="postgres",  # Connect to the default postgres database
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True  # Required for creating databases
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cur.fetchone()
        
        if exists:
            console.print(f"[yellow]Database '{DB_NAME}' already exists.[/]")
        else:
            # Create the database
            console.print(f"[bold]Creating database '{DB_NAME}'...[/]")
            cur.execute(f"CREATE DATABASE {DB_NAME}")
            console.print(f"[green]Database '{DB_NAME}' created successfully![/]")
        
        cur.close()
        conn.close()
        
        # Now test connecting to the created database
        test_conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        test_conn.close()
        console.print(f"[green]Successfully connected to database '{DB_NAME}'.[/]")
        console.print("\n[bold]Next steps:[/]")
        console.print("1. Run the migration script to import dictionary data:")
        console.print("   [bold]python dictionary_manager.py migrate --data-dir ../data[/]")
        
        return True
        
    except psycopg2.OperationalError as e:
        console.print(f"[bold red]Error connecting to PostgreSQL server:[/] {str(e)}")
        console.print("\n[bold]Troubleshooting:[/]")
        console.print("1. Make sure PostgreSQL is installed and running")
        console.print("2. Check that your PostgreSQL username and password are correct")
        console.print(f"3. Verify that PostgreSQL is running on {DB_HOST}:{DB_PORT}")
        console.print("\n[bold]Current settings:[/]")
        console.print(f"DB_NAME: {DB_NAME}")
        console.print(f"DB_USER: {DB_USER}")
        console.print(f"DB_HOST: {DB_HOST}")
        console.print(f"DB_PORT: {DB_PORT}")
        return False
    except Exception as e:
        console.print(f"[bold red]Error creating database:[/] {str(e)}")
        return False

if __name__ == "__main__":
    console.print("[bold]Filipino Dictionary Database Setup[/]")
    create_database() 