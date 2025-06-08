#!/usr/bin/env python3
import argparse
import json
import logging
import os
import io
import traceback # Keep traceback for error logging in lookup_word
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import re # Add import for regex

# Database connection and core helpers
from backend.dictionary_manager.db_helpers import (
    get_connection,
    with_transaction,
    create_or_update_tables,
    repair_database_issues,
    check_baybayin_consistency,
    setup_parts_of_speech, # ADDED IMPORT
)
# Text helpers needed directly
from backend.dictionary_manager.text_helpers import (
    normalize_lemma, # Used by lookup_word
)

# Enums (None needed directly)
# from backend.dictionary_manager.enums import ...

# Core DB library (for exception handling)
import psycopg2
import sys # Added for setup_logging
import codecs # Added for setup_logging

# Processors (All are needed by migrate_data)
from backend.dictionary_manager.processors.tagalog_processor import process_tagalog_words
from backend.dictionary_manager.processors.root_words_processor import process_root_words_cleaned
from backend.dictionary_manager.processors.kwf_processor import process_kwf_dictionary
from backend.dictionary_manager.processors.gay_slang_processor import process_gay_slang_json
from backend.dictionary_manager.processors.kaikki_processor import process_kaikki_jsonl
from backend.dictionary_manager.processors.marayum_processor import process_marayum_json, process_marayum_directory
from backend.dictionary_manager.processors.tagalog_processor import process_tagalog_words
from backend.dictionary_manager.processors.calderon_processor import process_calderon_json

# Utils
# from backend.utils.logging_config import setup_logging # Removed redundant import

# Rich library for CLI output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED as box_ROUNDED # Use alias to avoid potential name clash
from rich.text import Text
from rich.syntax import Syntax

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Initialize Rich Console
console = Console()

# -------------------------------------------------------------------
# Setup Logging (Moved from original backup)
# -------------------------------------------------------------------
def setup_logging():
    """Configure logging with proper Unicode handling."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Get the ROOT logger instead of the specific module logger
    _logger = logging.getLogger() 
    _logger.setLevel(logging.DEBUG) # Keep level at DEBUG
    # _logger.handlers = [] # Clearing handlers on root logger can be risky if other libs configure logging, but often ok if this is the main config point. Let's keep it cleared for now to ensure only our handlers are active.
    _logger.handlers.clear() # More explicit way to clear handlers

    file_path = f"{log_dir}/dictionary_manager_{timestamp}.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    _logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # --- Unicode Console Handling (from original backup) ---
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            # The following line might cause issues if stdout is redirected
            # Consider making console encoding more robust if needed
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                console_handler.stream = codecs.getwriter("utf-8")(sys.stdout.buffer)
            except (AttributeError, io.UnsupportedOperation):
                # Fallback if reconfigure fails or stdout doesn't support it
                _logger.warning("Could not reconfigure sys.stdout for UTF-8. Console output might have encoding issues.")
                # Use default stream handler without forcing encoding
                pass # console_handler.stream remains default
        else:
            # For non-Windows, assume UTF-8 is generally okay
            # If issues arise, specific checks for terminal encoding might be needed
            pass
    except Exception as e:
        # Catch potential errors during console setup (e.g., ctypes issues)
        _logger.error(f"Error setting up console for UTF-8: {e}. Console output might have encoding issues.")
        # Fallback: Try safe encoding wrapper (from original backup)
        def safe_encode(msg):
            try:
                # Attempt encoding with stream's encoding, replace errors
                stream_encoding = getattr(console_handler.stream, 'encoding', 'utf-8') or 'utf-8'
                return (
                    str(msg)
                    .encode(stream_encoding, "replace")
                    .decode(stream_encoding)
                )
            except Exception:
                # Absolute fallback to ASCII with replacement
                return str(msg).encode("ascii", "replace").decode("ascii")

        original_emit = console_handler.emit

        def safe_emit(record):
            try:
                record.msg = safe_encode(record.msg)
                original_emit(record)
            except Exception as emit_err:
                 # Log error during emit itself if safe_encode fails
                 _logger.error(f"Error during logging emit: {emit_err}", exc_info=False)

        console_handler.emit = safe_emit
    # --- End Unicode Console Handling ---

    _logger.addHandler(console_handler)
    return _logger

# Setup logging
logger = setup_logging() # Call the function to configure
# logger = logging.getLogger(__name__) # Keep getting the logger for this module - REMOVE THIS LINE

def check_pg_trgm_installed(cur):
    """Check if the pg_trgm extension is installed."""
    try:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
        return cur.fetchone() is not None
    except Exception as e:
        logger.warning(f"Could not check for pg_trgm extension: {str(e)}")
        return False

# --- Global Variables & Constants ---
STATS_COUNTERS = Counter()
MAX_WORKERS = 8 # Adjust based on system resources
DB_NAME = os.getenv("DB_NAME", "fil_dict_db") # Keep DB config vars for error messages
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")

# -------------------------------------------------------------------
# Command Line Interface Functions
# -------------------------------------------------------------------
def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage dictionary data in PostgreSQL."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Migrate Command (Keeping existing args) ---
    migrate_parser = subparsers.add_parser(
        "migrate", help="Create/update schema and load data"
    )
    migrate_parser.add_argument(
        "--check-exists", action="store_true", help="Skip identical existing entries"
    )
    migrate_parser.add_argument(
        "--force", action="store_true", help="Drop existing dictionary data and re-migrate all sources."
    )
    migrate_parser.add_argument(
        "--data-dir", type=str, help="Directory containing dictionary data files"
    )
    migrate_parser.add_argument(
        "--sources", type=str, help="Comma-separated list of source names to process"
    )
    migrate_parser.add_argument(
        "--file", type=str, help="Specific data file to process"
    )

    # --- Verify Command ---
    verify_parser = subparsers.add_parser("verify", help="Verify data integrity")
    verify_parser.add_argument(
        "--quick", action="store_true", help="Run quick verification checks"
    )
    verify_parser.add_argument(
        "--repair", action="store_true", help="Attempt to repair found issues"
    )
    verify_parser.add_argument(
        "--checks", type=str, default="all", help="Comma-separated list of checks to run (e.g., orphans,duplicates,search_vectors,baybayin)"
    )
    verify_parser.add_argument(
        "--repair-tasks", type=str, default="all", help="Comma-separated list of issues to attempt repairing (if --repair is used)"
    )

    # --- Update Command ---
    update_parser = subparsers.add_parser("update", help="Update DB with new data from a file")
    update_parser.add_argument(
        "--file", type=str, required=True, help="JSON or JSONL file containing new/updated entries"
    )
    update_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    # Add other relevant options for update if needed (e.g., --source-name)

    # --- Lookup Command ---
    lookup_parser = subparsers.add_parser("lookup", help="Look up word information")
    lookup_parser.add_argument(
        "term", help="Word lemma or ID to look up"
    )
    lookup_parser.add_argument(
        "--id", action="store_true", help="Indicates that 'term' is a word ID"
    )
    lookup_parser.add_argument(
        "--lang", type=str, help="Specify language code (e.g., tl, ceb) to narrow search"
    )
    lookup_parser.add_argument(
        "--debug", action="store_true", help="Show debug information"
    )
    lookup_parser.add_argument(
        "--format",
        choices=["text", "json", "rich"],
        default="rich",
        help="Output format",
    )

    # --- Stats Command ---
    stats_parser = subparsers.add_parser("stats", help="Display dictionary statistics")
    stats_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed statistics (may be slower)"
    )
    stats_parser.add_argument(
        "--export", type=str, help="Export statistics to file (e.g., stats.json, stats.csv)"
    )
    stats_parser.add_argument(
        "--table", type=str, help="Show detailed stats only for a specific table"
    )

    # --- Leaderboard Command ---
    leaderboard_parser = subparsers.add_parser("leaderboard", help="Display top contributors/sources")
    leaderboard_parser.add_argument(
        "--limit", type=int, default=10, help="Limit the number of entries shown per category"
    )
    leaderboard_parser.add_argument(
        "--sort-by", type=str, default="count", help="Specify sorting criteria (e.g., count, words, examples)"
    )

    # --- Help Command ---
    subparsers.add_parser("help", help="Display detailed help information")

    # --- Explore Command ---
    subparsers.add_parser("explore", help="Interactive dictionary explorer (CLI)")

    # --- Purge Command ---
    purge_parser = subparsers.add_parser("purge", help="Safely delete data from dictionary tables")
    purge_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt (USE WITH CAUTION!)"
    )
    purge_parser.add_argument(
        "--tables", type=str, help="Comma-separated list of specific tables to purge (default: all dictionary tables)"
    )

    return parser


def migrate_data(args):
    """Migrate dictionary data from various sources."""
    sources = [
        {
            "name": "Root Words",
            "file": "root_words_with_associated_words_cleaned.json",
            "handler": process_root_words_cleaned,
            "required": False,
        },
        {
            "name": "KWF Dictionary",
            "file": "kwf_dictionary.json",
            "handler": process_kwf_dictionary,
            "required": False,
        },
        {
            "name": "Kaikki.org (Tagalog)",
            "file": "kaikki.jsonl",
            "handler": process_kaikki_jsonl,
            "required": False,
        },
        {
            "name": "Kaikki.org (Cebuano)",
            "file": "kaikki-ceb.jsonl",
            "handler": process_kaikki_jsonl,
            "required": False,
        },
        {
            "name": "Project Marayum",
            "file": "marayum_dictionaries",  # Changed to directory name only
            "handler": process_marayum_directory,
            "required": False,
            "is_directory": True,
        },
        {
            "name": "Philippine Slang and Gay Dictionary",
            "file": "gay-slang.json",
            "handler": process_gay_slang_json,
            "required": False,  # Set to True if it should always be processed if present
            "is_directory": False,  # It's a file
        },
        {
            "name": "Tagalog Words",
            "file": "tagalog-words.json",
            "handler": process_tagalog_words,
            "required": False,
        },
        {
            "name": "Tagalog Words",
            "file": "..\\data\\tagalog-words.json",
            "handler": process_tagalog_words, # <-- Make sure it's this name
            "required": False,
        },
        {
            "name": "Calderon Diccionario 1915",
            "file": "..\\data\\calderon_dictionary.json",
            "handler": process_calderon_json,
            "required": False,
        },
    ]

    # Get data directory from args if provided, or use defaults
    if hasattr(args, "data_dir") and args.data_dir:
        data_dirs = [args.data_dir]
    else:
        # Added more possible data directory locations
        data_dirs = [
            "data", 
            os.path.join("..", "data"),
            os.path.join(os.path.dirname(__file__), "data"),
            os.path.join(os.path.dirname(__file__), "..", "data"),
            os.path.join(os.path.dirname(__file__), "..", "..", "data")
        ]

    # Check if any data directories exist
    existing_dirs = [d for d in data_dirs if os.path.isdir(d)]
    if not existing_dirs:
        console.print("[bold red]No data directories found![/]")
        console.print("Searched in:", ", ".join(data_dirs))
        console.print("\n[bold]To fix this issue:[/]")
        console.print("1. Create a 'data' directory in the project root")
        console.print("2. Add dictionary data files to this directory")
        console.print("3. Or specify a data directory with --data-dir")
        return

    # Filter sources if specific ones are requested via --sources
    if hasattr(args, "sources") and args.sources:
        requested_sources = [s.lower() for s in args.sources.split(",")]
        # More flexible matching for source names
        sources = [
            s
            for s in sources
            if any(
                req in s["name"].lower()
                or req in s["file"].lower()
                or (req == "marayum" and "marayum" in s["file"].lower())
                for req in requested_sources
            )
        ]
        if not sources:
            logger.error(f"No matching sources found for: {args.sources}")
            return
        for source in sources:
            source["required"] = True

    # Custom file overrides existing sources
    if hasattr(args, "file") and args.file:
        filename = args.file
        if filename.endswith(".jsonl"):
            handler = process_kaikki_jsonl
        elif "root_words" in filename.lower():
            handler = process_root_words_cleaned
        elif "kwf" in filename.lower():
            handler = process_kwf_dictionary
        elif "marayum" in filename.lower():
            handler = process_marayum_directory
        elif "tagalog" in filename.lower():
            handler = process_tagalog_words
        elif "gay" in filename.lower():
            handler = process_gay_slang_json

        basename = os.path.basename(filename)
        source_found = False
        for source in sources:
            if source["file"] == basename or (
                os.path.isdir(filename)
                and os.path.basename(source["file"]) == os.path.basename(filename)
            ):
                source["file"] = filename  # Use full path
                source["required"] = True
                source_found = True
                break
        if not source_found:
            sources.append(
                {
                    "name": f"Custom ({basename})",
                    "file": filename,
                    "handler": handler,
                    "required": True,
                    "is_directory": os.path.isdir(filename),
                }
            )

    conn = None
    cur = None
    console = Console()
    try:
        conn = get_connection()
        conn.autocommit = False  # Set autocommit mode *before* creating cursor
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        console.print(f"\n[bold red]Failed to connect to the database:[/] {str(e)}")
        console.print(
            "Please check your database configuration and ensure the database exists."
        )
        console.print(
            f"Current settings: DB_NAME={DB_NAME}, DB_HOST={DB_HOST}, DB_PORT={DB_PORT}, DB_USER={DB_USER}"
        )
        console.print(
            "\n[bold]To fix this issue:[/]\n1. Make sure PostgreSQL is running\n2. Create the database if it doesn't exist (e.g., `createdb {DB_NAME}`)\n3. Verify your .env settings."
        )
        return

    try:
        cur = conn.cursor()

        # --- ADDED: Purge data if --force is specified ---
        if hasattr(args, 'force') and args.force:
            console.print("[bold yellow]--force specified: Purging existing dictionary data before migration...[/]")
            try:
                # Calling the purge_database_tables defined locally in dictionary_manager.py
                # This function is decorated with @with_transaction(commit=True)
                # and expects a cursor.
                purge_database_tables(cur) 
                console.print("[green]Existing dictionary data purged successfully.[/]")
            except Exception as e:
                logger.error(f"Error during forced purge: {str(e)}", exc_info=True)
                console.print(f"[bold red]Failed to purge existing data: {str(e)}[/]")
                # Re-raise to stop the migration if purge fails when forced.
                raise
        # --- END ADDED CODE ---

        console.print("[bold]Setting up database schema...[/]")
        create_or_update_tables(conn)  # This function handles its own commit/rollback

        logger.info("Setting up initial parts of speech...") # ADDED
        setup_parts_of_speech(cur) # ADDED: Call the setup function

        console.print("[bold]Processing data sources...[/]")
        # --- Transaction for Data Processing is already implicitly started ---
        # conn.autocommit = False  # Removed from here

        # Check if any data files exist before starting the migration
        available_files = []
        for source in sources:
            for data_dir in existing_dirs:
                potential_path = os.path.join(data_dir, source["file"])
                if source.get("is_directory", False):
                    if os.path.isdir(potential_path):
                        available_files.append(f"{source['name']} ({potential_path})")
                        break
                else:
                    if os.path.isfile(potential_path):
                        available_files.append(f"{source['name']} ({potential_path})")
                        break
        
        if not available_files:
            console.print("[bold red]No data files found in the data directories![/]")
            console.print("Expected files:")
            for source in sources:
                console.print(f"- {source['file']} ({source['name']})")
            console.print("\nCheck that your data files are in one of these directories:", ", ".join(existing_dirs))
            return

        console.print("[green]Found data files:[/]")
        for file in available_files:
            console.print(f"- {file}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            total_files = len(sources)
            main_task = progress.add_task("Migrating data", total=total_files)

            for idx, source in enumerate(sources):
                source_name = source["name"]
                progress.update(
                    main_task,
                    description=f"Processing source {idx+1}/{total_files}: {source_name}",
                    advance=0,
                )

                # Look for file in provided data directories if not an absolute path
                if os.path.isabs(source["file"]):
                    filepath = source["file"]
                else:
                    filepath = None
                    for data_dir in existing_dirs:
                        potential_path = os.path.join(data_dir, source["file"])
                        if source.get("is_directory", False):
                            if os.path.isdir(potential_path):
                                filepath = potential_path
                                break
                        else:
                            if os.path.isfile(potential_path):
                                filepath = potential_path
                                break

                if not filepath:
                    msg = f"Required {'directory' if source.get('is_directory', False) else 'file'} not found: {source['file']}"
                    if source["required"]:
                        logger.error(msg)
                        # Raise an error to trigger rollback
                        raise FileNotFoundError(msg)
                    else:
                        logger.warning(f"{msg}. Skipping source: {source_name}")
                        progress.update(
                            main_task, advance=1
                        )  # Advance main task when skipping
                        continue

                # Use the found absolute path
                source_task_desc = (
                    f"Processing {source_name} ({os.path.basename(filepath)})..."
                )
                # Add a sub-task for individual file progress if handler supports it
                # For now, just log start/end
                logger.info(f"Starting processing for {source_name} from {filepath}")

                try:
                    # Call the handler function for the source
                    # Handlers might use @with_transaction, but they operate within the larger transaction managed here
                    source["handler"](cur, filepath)
                    logger.info(f"Successfully processed source: {source_name}")
                    # --- MODIFICATION: Commit after each successful handler --- 
                    logger.info(f"Committing transaction after successful processing of {source_name}...")
                    conn.commit()
                    logger.info(f"Commit successful for {source_name}.")
                    
                    # Force explicit commit for root words processor which sometimes has transaction issues
                    if source["name"] == "Root Words":
                        try:
                            # Ensure this specific processor's changes are definitely committed
                            logger.info("Forcing extra commit for Root Words processor")
                            cur.execute("SELECT 1") # Verify connection is working
                            conn.commit()
                            logger.info("Extra commit for Root Words processor successful")
                        except Exception as commit_err:
                            logger.error(f"Error during extra commit for Root Words processor: {commit_err}")
                except Exception as handler_error:
                    # Log error and rollback the entire migration if a handler fails
                    logger.error(
                        f"Error processing source '{source_name}' from {filepath}: {handler_error}",
                        exc_info=True,
                    )
                    # Raise the error to trigger the outer rollback
                    raise handler_error

                finally:
                    # Ensure the main progress bar always advances
                    progress.update(main_task, advance=1)

        # --- If all sources processed without error, commit the transaction ---
        # --- MODIFICATION: Commit is now handled after each processor ---
        # logger.info("All sources processed successfully. Committing transaction...")
        # conn.commit()  # Commit all changes made by handlers
        console.print("[bold green]Migration completed successfully.[/]")
        migration_successful = True # Flag success

    except Exception as e:
        # Rollback transaction if any error occurred during the migration process
        logger.error(f"Error during migration: {str(e)}", exc_info=True)
        console.print(f"\n[bold red]Migration failed:[/] {str(e)}")
        migration_successful = False # Flag failure
        if conn:
            try:
                logger.info("Rolling back transaction due to error.")
                conn.rollback()
            except Exception as rb_err:
                logger.error(f"Failed to rollback transaction: {rb_err}")
        # Re-raise the exception to indicate failure
        # raise e # Optionally re-raise

    finally:
        # Ensure cursor and connection are closed, AFTER commit/rollback
        if cur:
            try:
                cur.close()
            except Exception as cur_close_err:
                logger.warning(f"Error closing cursor: {cur_close_err}")
        if conn:
            try:
                # Only try to reset autocommit if connection is not closed and transaction ended
                if not conn.closed:
                    # If an error occurred and rollback might have failed, 
                    # setting autocommit might still raise the error we saw.
                    # It might be safer to just close without resetting.
                    # conn.autocommit = True # Potentially problematic line
                    pass # Just proceed to close
                conn.close()
                logger.info("Database connection closed.")
            except Exception as conn_close_err:
                # Avoid logging the 'set_session' error specifically if it happens here
                if "set_session cannot be used inside a transaction" not in str(conn_close_err):
                     logger.warning(f"Error closing connection: {conn_close_err}")


def verify_database(args):
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        console = Console()
        issues = []
        table_stats = Table(title="Table Statistics", box=box_ROUNDED)
        table_stats.add_column("Table", style="cyan")
        table_stats.add_column("Count", justify="right", style="green")
        tables = [
            "words",
            "definitions",
            "relations",
            "etymologies",
            "affixations",
            "definition_relations",
            "parts_of_speech",
        ]
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            table_stats.add_row(t, f"{count:,}")
        console.print(table_stats)
        console.print()

        # Display relation types and counts
        console.print("[bold]Relation Types and Counts[/]")
        cur.execute(
            """
            SELECT relation_type, COUNT(*) as count
            FROM relations
            GROUP BY relation_type
            ORDER BY count DESC
        """
        )
        relation_stats = cur.fetchall()
        if relation_stats:
            rel_table = Table(box=box_ROUNDED)
            rel_table.add_column("Relation Type", style="yellow")
            rel_table.add_column("Count", justify="right", style="green")

            for rel_type, count in relation_stats:
                rel_table.add_row(rel_type or "Unspecified", f"{count:,}")

            console.print(rel_table)
        else:
            console.print("[italic]No relations found[/]")

        console.print()

        # Display language distributions
        console.print("[bold]Language Distribution[/]")
        cur.execute(
            """
            SELECT language_code, COUNT(*) as count
            FROM words
            GROUP BY language_code
            ORDER BY count DESC
        """
        )
        lang_stats = cur.fetchall()
        if lang_stats:
            lang_table = Table(box=box_ROUNDED)
            lang_table.add_column("Language", style="magenta")
            lang_table.add_column("Count", justify="right", style="green")

            for lang, count in lang_stats:
                lang_name = (
                    "Tagalog" if lang == "tl" else "Cebuano" if lang == "ceb" else lang
                )
                lang_table.add_row(lang_name, f"{count:,}")

            console.print(lang_table)

        # Display parts of speech distribution
        console.print()
        console.print("[bold]Parts of Speech Distribution[/]")
        cur.execute(
            """
            SELECT p.name_tl, COUNT(*) as count
            FROM definitions d
            JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            GROUP BY p.name_tl
            ORDER BY COUNT(*) DESC
        """
        )
        pos_stats = cur.fetchall()
        if pos_stats:
            pos_table = Table(box=box_ROUNDED)
            pos_table.add_column("Part of Speech", style="blue")
            pos_table.add_column("Count", justify="right", style="green")

            for pos, count in pos_stats:
                pos_table.add_row(pos or "Uncategorized", f"{count:,}")

            console.print(pos_table)

        # Display definition sources
        console.print()
        console.print("[bold]Definition Sources[/]")
        cur.execute(
            """
            SELECT
                CASE 
                    WHEN sources = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                    WHEN sources = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                    WHEN sources = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                    WHEN sources = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                    WHEN sources = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                    ELSE sources
                END as source_name,
                COUNT(*) as count
            FROM definitions
            WHERE sources IS NOT NULL
            GROUP BY sources
            ORDER BY count DESC
        """
        )
        source_stats = cur.fetchall()
        if source_stats:
            source_table = Table(box=box_ROUNDED)
            source_table.add_column("Source", style="cyan")
            source_table.add_column("Count", justify="right", style="green")

            for source, count in source_stats:
                source_table.add_row(source or "Unknown", f"{count:,}")

            console.print(source_table)

        # Explain empty tables
        console.print()
        console.print("[bold]Notes on Empty Tables[/]")
        console.print(
            Panel(
                """
- [bold]affixations[/bold]: This table is for storing information about word affixation patterns, which are linguistic processes where affixes (prefixes, suffixes, infixes) are added to root words to create new words. These are populated by specialized affix analysis functions.

- [bold]definition_relations[/bold]: This table stores semantic relationships between definitions (rather than between words). These are typically populated during advanced linguistic analysis.

Both tables might be empty if no specialized linguistic analysis has been performed on the dataset yet.
""",
                title="Table Explanations",
                border_style="blue",
            )
        )

        if args.quick:
            console.print("[yellow]Sample entries from 'words' table:[/]")
            cur.execute(
                "SELECT id, lemma, language_code, root_word_id FROM words LIMIT 5"
            )
            sample_table = Table(show_header=True)
            sample_table.add_column("ID")
            sample_table.add_column("Lemma")
            sample_table.add_column("Language")
            sample_table.add_column("Root ID")
            for row in cur.fetchall():
                sample_table.add_row(*[str(x) for x in row])
            console.print(sample_table)
            return

        def check_data_integrity(cur) -> List[str]:
            integrity_issues = []
            cur.execute(
                """
                SELECT COUNT(*) FROM relations r
                WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.from_word_id)
                   OR NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.to_word_id)
            """
            )
            if cur.fetchone()[0] > 0:
                integrity_issues.append("Found orphaned relations")
            cur.execute(
                """
                SELECT COUNT(*) 
                FROM (
                    SELECT word_id, definition_text, COUNT(*)
                    FROM definitions
                    GROUP BY word_id, definition_text
                    HAVING COUNT(*) > 1
                ) dupes
            """
            )
            if cur.fetchone()[0] > 0:
                integrity_issues.append("Found duplicate definitions")
            cur.execute("SELECT COUNT(*) FROM words WHERE search_text IS NULL")
            if cur.fetchone()[0] > 0:
                integrity_issues.append("Found words with missing search vectors")
            cur.execute(
                """
                SELECT baybayin_form, COUNT(*)
                FROM words
                WHERE has_baybayin = TRUE
                GROUP BY baybayin_form
                HAVING COUNT(*) > 1
            """
            )
            dupes = cur.fetchall()
            if dupes and len(dupes) > 0:
                integrity_issues.append("Found duplicate Baybayin forms")
            return integrity_issues

        integrity_issues = check_data_integrity(cur)
        if integrity_issues:
            issues.extend(integrity_issues)
        baybayin_issues = check_baybayin_consistency(cur)
        if baybayin_issues:
            issues.extend(baybayin_issues)
        if issues:
            console.print("\n[bold red]Found Issues:[/]")
            issues_table = Table(show_header=True)
            issues_table.add_column("Issue")
            issues_table.add_column("Details")
            for issue in issues:
                issues_table.add_row(issue, "")
            console.print(issues_table)
            if args.repair:
                console.print("\n[yellow]Attempting to repair issues...[/]")
                repair_database_issues(cur, issues)
        else:
            console.print("\n[bold green]No issues found![/]")
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        raise
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        logger.info("Database verification completed")


def display_help(args):
    """Displays comprehensive help information using Rich components."""
    console = Console()
    console.print("[bold cyan]📖 Dictionary Manager CLI Help[/]", justify="center")
    console.print(
        "[dim]A comprehensive tool for managing Filipino dictionary data in PostgreSQL[/]",
        justify="center",
    )
    usage_panel = Panel(
        Text.from_markup("python dictionary_manager.py [command] [options...]"),
        title="Basic Usage",
        border_style="blue",
        expand=False
    )
    console.print(usage_panel)
    console.print()

    console.print("[bold underline]Available Commands:[/]")

    # --- Command Details --- 
    # Using a list of dictionaries for better structure
    commands_info = [
        {
            "name": "migrate",
            "icon": "🔄",
            "description": "Create/update database schema and load data from various sources.",
            "arguments": [
                ("--check-exists", "Skip inserting entries if an identical one already exists."),
                ("--force", "Purge existing dictionary data before migrating. Use with caution!"),
                ("--data-dir <dir>", "Specify the directory containing data files (default: searches './data', '../data', etc.)."),
                ("--sources <s1,s2..>", "Comma-separated list of source names to process (e.g., 'Kaikki', 'KWF'). Matches names/filenames."),
                ("--file <path>", "Process only a specific data file (JSON or JSONL). Overrides --sources.")
            ],
            "example": "python dictionary_manager.py migrate --force --sources Kaikki,KWF"
        },
        {
            "name": "verify",
            "icon": "✅",
            "description": "Check database integrity, statistics, and consistency.",
            "arguments": [
                ("--quick", "Show basic table counts and sample entries."),
                ("--repair", "[bold red]Not Implemented[/] Attempt to repair found issues automatically."), # Marked as not implemented
                ("--checks <c1,c2..>", "Comma-separated list of specific checks (default: all). Checks: orphans, duplicates, search_vectors, baybayin."),
                ("--repair-tasks <t1..>", "[bold red]Not Implemented[/] Comma-separated list of repair tasks (used with --repair).") # Marked as not implemented
            ],
            "example": "python dictionary_manager.py verify --checks orphans,baybayin"
        },
        {
            "name": "lookup",
            "icon": "🔍",
            "description": "Look up detailed information for a specific word by lemma or ID.",
            "arguments": [
                ("term", "The word lemma or ID to look up (required)."),
                ("--id", "Indicates that 'term' is a numerical word ID, not a lemma."),
                ("--lang <lc>", "Specify a language code (e.g., 'tl', 'ceb') to narrow the search."),
                ("--format <fmt>", "Output format: 'rich' (default, colored tables), 'text' (plain), 'json'."),
                ("--debug", "Show additional debug information during lookup.")
            ],
            "example": "python dictionary_manager.py lookup \"puso\" --lang tl --format rich"
        },
        {
            "name": "stats",
            "icon": "📊",
            "description": "Display detailed statistics about the dictionary content.",
            "arguments": [
                ("--detailed", "Show more detailed statistics (e.g., breakdown by POS, relation type)."),
                ("--export <file>", "Export statistics to a file (e.g., stats.json, stats.csv - [bold red]Not Implemented[/])."), # Marked as not implemented
                ("--table <table>", "Show detailed stats only for a specific table ([bold red]Not Implemented[/]).") # Marked as not implemented
            ],
            "example": "python dictionary_manager.py stats --detailed"
        },
        {
            "name": "leaderboard",
            "icon": "🏆",
            "description": "Show contribution leaderboards based on data sources.",
            "arguments": [
                ("--limit <num>", "Number of top entries to display per category (default: 10)."),
                ("--sort-by <crit>", "Sorting criteria (default: 'count'). See specific tables for valid options (e.g., 'definitions', 'words', 'examples' for Definition Contributors).")
            ],
            "example": "python dictionary_manager.py leaderboard --limit 5 --sort-by examples"
        },
         {
            "name": "explore",
            "icon": "🧭",
            "description": "Launch an interactive CLI explorer to browse the dictionary.",
            "arguments": [], # No arguments for explore
            "example": "python dictionary_manager.py explore"
        },
        {
            "name": "purge",
            "icon": "🗑️",
            "description": "Purge (delete) data from dictionary tables. [bold red]Use with extreme caution![/]",
            "arguments": [
                ("--force", "Skip the confirmation prompt. DANGEROUS!"),
                ("--tables <t1,t2..>", "Comma-separated list of specific tables to purge (default: purges all main dictionary tables).")
            ],
            "example": "python dictionary_manager.py purge --tables words,definitions"
        },
        {
            "name": "update",
            "icon": "📝",
            "description": "[bold yellow](Placeholder)[/] Update dictionary with new data from a file.",
            "arguments": [
                 ("--file <path>", "JSON or JSONL file with entries to update/add (required)."),
                 ("--dry-run", "Preview changes without applying them.")
            ],
            "example": "python dictionary_manager.py update --file new_words.json --dry-run"
        },
        {
            "name": "help",
            "icon": "❓",
            "description": "Display this detailed help message.",
            "arguments": [],
            "example": "python dictionary_manager.py help"
        }
    ]

    for cmd_info in commands_info:
        # Create a panel for each command
        cmd_panel_content = Text()
        cmd_panel_content.append(Text.from_markup(f"{cmd_info['description']}\n"))
        
        if cmd_info['arguments']:
            cmd_panel_content.append(Text.from_markup("[bold]Arguments:[/]\n"))
            for arg, desc in cmd_info['arguments']:
                # Simple heuristic to separate argument name from description for formatting
                arg_parts = arg.split(" ", 1)
                arg_name = arg_parts[0]
                arg_val = f" {arg_parts[1]}" if len(arg_parts) > 1 else ""
                cmd_panel_content.append(Text.from_markup(f"  [bold cyan]{arg_name}[/][dim]{arg_val}[/]: {desc}\n"))
        else:
            cmd_panel_content.append(Text.from_markup("[dim]No arguments for this command.[/]\n"))

        cmd_panel_content.append(Text.from_markup("\n[bold]Example:[/]\n"))
        cmd_panel_content.append(Text.from_markup(f"  [green]{cmd_info['example']}[/]"))

        cmd_panel = Panel(
            cmd_panel_content,
            title=f"{cmd_info['icon']} [bold yellow]{cmd_info['name']}[/]",
            border_style="blue" if cmd_info['name'] != 'purge' else "red", # Highlight purge
            expand=False,
            padding=(1, 2)
        )
        console.print(cmd_panel)
        console.print() # Add space between panels

    # Keep the footer note
    console.print(
        "\n[dim]For the most up-to-date and detailed information, please refer to the project documentation.[/]",
        justify="center",
    )
    console.print()


def lookup_word(args):
    """Look up a word and display its information, handling format and filtering."""
    term = args.term
    is_id_lookup = args.id
    lang_filter = args.lang
    output_format = args.format
    debug_mode = args.debug

    logger.info(f"Starting lookup for {'ID' if is_id_lookup else 'term'}: '{term}', Lang: {lang_filter or 'any'}, Format: {output_format}, Debug: {debug_mode}")

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # Use DictCursor for easier access
        console = Console()

        if is_id_lookup:
            try:
                word_id = int(term)
                lookup_by_id(cur, word_id, console, output_format, debug_mode)
            except ValueError:
                logger.error(f"Invalid ID provided: {term}")
                console.print(f"[red]Invalid ID: '{term}'. Please provide a number when using --id.[/]")
            except Exception as e:
                logger.error(f"Error looking up ID {term}: {e}", exc_info=True)
                console.print(f"[red]Error looking up ID {term}: {e}[/]")
            return # Exit after ID lookup attempt

        # --- Term Lookup Logic --- 
        with conn:
            with cur:
                normalized_word = normalize_lemma(term)
                if debug_mode:
                    logger.info(f"Normalized term: '{normalized_word}'")
                    logger.info(f"pg_trgm installed: {check_pg_trgm_installed(cur)}")

                base_query = """
                    SELECT id, lemma, language_code, root_word_id
                    FROM words 
                    WHERE {match_clause}
                """
                lang_clause = " AND language_code = %s " if lang_filter else ""
                params = []

                # 1. Try exact match
                logger.info(f"Executing exact match query for '{term}' (Lang: {lang_filter or 'any'})")
                exact_match_clause = " normalized_lemma = %s " + lang_clause
                exact_params = [normalized_word]
                if lang_filter: exact_params.append(lang_filter)
                
                cur.execute(base_query.format(match_clause=exact_match_clause), tuple(exact_params))
                results = cur.fetchall()
                logger.info(f"Found {len(results)} exact matches")

                # 2. If no exact matches, try fuzzy search (if pg_trgm available)
                if not results and check_pg_trgm_installed(cur):
                    logger.info(f"No exact matches, trying fuzzy search for '{term}' (Lang: {lang_filter or 'any'})")
                    fuzzy_match_clause = " similarity(normalized_lemma, %s) > 0.4 " + lang_clause
                    fuzzy_query = """
                        SELECT id, lemma, language_code, root_word_id, 
                               similarity(normalized_lemma, %s) as sim
                        FROM words 
                        WHERE {match_clause}
                        ORDER BY sim DESC
                        LIMIT 10
                    """
                    fuzzy_params = [normalized_word, normalized_word] # Similarity needs it twice
                    if lang_filter: fuzzy_params.append(lang_filter)
                    
                    try:
                        cur.execute(fuzzy_query.format(match_clause=fuzzy_match_clause), tuple(fuzzy_params))
                        results = cur.fetchall()
                        logger.info(f"Found {len(results)} fuzzy matches")
                    except Exception as e:
                        logger.warning(f"Error in fuzzy search: {e}")
                        results = [] # Reset results if fuzzy search fails
                elif not results:
                    logger.info("Skipping fuzzy search as pg_trgm is not installed or not needed.")

                # 3. If still no matches, try ILIKE search
                if not results:
                    logger.info(f"No fuzzy matches, falling back to ILIKE search for '{term}' (Lang: {lang_filter or 'any'})")
                    ilike_match_clause = " (lemma ILIKE %s OR normalized_lemma ILIKE %s) " + lang_clause
                    ilike_params = [f"%{term}%", f"%{normalized_word}%"]
                    if lang_filter: ilike_params.append(lang_filter)
                    
                    cur.execute(base_query.format(match_clause=ilike_match_clause), tuple(ilike_params))
                    results = cur.fetchall()
                    logger.info(f"ILIKE search found {len(results)} matches")

                # --- Process Results ---
                if not results:
                    # Run diagnostics if requested
                    if debug_mode:
                        logger.info("Diagnostic: checking if word exists in any form")
                        cur.execute(
                            "SELECT EXISTS(SELECT 1 FROM words WHERE lemma = %s OR normalized_lemma = %s)",
                            (term, normalized_word),
                        )
                        logger.info(f"Diagnostic result: Word '{term}' exists (exact match): {cur.fetchone()[0]}")
                        cur.execute(
                            "SELECT EXISTS(SELECT 1 FROM words WHERE lemma LIKE %s OR normalized_lemma LIKE %s)",
                            (f"%{term}%", f"%{normalized_word}%"),
                        )
                        logger.info(f"Diagnostic result: Word '{term}' exists (partial match): {cur.fetchone()[0]}")
                        logger.info("Diagnostic: verifying words table schema")
                        cur.execute(
                            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'words' ORDER BY ordinal_position"
                        )
                        columns = cur.fetchall()
                        logger.info(f"Diagnostic: words table columns: {[(c['column_name'], c['data_type']) for c in columns]}")
                    
                    console.print(f"\nNo entries found for '{term}'{f' in language {lang_filter}' if lang_filter else ''}.")
                    return

                if len(results) == 1:
                    word_id = results[0]['id']
                    lookup_by_id(cur, word_id, console, output_format, debug_mode)
                else:
                    # Multiple results - Display selection table (Rich format only for selection)
                    console.print("\n[bold]Multiple matches found:[/]")
                    table = Table(show_header=True, header_style="bold", box=box_ROUNDED)
                    table.add_column("ID", style="dim")
                    table.add_column("Word")
                    table.add_column("Language")
                    table.add_column("Root")

                    result_ids = []
                    for row in results:
                        result_ids.append(row['id'])
                        # Get root word if available
                        root_word = "N/A"
                        if row['root_word_id']:
                            cur.execute("SELECT lemma FROM words WHERE id = %s", (row['root_word_id'],))
                            root_row = cur.fetchone()
                            if root_row: root_word = root_row['lemma']

                        table.add_row(
                            str(row['id']),
                            row['lemma'],
                            row['language_code'] or "unknown",
                            root_word,
                        )

                    console.print(table)
                    choice = input("\nEnter ID to view details (or press Enter to exit): ")
                    if choice.strip():
                        try:
                            chosen_id = int(choice.strip())
                            if chosen_id in result_ids:
                                lookup_by_id(cur, chosen_id, console, output_format, debug_mode)
                            else:
                                console.print("[red]Invalid ID entered.[/]")
                        except ValueError:
                            console.print("[red]Invalid ID. Please enter a number.[/]")
                        except Exception as e:
                            logger.error(f"Error looking up chosen ID {choice}: {e}", exc_info=True)
                            console.print(f"[red]Error looking up word: {str(e)}[/]")

    except psycopg2.OperationalError as db_err:
        logger.error(f"Database connection failed during lookup: {db_err}", exc_info=True)
        console = Console() # Ensure console exists for error message
        console.print(f"[bold red]Database connection error:[/]\n[dim]{db_err}[/]")
    except Exception as e:
        logger.error(f"Error during word lookup: {str(e)}", exc_info=True)
        console = Console() # Ensure console exists
        console.print(f"[red]An unexpected error occurred during lookup: {str(e)}[/]")
    finally:
        if conn is not None:
            try:
                conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


def display_dictionary_stats_cli(args):
    # setup_logging_basic() # Basic logging for CLI, consider if needed or handled by main
    logger.info(f"Display dictionary stats called with args: {args}")
    display_dictionary_stats(args=args) # Pass args to the main function


@with_transaction(commit=False)
def display_dictionary_stats(cur, args):
    """Display comprehensive dictionary statistics."""
    console = Console()
    conn = cur.connection # Get connection from cursor for potential rollback

    # --- Overall Statistics ---
    try:
        console.print("\n[bold]Dictionary Statistics[/]")
        overall_table = Table(title="[bold blue]Overall Statistics[/]", box=box_ROUNDED)
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Count", justify="right", style="green")
        overall_table.add_column("Details", style="dim")

        cur.execute(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'words' AND table_schema = 'public'
        """
        )
        available_columns = {row[0] for row in cur.fetchall()}

        # Pre-fetch schema data needed for checks (if not already available)
        # This is a placeholder; ideally, schema info would be passed or cached.
        schema_data = {} # Initialize schema_data HERE, before basic_queries
        # TODO: Implement schema fetching or passing if needed for robust column/table checks
        # logger.warning("Schema checks in stats are currently disabled; results may be inaccurate if tables/columns are missing.")

        basic_queries = {
            # --- Counts from 'words' table ---
            "Total Words": ("SELECT COUNT(*) FROM words", None),
            "Words with Root": (
                "SELECT COUNT(*) FROM words WHERE root_word_id IS NOT NULL",
                None,
            ) if "root_word_id" in available_columns else None,
            "Proper Nouns": (
                "SELECT COUNT(*) FROM words WHERE is_proper_noun = TRUE",
                None,
            ) if "is_proper_noun" in available_columns else None,
            "Abbreviations": (
                "SELECT COUNT(*) FROM words WHERE is_abbreviation = TRUE",
                None,
            ) if "is_abbreviation" in available_columns else None,
            "Initialisms": (
                "SELECT COUNT(*) FROM words WHERE is_initialism = TRUE",
                None,
            ) if "is_initialism" in available_columns else None,
             "Words with Baybayin": (
                "SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE",
                None,
            ) if "has_baybayin" in available_columns else None,
            "Words with Idioms": (
                "SELECT COUNT(*) FROM words WHERE idioms IS NOT NULL AND jsonb_array_length(idioms) > 0",
                 None,
            ) if "idioms" in available_columns else None,
            "Words with Tags": (
                "SELECT COUNT(*) FROM words WHERE tags IS NOT NULL AND tags != ''",
                 None,
            ) if "tags" in available_columns else None,
             "Words with Word Metadata": (
                "SELECT COUNT(*) FROM words WHERE word_metadata IS NOT NULL AND word_metadata != '{}'::jsonb",
                 None,
            ) if "word_metadata" in available_columns else None,
            "Words with Pronunciation Data (JSON)": (
                 "SELECT COUNT(*) FROM words WHERE pronunciation_data IS NOT NULL AND pronunciation_data != '{}'::jsonb",
                 None,
            ) if "pronunciation_data" in available_columns else None,
             "Words with Hyphenation Data (JSON)": (
                 "SELECT COUNT(*) FROM words WHERE hyphenation IS NOT NULL AND hyphenation != '{}'::jsonb",
                 None,
            ) if "hyphenation" in available_columns else None,

            # --- Counts from 'definitions' table ---
            "Total Definitions": ("SELECT COUNT(*) FROM definitions", None),
             "Definitions with Inline Examples": ( # Clarified metric
                """
                SELECT COUNT(*) 
                FROM definitions 
                WHERE examples IS NOT NULL AND examples != ''
            """,
                """
                SELECT COUNT(DISTINCT word_id)
                FROM definitions
                WHERE examples IS NOT NULL AND examples != ''
            """,
            ), # Removed the schema_data check here
            "Definitions with Linked Examples": (
                 "SELECT COUNT(DISTINCT definition_id) FROM definition_examples",
                 None, # Could add detail on total examples if needed
            ), # Removed the schema_data check here
            "Definitions with Usage Notes": (
                 "SELECT COUNT(*) FROM definitions WHERE usage_notes IS NOT NULL AND usage_notes != ''",
                 "SELECT COUNT(DISTINCT word_id) FROM definitions WHERE usage_notes IS NOT NULL AND usage_notes != ''",
            ), # Removed the schema_data check here
            "Definitions with Tags": (
                 "SELECT COUNT(*) FROM definitions WHERE tags IS NOT NULL AND tags != ''",
                 "SELECT COUNT(DISTINCT word_id) FROM definitions WHERE tags IS NOT NULL AND tags != ''",
            ), # Removed the schema_data check here
            "Definitions with Definition Metadata": (
                 "SELECT COUNT(*) FROM definitions WHERE definition_metadata IS NOT NULL AND definition_metadata != '{}'::jsonb",
                 "SELECT COUNT(DISTINCT word_id) FROM definitions WHERE definition_metadata IS NOT NULL AND definition_metadata != '{}'::jsonb",
            ), # Removed the schema_data check here

            # --- Counts from related tables ---
            "Total Relations": ("SELECT COUNT(*) FROM relations", None), # Removed check
            "Relations with Metadata": (
                 "SELECT COUNT(*) FROM relations WHERE metadata IS NOT NULL AND metadata != '{}'::jsonb",
                 None,
            ), # Removed check
            "Total Etymologies": ("SELECT COUNT(*) FROM etymologies", None), # Removed check
            "Words with Etymology (Distinct)": (
                """
                SELECT COUNT(DISTINCT word_id) 
                FROM etymologies
            """,
                None,
            ), # Removed check
            "Total Pronunciations (Entries)": ("SELECT COUNT(*) FROM pronunciations", None), # Removed check
             "Words with Pronunciation Entries (Distinct)": (
                """
                SELECT COUNT(DISTINCT word_id) 
                FROM pronunciations
            """,
                None,
            ), # Removed check
            "Total Credits": ("SELECT COUNT(*) FROM credits", None), # Removed check
        }

        # Filter out None values from basic_queries in case columns/tables don't exist
        # (This primarily filters based on `available_columns` checks now)
        active_basic_queries = {k: v for k, v in basic_queries.items() if v is not None}

        # logger.warning("Schema checks in stats are currently disabled; results may be inaccurate if tables/columns are missing.")

        total_word_count_for_overall = 0 # Store total word count for details calc
        for label, queries in active_basic_queries.items():
            query = queries[0]
            detail_query = queries[1] if len(queries) > 1 else None
            try:
                cur.execute(query)
                count_result = cur.fetchone()
                count = count_result[0] if count_result else 0

                if label == "Total Words":
                    total_word_count_for_overall = count # Capture for percentage calc

                details = ""
                if detail_query:
                    cur.execute(detail_query)
                    detail_result = cur.fetchone()
                    detail_count = detail_result[0] if detail_result else 0

                    # Improved detail formatting
                    if detail_count > 0 and total_word_count_for_overall > 0:
                        percent_words = (detail_count / total_word_count_for_overall * 100)
                        details = f"from {detail_count:,} unique words ({percent_words:.1f}%)"
                    elif detail_count > 0:
                         details = f"from {detail_count:,} unique words"
                    else:
                        details = "(0 unique words)" # Explicitly show zero

                overall_table.add_row(label, f"{count:,}", details or "")
            except psycopg2.Error as e:
                logger.warning(f"Error getting overall stats for {label}: {e}")
                overall_table.add_row(label, "[red]Error[/]", f"{e.pgcode}")
                conn.rollback() # Reset transaction state after error in this section
            except Exception as e:
                 logger.warning(f"Non-DB error getting overall stats for {label}: {e}", exc_info=True) # Added exc_info
                 overall_table.add_row(label, "[red]Error[/]", "Non-DB Err")

        console.print(overall_table)

    except Exception as e:
        logger.error(f"Error displaying Overall statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error displaying Overall statistics: {str(e)}[/]")
        if conn and not conn.closed:
            try:
                conn.rollback() # Ensure rollback if error occurs early
            except psycopg2.Error as rb_err:
                 logger.warning(f"Rollback failed after Overall stats error: {rb_err}")

    # --- Language Statistics ---
    try:
        cur.execute(
            """
            WITH LangCounts AS (
                SELECT 
                    w.language_code,
                    COUNT(DISTINCT w.id) as word_count -- Count distinct words per language
                FROM words w
                GROUP BY w.language_code
            ), TotalWords AS (
                SELECT SUM(word_count) as total_count FROM LangCounts
            )
            SELECT 
                lc.language_code,
                lc.word_count,
                COUNT(DISTINCT d.id) as def_count,
                COUNT(DISTINCT e.id) as etym_count,
                COUNT(DISTINCT p.id) as pron_count,
                COUNT(DISTINCT CASE WHEN w_main.has_baybayin THEN w_main.id END) as baybayin_count,
                tw.total_count -- Include total word count for percentage calculation
            FROM LangCounts lc
            JOIN words w_main ON lc.language_code = w_main.language_code -- Join back to words to link other tables
            LEFT JOIN definitions d ON w_main.id = d.word_id
            LEFT JOIN etymologies e ON w_main.id = e.word_id
            LEFT JOIN pronunciations p ON w_main.id = p.word_id
            CROSS JOIN TotalWords tw -- Get the total word count
            GROUP BY lc.language_code, lc.word_count, tw.total_count -- Group by per-language word count and total
            ORDER BY lc.word_count DESC;
        """
        )

        lang_table = Table(title="[bold blue]Words by Language[/]", box=box_ROUNDED)
        lang_table.add_column("Language", style="yellow")
        lang_table.add_column("Words (%)", justify="right", style="green") # Combined Col
        lang_table.add_column("Definitions", justify="right", style="green")
        lang_table.add_column("Etymologies", justify="right", style="green")
        lang_table.add_column("Pronunciations", justify="right", style="green")
        lang_table.add_column("Baybayin", justify="right", style="green")

        results = cur.fetchall()
        if results:
            total_words = results[0][-1] if results else 0 # Get total from first row

            for lang_code, words, defs, etyms, prons, bayb, _ in results:
                percentage = (words / total_words * 100) if total_words > 0 else 0
                lang_table.add_row(
                    lang_code or "[NULL]", # Handle potential NULL lang code
                    f"{words:,} ({percentage:.1f}%)",
                    f"{defs:,}",
                    f"{etyms:,}",
                    f"{prons:,}",
                    f"{bayb:,}",
                )
            console.print(lang_table)
        else:
             console.print("[yellow]No language statistics found.[/]")

    except psycopg2.Error as e:
        logger.error(f"Error displaying language statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error displaying language statistics: {e.pgcode}[/]")
        if conn and not conn.closed: conn.rollback() # Reset transaction state
    except Exception as e:
         logger.error(f"Non-DB error displaying language statistics: {str(e)}", exc_info=True)
         console.print(f"[red]Error displaying language statistics: Non-DB Err[/]")


    # --- Parts of Speech Statistics ---
    try:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = 'definitions' AND column_name = 'standardized_pos_id'
            )
        """
        )
        has_standardized_pos = cur.fetchone()[0]

        if has_standardized_pos:
            cur.execute(
                """
                SELECT 
                    p.name_tl,
                    COUNT(*) as count,
                    COUNT(DISTINCT d.word_id) as unique_words,
                    COUNT(CASE WHEN d.examples IS NOT NULL AND d.examples != '' THEN 1 END) as with_examples -- Fixed condition
                FROM definitions d
                JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                GROUP BY p.name_tl
                ORDER BY count DESC
            """
            )
        else:
            # Fallback if standardized_pos_id doesn't exist
             cur.execute(
                """
                SELECT 
                    COALESCE(original_pos, 'Unknown'), -- Use original_pos
                    COUNT(*) as count,
                    COUNT(DISTINCT word_id) as unique_words,
                    COUNT(CASE WHEN examples IS NOT NULL AND examples != '' THEN 1 END) as with_examples -- Fixed condition
                FROM definitions
                GROUP BY original_pos -- Group by original_pos
                ORDER BY count DESC
            """
            )


        pos_table = Table(title="[bold blue]Parts of Speech[/]", box=box_ROUNDED)
        pos_table.add_column("Part of Speech", style="yellow")
        pos_table.add_column("Definitions", justify="right", style="green")
        pos_table.add_column("Unique Words", justify="right", style="green")
        pos_table.add_column("With Examples", justify="right", style="green")

        pos_results = cur.fetchall()
        if pos_results:
            for pos, count, unique_words, with_examples in pos_results:
                pos_table.add_row(
                    pos or "Uncategorized",
                    f"{count:,}",
                    f"{unique_words:,}",
                    f"{with_examples:,}", # Show the corrected count
                )
            console.print()
            console.print(pos_table)
        else:
            console.print("\n[yellow]No Part of Speech statistics found.[/]")

    except psycopg2.Error as e:
        logger.error(f"Error displaying part of speech statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error displaying Part of Speech statistics: {e.pgcode}[/]")
        if conn and not conn.closed: conn.rollback() # Reset transaction state
    except Exception as e:
        logger.error(f"Non-DB error displaying part of speech statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error displaying Part of Speech statistics: Non-DB Err[/]")


    # --- Relationship Statistics ---
    try:
        cur.execute(
            """
            SELECT 
                r.relation_type,
                COUNT(*) as count,
                COUNT(DISTINCT r.from_word_id) as unique_sources,
                COUNT(DISTINCT r.to_word_id) as unique_targets
            FROM relations r
            GROUP BY r.relation_type
            ORDER BY count DESC
        """
        )

        rel_results = cur.fetchall()
        if rel_results:
            rel_table = Table(
                title="[bold blue]Relationship Types[/]", box=box_ROUNDED
            )
            rel_table.add_column("Type", style="yellow")
            rel_table.add_column("Total", justify="right", style="green")
            rel_table.add_column("Unique Sources", justify="right", style="green")
            rel_table.add_column("Unique Targets", justify="right", style="green")

            for rel_type, count, sources, targets in rel_results:
                rel_table.add_row(
                    rel_type or "Unknown",
                    f"{count:,}",
                    f"{sources:,}",
                    f"{targets:,}",
                )
            console.print()
            console.print(rel_table)
        else:
             console.print("\n[yellow]No Relationship statistics found.[/]")

    except psycopg2.Error as e:
        logger.error(f"Error displaying relationship statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error displaying Relationship statistics: {e.pgcode}[/]")
        if conn and not conn.closed: conn.rollback() # Reset transaction state
    except Exception as e:
         logger.error(f"Non-DB error displaying relationship statistics: {str(e)}", exc_info=True)
         console.print(f"[red]Error displaying Relationship statistics: Non-DB Err[/]")


    # --- Source Statistics ---
    try:
        # Check which source columns exist
        cur.execute(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name IN ('words', 'definitions') AND table_schema = 'public'
            AND column_name IN ('source_info', 'sources')
        """
        )
        available_source_columns = {row[0] for row in cur.fetchall()}

        # Source Stats from words.source_info (JSONB)
        if "source_info" in available_source_columns:
            try:
                # Ensure the correct query with JSON extraction is used
                cur.execute(
                    """
                    SELECT
                        COALESCE((source_info -> 'files') ->> 0, 'Unknown') as source_name, -- Extract first element of 'files' array as text
                        COUNT(*) as word_count
                    FROM words
                    GROUP BY source_name -- Group by the extracted source name
                    ORDER BY word_count DESC;
                """
                )
                source_results = cur.fetchall()
                if source_results:
                    source_table = Table(
                        title="[bold blue]Source Distribution (from Words)[/]", box=box_ROUNDED
                    )
                    source_table.add_column("Source", style="yellow")
                    source_table.add_column("Words", justify="right", style="green")

                    for source, count in source_results:
                        # The ->> operator extracts as text, should be clean
                        display_source = source if source else 'Unknown'
                        source_table.add_row(display_source, f"{count:,}")

                    console.print()
                    console.print(source_table)
                else:
                    console.print("\n[yellow]No source statistics found in 'words' table.[/]")

            except psycopg2.Error as e:
                # Catch error specifically for this query
                logger.error(f"Error querying words.source_info statistics: {str(e)}", exc_info=True)
                console.print(f"[red]Error displaying Word Source statistics: {e.pgcode}[/]")
                if conn and not conn.closed: conn.rollback() # Rollback this specific error
            except Exception as e:
                 logger.error(f"Non-DB error querying words.source_info statistics: {str(e)}", exc_info=True)
                 console.print(f"[red]Error displaying Word Source statistics: Non-DB Err[/]")


        # Source Stats from definitions.sources (TEXT)
        if "sources" in available_source_columns:
            try:
                cur.execute(
                    """
                    SELECT 
                        COALESCE(sources, 'Unknown') as source_name,
                        COUNT(*) as def_count,
                        COUNT(DISTINCT word_id) as word_count,
                        COUNT(CASE WHEN examples IS NOT NULL AND examples != '' THEN 1 END) as example_count -- Fixed condition
                    FROM definitions
                    GROUP BY sources
                    ORDER BY def_count DESC;
                """
                )

                def_source_results = cur.fetchall()
                if def_source_results:
                    def_source_table = Table(
                        title="[bold blue]Definition Sources[/]", box=box_ROUNDED
                    )
                    def_source_table.add_column("Source", style="yellow")
                    def_source_table.add_column(
                        "Definitions", justify="right", style="green"
                    )
                    def_source_table.add_column("Unique Words", justify="right", style="green")
                    def_source_table.add_column(
                        "With Examples", justify="right", style="green"
                    )

                    for source, def_count, word_count, example_count in def_source_results:
                        def_source_table.add_row(
                            source or "Unknown",
                            f"{def_count:,}",
                            f"{word_count:,}",
                            f"{example_count:,}", # Show corrected count
                        )

                    console.print()
                    console.print(def_source_table)
                else:
                     console.print("\n[yellow]No source statistics found in 'definitions' table.[/]")
            except psycopg2.Error as e:
                 # Catch error specifically for this query
                logger.error(f"Error querying definitions.sources statistics: {str(e)}", exc_info=True)
                console.print(f"[red]Error displaying Definition Source statistics: {e.pgcode}[/]")
                if conn and not conn.closed: conn.rollback() # Rollback this specific error
            except Exception as e:
                 logger.error(f"Non-DB error querying definitions.sources statistics: {str(e)}", exc_info=True)
                 console.print(f"[red]Error displaying Definition Source statistics: Non-DB Err[/]")

    except Exception as e:
        # Catch errors in the outer source stats logic (e.g., checking columns)
        logger.error(f"Error displaying Source statistics section: {str(e)}", exc_info=True)
        console.print(f"[red]Error displaying Source statistics section: {str(e)}[/]")
        if conn and not conn.closed:
            try:
                conn.rollback() # Rollback if outer section fails
            except psycopg2.Error as rb_err:
                 logger.warning(f"Rollback failed after Source stats error: {rb_err}")


    # --- Baybayin Statistics ---
    try:
        # Re-fetch available columns in case transaction was rolled back
        cur.execute(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'words' AND table_schema = 'public'
        """
        )
        available_columns = {row[0] for row in cur.fetchall()}

        baybayin_table = Table(
            title="[bold blue]Baybayin Statistics[/]", box=box_ROUNDED
        )
        baybayin_table.add_column("Metric", style="yellow")
        baybayin_table.add_column("Count", justify="right", style="green")
        baybayin_table.add_column("Details", style="dim")

        baybayin_queries = {}
        if "baybayin_form" in available_columns:
             baybayin_queries["Total Baybayin Forms"] = (
                "SELECT COUNT(*) FROM words WHERE baybayin_form IS NOT NULL",
                """SELECT COUNT(DISTINCT language_code) 
                FROM words WHERE baybayin_form IS NOT NULL""",
            )
        if "romanized_form" in available_columns:
             baybayin_queries["With Romanization"] = (
                "SELECT COUNT(*) FROM words WHERE romanized_form IS NOT NULL",
                None,
            )
        if "has_baybayin" in available_columns and "baybayin_form" in available_columns:
            baybayin_queries["Verified Forms (has_baybayin=T)"] = ( # Clarified metric
                """SELECT COUNT(*) FROM words 
                WHERE has_baybayin = TRUE 
                AND baybayin_form IS NOT NULL""",
                None,
            )

        if "badlit_form" in available_columns:
            baybayin_queries["With Badlit"] = (
                "SELECT COUNT(*) FROM words WHERE badlit_form IS NOT NULL",
                None,
            )
            if "baybayin_form" in available_columns and "romanized_form" in available_columns:
                baybayin_queries["Complete Forms (Baybayin+Roman+Badlit)"] = ( # Clarified metric
                    """SELECT COUNT(*) FROM words 
                    WHERE baybayin_form IS NOT NULL 
                    AND romanized_form IS NOT NULL 
                    AND badlit_form IS NOT NULL""",
                    None,
                )

        if baybayin_queries: # Only proceed if there are queries to run
            for label, queries in baybayin_queries.items():
                query = queries[0]
                detail_query = queries[1] if len(queries) > 1 else None
                try:
                    cur.execute(query)
                    count = cur.fetchone()[0]
                    details = ""
                    if detail_query:
                        cur.execute(detail_query)
                        details = f"across {cur.fetchone()[0]} languages"
                    baybayin_table.add_row(label, f"{count:,}", details or "")
                except psycopg2.Error as e:
                    logger.warning(f"Error getting Baybayin stats for {label}: {e}")
                    baybayin_table.add_row(label, "[red]Error[/]", f"{e.pgcode}")
                    # No rollback here, let the outer handler manage final state
                except Exception as e:
                     logger.warning(f"Non-DB error getting Baybayin stats for {label}: {e}")
                     baybayin_table.add_row(label, "[red]Error[/]", "Non-DB Err")

            console.print()
            console.print(baybayin_table)
        else:
            console.print("\n[yellow]No Baybayin related columns found for statistics.[/]")

    except psycopg2.Error as e:
        logger.error(f"Error displaying Baybayin statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error displaying Baybayin statistics: {e.pgcode}[/]")
        # No rollback here, final state handled by decorator
    except Exception as e:
        logger.error(f"Non-DB error displaying Baybayin statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error displaying Baybayin statistics: Non-DB Err[/]")


    # Print timestamp
    console.print(
        f"\n[dim]Statistics generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]"
    )

# --- End of display_dictionary_stats function ---

def normalize_source_name(source_str: Optional[str]) -> str:
    """Standardizes source names using common patterns."""
    if not source_str:
        return "Unknown"
    
    s_lower = source_str.lower()
    
    # More specific checks first
    if re.search(r'marayum', s_lower): return "Project Marayum"
    if re.search(r'kaikki.*ceb', s_lower): return "kaikki.org (Cebuano)"
    if re.search(r'kaikki', s_lower): return "kaikki.org (Tagalog)" # General kaikki after specific ceb
    if re.search(r'kwf', s_lower): return "KWF Diksiyonaryo"
    if re.search(r'tagalog\.com|root_words', s_lower): return "tagalog.com"
    if re.search(r'diksiyonaryo\.ph|tagalog-words', s_lower): return "diksiyonaryo.ph"
    if re.search(r'calderon', s_lower): return "Calderon Diccionario 1915"
    if re.search(r'gay.?slang', s_lower): return "Philippine Slang/Gay Dictionary"
    
    # Fallback to the original string if no pattern matches
    # Optionally, could add more patterns or cleanup here
    return source_str # Return original if no match


@with_transaction(commit=False) # Read-only operations
def display_leaderboard(cur, console, args):
    """Displays various contribution leaderboards, respecting limit and sort_by args."""
    # Ensure cursor is DictCursor for name-based access
    if not isinstance(cur.description, list) or not all(isinstance(col, psycopg2.extensions.Column) for col in cur.description):
         # This check might be overly simplistic. A better check might be needed
         # If not a DictCursor, raise an error or try to get one.
         # For now, assume the caller provided the right cursor type.
         logger.warning("display_leaderboard expected a DictCursor, but might have received a standard cursor.")

    limit = args.limit
    sort_by = args.sort_by.lower() if args.sort_by else 'count' # Default to count if not specified
    
    logger.info(f"Generating leaderboard with limit={limit}, sort_by='{sort_by}'")

    console.print(
        "\n[bold magenta underline]📊 Dictionary Contributors Leaderboard[/]\n"
    )

    # --- Overall Stats ---
    # (This section remains unchanged as limit/sort don't apply)
    overall_stats_table = Table(
        title="[bold blue]Overall Statistics[/]",
        box=box_ROUNDED,
        show_header=False,
    )
    overall_stats_table.add_column("Statistic", style="cyan")
    overall_stats_table.add_column("Value", justify="right", style="green")
    try:
        cur.execute("SELECT COUNT(*) FROM words")
        total_words = cur.fetchone()[0]
        overall_stats_table.add_row("Total Words", f"{total_words:,}")
        
        cur.execute("SELECT COUNT(*) FROM definitions")
        total_definitions = cur.fetchone()[0]
        overall_stats_table.add_row(
            "Total Definitions", f"{total_definitions:,}"
        )

        cur.execute("SELECT COUNT(*) FROM relations")
        total_relations = cur.fetchone()[0]
        overall_stats_table.add_row("Total Relations", f"{total_relations:,}")

        cur.execute("SELECT COUNT(*) FROM etymologies")
        total_etymologies = cur.fetchone()[0]
        overall_stats_table.add_row(
            "Total Etymologies", f"{total_etymologies:,}"
        )

        cur.execute(
            "SELECT COUNT(DISTINCT standardized_pos_id) FROM definitions WHERE standardized_pos_id IS NOT NULL"
        )
        total_pos = cur.fetchone()[0]
        overall_stats_table.add_row("Unique Parts of Speech", str(total_pos))

        cur.execute(
            "SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE OR baybayin_form IS NOT NULL"
        )
        words_with_baybayin = cur.fetchone()[0]
        overall_stats_table.add_row(
            "Words w/ Baybayin", f"{words_with_baybayin:,}"
        )
        console.print(overall_stats_table)
        console.print()
    except Exception as e:
        logger.error(f"Error generating overall statistics: {str(e)}", exc_info=True)
        console.print(
            f"[yellow]Could not generate overall statistics: {str(e)}[/]"
        )
        if cur.connection and not cur.connection.closed: cur.connection.rollback() # Ensure rollback if error occurs

    # --- Definition Contributors ---
    try:
        valid_def_sorts = {
            'definitions': 'def_count', 'defs': 'def_count', 'count': 'def_count', 
            'words': 'unique_words', 'examples': 'with_examples', 'notes': 'with_notes', 
            'pos': 'pos_count', 'example_coverage': 'example_percentage', 
            'notes_coverage': 'notes_percentage'
        }
        def_order_by_col = valid_def_sorts.get(sort_by, 'def_count')
        logger.info(f"Sorting Definition Contributors by: {def_order_by_col}")

        # Use Python triple-quoted f-string for the SQL query
        def_query = f'''
            WITH source_mapping AS (
                        SELECT
                    id,
                            CASE
                                WHEN sources ILIKE '%%project marayum%%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%%marayum%%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%%kaikki-ceb%%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%%kaikki.jsonl%%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%%kaikki%%' AND sources ILIKE '%%ceb%%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%%kaikki%%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%%kwf%%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%%kwf_dictionary%%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%%tagalog.com%%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%%root_words%%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%%diksiyonaryo.ph%%' THEN 'diksiyonaryo.ph'
                                WHEN sources ILIKE '%%tagalog-words%%' THEN 'diksiyonaryo.ph'
                        WHEN sources ILIKE '%%calderon%%' THEN 'Calderon Diccionario 1915'
                        WHEN sources ILIKE '%%gay-slang%%' OR sources ILIKE '%%gay slang%%' THEN 'Philippine Slang/Gay Dictionary'
                                ELSE COALESCE(sources, 'Unknown')
                    END AS normalized_source_name,
                    word_id,
                    standardized_pos_id,
                    (examples IS NOT NULL AND examples != \'\') as has_example,
                    (usage_notes IS NOT NULL AND usage_notes != \'\') as has_notes
                FROM definitions
            ),
            source_stats AS (
                SELECT
                    normalized_source_name,
                            COUNT(*) AS def_count,
                            COUNT(DISTINCT word_id) AS unique_words,
                    COUNT(CASE WHEN has_example THEN 1 END) AS with_examples,
                            COUNT(DISTINCT standardized_pos_id) AS pos_count,
                    COUNT(CASE WHEN has_notes THEN 1 END) AS with_notes
                FROM source_mapping
                GROUP BY normalized_source_name
                    )
                    SELECT
                normalized_source_name AS source_name, 
                        def_count,
                        unique_words,
                        with_examples,
                        pos_count,
                        with_notes,
                        ROUND(100.0 * with_examples / NULLIF(def_count, 0), 1) as example_percentage,
                        ROUND(100.0 * with_notes / NULLIF(def_count, 0), 1) as notes_percentage
                    FROM source_stats
            ORDER BY {def_order_by_col} DESC NULLS LAST
            LIMIT %s
            '''
        
        cur.execute(def_query, (limit,))
        def_results = cur.fetchall()
        
        if def_results:
            def_table = Table(
                title=f"[bold blue]Top {limit} Definition Contributors (Sorted by {sort_by})[/]", 
                box=box_ROUNDED
            )
            def_table.add_column("Source", style="yellow")
            def_table.add_column("Definitions", justify="right", style="green")
            def_table.add_column("Words", justify="right", style="green")
            def_table.add_column("Examples", justify="right", style="cyan")
            def_table.add_column("POS Types", justify="right", style="cyan")
            def_table.add_column("Notes", justify="right", style="cyan")
            def_table.add_column("Coverage (%)", style="dim") # Combined coverage

            for row in def_results:
                ex_pct = row.get('example_percentage', 0) or 0.0
                notes_pct = row.get('notes_percentage', 0) or 0.0
                coverage = f"Ex: {ex_pct:.1f}, Notes: {notes_pct:.1f}"

                def_table.add_row(
                    row.get('source_name', 'Unknown'),
                    f"{row.get('def_count', 0):,}",
                    f"{row.get('unique_words', 0):,}",
                    f"{row.get('with_examples', 0):,}",
                    str(row.get('pos_count', 0)),
                    f"{row.get('with_notes', 0):,}",
                    coverage,
                )
            console.print(def_table)
            console.print()
        else:
             console.print("[yellow]No definition contributor data found.[/]")

    except psycopg2.Error as db_err:
        logger.error(f"Error generating definition statistics: {str(db_err)}", exc_info=True)
        console.print(
            f"[red]Error generating definition statistics: {db_err.pgcode}[/]"
        )
        if cur.connection and not cur.connection.closed: cur.connection.rollback() # Ensure rollback
    except Exception as e:
        logger.error(f"Non-DB error generating definition statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error: {str(e)}[/]")


    # --- Etymology Contributors ---
    try:
        valid_etym_sorts = {
            'etymologies': 'etym_count', 'count': 'etym_count', 'words': 'unique_words',
            'components': 'with_components', 'langs': 'with_lang_codes',
            'comp_coverage': 'comp_percentage', 'lang_coverage': 'lang_percentage'
        }
        etym_order_by_col = valid_etym_sorts.get(sort_by, 'etym_count')
        logger.info(f"Sorting Etymology Contributors by: {etym_order_by_col}")

        etym_query = f'''
            WITH source_mapping AS (
                        SELECT
                            CASE
                                WHEN sources ILIKE '%%project marayum%%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%%marayum%%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%%kaikki-ceb%%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%%kaikki.jsonl%%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%%kaikki%%' AND sources ILIKE '%%ceb%%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%%kaikki%%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%%kwf%%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%%kwf_dictionary%%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%%tagalog.com%%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%%root_words%%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%%diksiyonaryo.ph%%' THEN 'diksiyonaryo.ph'
                                WHEN sources ILIKE '%%tagalog-words%%' THEN 'diksiyonaryo.ph'
                        WHEN sources ILIKE '%%calderon%%' THEN 'Calderon Diccionario 1915'
                        WHEN sources ILIKE '%%gay-slang%%' OR sources ILIKE '%%gay slang%%' THEN 'Philippine Slang/Gay Dictionary'
                                ELSE COALESCE(sources, 'Unknown')
                    END AS normalized_source_name,
                    word_id,
                    CASE 
                        WHEN normalized_components LIKE '[%%' AND normalized_components LIKE '%%]' THEN -- Check if it looks like an array
                            (jsonb_typeof(normalized_components::jsonb) = 'array' AND jsonb_array_length(normalized_components::jsonb) > 0)
                        ELSE FALSE
                    END as has_components,
                    CASE
                        WHEN language_codes LIKE '[%%' AND language_codes LIKE '%%]' THEN -- Check if it looks like an array
                            (jsonb_typeof(language_codes::jsonb) = 'array' AND jsonb_array_length(language_codes::jsonb) > 0)
                        ELSE FALSE
                    END as has_lang_codes
                FROM etymologies
            ),
            etym_stats AS (
                SELECT
                    normalized_source_name,
                            COUNT(*) AS etym_count,
                            COUNT(DISTINCT word_id) AS unique_words,
                    COUNT(CASE WHEN has_components THEN 1 END) AS with_components,
                    COUNT(CASE WHEN has_lang_codes THEN 1 END) AS with_lang_codes
                FROM source_mapping
                GROUP BY normalized_source_name
            )
            SELECT 
                normalized_source_name AS source_name, 
                etym_count,
                unique_words,
                with_components,
                with_lang_codes,
                        ROUND(100.0 * with_components / NULLIF(etym_count, 0), 1) as comp_percentage,
                        ROUND(100.0 * with_lang_codes / NULLIF(etym_count, 0), 1) as lang_percentage
                    FROM etym_stats
            ORDER BY {etym_order_by_col} DESC NULLS LAST
            LIMIT %s
            '''
        
        cur.execute(etym_query, (limit,))
        etym_results = cur.fetchall()

        if etym_results:
            etym_table = Table(
                title=f"[bold blue]Top {limit} Etymology Contributors (Sorted by {sort_by})[/]", 
                box=box_ROUNDED
            )
            etym_table.add_column("Source", style="yellow")
            etym_table.add_column("Etymologies", justify="right", style="green")
            etym_table.add_column("Words", justify="right", style="green")
            etym_table.add_column("Components", justify="right", style="cyan")
            etym_table.add_column("Lang Codes", justify="right", style="cyan")
            etym_table.add_column("Coverage (%)", style="dim") # Combined coverage

            for row in etym_results:
                 comp_pct = row.get('comp_percentage', 0) or 0.0
                 lang_pct = row.get('lang_percentage', 0) or 0.0
                 coverage = f"Comp: {comp_pct:.1f}, Lang: {lang_pct:.1f}"

                 etym_table.add_row(
                    row.get('source_name', 'Unknown'),
                    f"{row.get('etym_count', 0):,}",
                    f"{row.get('unique_words', 0):,}",
                    f"{row.get('with_components', 0):,}",
                    f"{row.get('with_lang_codes', 0):,}",
                    coverage,
                )
            console.print(etym_table)
            console.print()
        else:
             console.print("[yellow]No etymology contributor data found.[/]")
             
    except psycopg2.Error as db_err:
        logger.error(f"Error generating etymology statistics: {str(db_err)}", exc_info=True)
        console.print(
            f"[red]Error generating etymology statistics: {db_err.pgcode}[/]"
        )
        if cur.connection and not cur.connection.closed: cur.connection.rollback() # Ensure rollback
    except Exception as e:
        logger.error(f"Non-DB error generating etymology statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error: {str(e)}[/]")


    # --- Relationship Contributors ---
    # (Limit and sort are less applicable here without more specific metrics)
    try:
        # Use Python triple-quoted f-string for the SQL query
        rel_query = f'''
            WITH source_mapping AS (
                        SELECT
                    relation_type,
                            CASE
                                WHEN sources ILIKE '%%project marayum%%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%%marayum%%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%%kaikki-ceb%%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%%kaikki.jsonl%%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%%kaikki%%' AND sources ILIKE '%%ceb%%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%%kaikki%%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%%kwf%%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%%kwf_dictionary%%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%%tagalog.com%%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%%root_words%%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%%diksiyonaryo.ph%%' THEN 'diksiyonaryo.ph'
                                WHEN sources ILIKE '%%tagalog-words%%' THEN 'diksiyonaryo.ph'
                        WHEN sources ILIKE '%%calderon%%' THEN 'Calderon Diccionario 1915'
                        WHEN sources ILIKE '%%gay-slang%%' OR sources ILIKE '%%gay slang%%' THEN 'Philippine Slang/Gay Dictionary'
                                ELSE COALESCE(sources, 'Unknown')
                    END AS normalized_source_name
                FROM relations
            )
            SELECT 
                normalized_source_name as source_name, 
                            relation_type,
                            COUNT(*) AS rel_count
            FROM source_mapping
                        GROUP BY source_name, relation_type
                    ORDER BY source_name, rel_count DESC
            ''' # No LIMIT here by default, could add if needed later
        
        cur.execute(rel_query)
        rel_results = cur.fetchall()

        if rel_results:
            rel_table = Table(
                title="[bold blue]Relationship Contributors (by Source & Type)[/]", 
                box=box_ROUNDED
            )
            rel_table.add_column("Source", style="yellow")
            rel_table.add_column("Relation Type", style="magenta")
            rel_table.add_column("Count", justify="right", style="green")

            current_source = None
            for row in rel_results:
                source = row.get('source_name', 'Unknown')
                rel_type = row.get('relation_type', 'Unknown')
                count = row.get('rel_count', 0)

                # Group by source visually
                if source != current_source:
                    if current_source is not None: 
                         rel_table.add_row("","","") # Simple empty row separator
                    rel_table.add_row(source, rel_type, f"{count:,}")
                    current_source = source
                else:
                    rel_table.add_row("", rel_type, f"{count:,}") # Blank source for subsequent rows
            console.print(rel_table)
            console.print()
        else:
            console.print("[yellow]No relationship contributor data found.[/]")

    except psycopg2.Error as db_err:
        logger.error(f"Error generating relationship statistics: {str(db_err)}", exc_info=True)
        console.print(
            f"[red]Error generating relationship statistics: {db_err.pgcode}[/]"
        )
        if cur.connection and not cur.connection.closed: cur.connection.rollback()
    except Exception as e:
        logger.error(f"Non-DB error generating relationship statistics: {str(e)}", exc_info=True)
        console.print(f"[red]Error: {str(e)}[/]")
        

    console.print(
        f"[dim]Leaderboard generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]"
    )

# --- End of display_leaderboard function ---


@with_transaction(commit=True)
def purge_database_tables(cur):
    """
    Safely delete all data from all dictionary tables.
    This version uses the transaction decorator for better reliability.
    """
    tables = [
        "definition_relations",
        "affixations",
        "relations",
        "etymologies",
        "definitions",
        "words",
        "parts_of_speech",
    ]

    for table in tables:
        print(f"Purging {table}...")
        cur.execute(f"DELETE FROM {table}")

    return True


def explore_dictionary():
    """Launches interactive dictionary explorer."""
    console = Console()
    console.print(
        "\n[bold cyan]🔍 Interactive Dictionary Explorer[/]", justify="center"
    )
    console.print(
        "[dim]Navigate Filipino dictionary data with ease[/]\n", justify="center"
    )

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        while True:
            console.print("\n[bold]Choose an option:[/]")
            options = [
                "1. Search for a word",
                "2. Browse random words",
                "3. Explore Baybayin words",
                "4. View word relations",
                "5. Show statistics",
                "0. Exit explorer",
            ]

            for option in options:
                console.print(f"  {option}")

            choice = input("Enter your choice (0-5): ")

            if choice == "0":
                break
            elif choice == "1":
                search_term = input("Enter search term: ")
                if not search_term.strip():
                    continue

                cur.execute(
                    """
                    SELECT id, lemma, language_code, has_baybayin
                    FROM words
                    WHERE 
                         lemma ILIKE %s OR
                         normalized_lemma ILIKE %s OR
                         search_text @@ plainto_tsquery('simple', %s)
                    ORDER BY 
                         CASE WHEN lemma ILIKE %s THEN 0
                              WHEN lemma ILIKE %s THEN 1
                              ELSE 2
                         END,
                         length(lemma)
                    LIMIT 20
                """,
                    (
                        f"{search_term}%",
                        f"{search_term}%",
                        search_term,
                        search_term,
                        f"{search_term}%",
                    ),
                )

                results = cur.fetchall()

                if not results:
                    console.print("[yellow]No matches found.[/]")
                    continue

                result_table = Table(
                    title=f"Search Results for '{search_term}'", box=box_ROUNDED
                )
                result_table.add_column("ID", style="dim")
                result_table.add_column("Word", style="cyan")
                result_table.add_column("Language", style="green")
                result_table.add_column("Baybayin", style="magenta")

                for word_id, lemma, lang_code, has_baybayin in results:
                    result_table.add_row(
                        str(word_id),
                        lemma,
                        "Tagalog" if lang_code == "tl" else "Cebuano",
                        "✓" if has_baybayin else "",
                    )

                console.print(result_table)

                word_choice = input(
                    "\nEnter word ID to view details (or press Enter to return): "
                )
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)

            elif choice == "2":
                cur.execute(
                    """
                    SELECT id, lemma, language_code, has_baybayin
                    FROM words
                    ORDER BY RANDOM()
                    LIMIT 10
                """
                )

                results = cur.fetchall()

                if not results:
                    console.print("[yellow]No words found in the database.[/]")
                    continue

                result_table = Table(title="Random Words", box=box_ROUNDED)
                result_table.add_column("ID", style="dim")
                result_table.add_column("Word", style="cyan")
                result_table.add_column("Language", style="green")
                result_table.add_column("Baybayin", style="magenta")

                for word_id, lemma, lang_code, has_baybayin in results:
                    result_table.add_row(
                        str(word_id),
                        lemma,
                        "Tagalog" if lang_code == "tl" else "Cebuano",
                        "✓" if has_baybayin else "",
                    )

                console.print(result_table)

                word_choice = input(
                    "\nEnter word ID to view details (or press Enter to return): "
                )
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)

            elif choice == "3":
                cur.execute(
                    """
                    SELECT id, lemma, baybayin_form, romanized_form
                    FROM words
                    WHERE has_baybayin = TRUE
                    ORDER BY RANDOM()
                    LIMIT 10
                """
                )

                results = cur.fetchall()

                if not results:
                    console.print("[yellow]No Baybayin words found in the database.[/]")
                    continue

                result_table = Table(title="Baybayin Words", box=box_ROUNDED)
                result_table.add_column("ID", style="dim")
                result_table.add_column("Word", style="cyan")
                result_table.add_column("Baybayin", style="magenta")
                result_table.add_column("Romanized", style="green")

                for word_id, lemma, baybayin, romanized in results:
                    result_table.add_row(
                        str(word_id), lemma, baybayin or "", romanized or ""
                    )

                console.print(result_table)

                word_choice = input(
                    "\nEnter word ID to view details (or press Enter to return): "
                )
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)

            elif choice == "4":
                word_input = input("Enter word to find relations: ")
                if not word_input.strip():
                    continue

                cur.execute(
                    """
                    SELECT id FROM words
                    WHERE lemma = %s OR normalized_lemma = %s
                    LIMIT 1
                """,
                    (word_input, normalize_lemma(word_input)),
                )

                result = cur.fetchone()

                if not result:
                    console.print(f"[yellow]Word '{word_input}' not found.[/]")
                    continue

                word_id = result[0]

                cur.execute(
                    """
                    SELECT r.relation_type, w.id, w.lemma
                    FROM relations r
                    JOIN words w ON r.to_word_id = w.id
                    WHERE r.from_word_id = %s
                    ORDER BY r.relation_type, w.lemma
                """,
                    (word_id,),
                )

                relations = cur.fetchall()

                if not relations:
                    console.print(f"[yellow]No relations found for '{word_input}'.[/]")
                    continue

                relation_groups = {}
                for rel_type, rel_id, rel_word in relations:
                    if rel_type not in relation_groups:
                        relation_groups[rel_type] = []
                    relation_groups[rel_type].append((rel_id, rel_word))

                for rel_type, words in relation_groups.items():
                    console.print(f"\n[bold]{rel_type.title()}:[/]")
                    for rel_id, rel_word in words:
                        console.print(f"  {rel_word} (ID: {rel_id})")

                word_choice = input(
                    "\nEnter word ID to view details (or press Enter to return): "
                )
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)

            elif choice == "5":
                cur.execute("SELECT COUNT(*) FROM words")
                word_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM words WHERE language_code = 'tl'")
                tagalog_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM words WHERE language_code = 'ceb'")
                cebuano_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE")
                baybayin_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM definitions")
                def_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM relations")
                rel_count = cur.fetchone()[0]

                stats_table = Table(title="Quick Statistics", box=box_ROUNDED)
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Count", justify="right", style="green")

                stats_table.add_row("Total Words", f"{word_count:,}")
                stats_table.add_row("Tagalog Words", f"{tagalog_count:,}")
                stats_table.add_row("Cebuano Words", f"{cebuano_count:,}")
                stats_table.add_row("Baybayin Words", f"{baybayin_count:,}")
                stats_table.add_row("Definitions", f"{def_count:,}")
                stats_table.add_row("Relations", f"{rel_count:,}")

                console.print(stats_table)
                input("\nPress Enter to continue...")
            else:
                console.print("[yellow]Invalid choice. Please try again.[/]")
    except Exception as e:
        logger.error(f"Error in dictionary explorer: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/]")
    finally:
        if conn:
            conn.close()
        console.print("\n[bold green]Thank you for using the Dictionary Explorer![/]")


@with_transaction(commit=False)
def lookup_by_id(cur, word_id: int, console: Console, format: str = 'rich', debug: bool = False):
    """Look up a word by its ID and display its information in the specified format."""
    if debug:
        logger.info(f"Looking up details for Word ID: {word_id}, Format: {format}")
    
    word_data = {}
    try:
        # 1. Fetch Core Word Info
        cur.execute(
            """
            SELECT w.*, r.lemma as root_lemma
            FROM words w
            LEFT JOIN words r ON w.root_word_id = r.id
            WHERE w.id = %s
        """, (word_id,)
        )
        word_info = cur.fetchone()

        if not word_info:
            message = f"Word with ID {word_id} not found."
            if format == 'rich':
                console.print(f"[yellow]{message}[/]")
            elif format == 'text':
                print(message)
            # JSON format will just output empty data
            return
        
        word_data = dict(word_info) # Convert Row object to dict
        # Clean up potentially large/binary fields for JSON/text
        word_data.pop('search_vector', None) 
        # Convert datetime/jsonb for JSON serialization if needed (handled by json.dumps default)

        # 2. Fetch Definitions
        cur.execute(
            """
            SELECT d.*, p.name_tl as pos_tl, p.name_en as pos_en
            FROM definitions d
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE d.word_id = %s
            ORDER BY p.id NULLS LAST, d.created_at
        """, (word_id,)
        )
        definitions = [dict(row) for row in cur.fetchall()]
        word_data['definitions'] = definitions

        # 3. Fetch Relations (Both directions)
        cur.execute(
            """
            SELECT r.relation_type, r.to_word_id as related_word_id, w.lemma as related_lemma, 'outgoing' as direction
            FROM relations r JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = %s
            UNION ALL
            SELECT r.relation_type, r.from_word_id as related_word_id, w.lemma as related_lemma, 'incoming' as direction
            FROM relations r JOIN words w ON r.from_word_id = w.id
            WHERE r.to_word_id = %s
            ORDER BY relation_type, direction, related_lemma
        """, (word_id, word_id)
        )
        relations = [dict(row) for row in cur.fetchall()]
        word_data['relations'] = relations

        # 4. Fetch Etymologies
        cur.execute(
            "SELECT * FROM etymologies WHERE word_id = %s ORDER BY created_at", (word_id,)
        )
        etymologies = [dict(row) for row in cur.fetchall()]
        word_data['etymologies'] = etymologies
        
        # 5. Fetch Pronunciations
        cur.execute(
            "SELECT * FROM pronunciations WHERE word_id = %s ORDER BY created_at", (word_id,)
        )
        pronunciations = [dict(row) for row in cur.fetchall()]
        word_data['pronunciations'] = pronunciations

        # 6. Fetch Word Forms
        cur.execute(
            "SELECT * FROM word_forms WHERE word_id = %s ORDER BY is_canonical DESC, form", (word_id,)
        )
        word_forms = [dict(row) for row in cur.fetchall()]
        word_data['word_forms'] = word_forms
        
        # --- Output Formatting ---
        if format == 'json':
            # Default json handler deals with date/jsonb etc reasonably well
            print(json.dumps(word_data, indent=2, default=str)) 
        
        elif format == 'text':
            print(f"--- Word ID: {word_data.get('id')} ---")
            print(f"Lemma: {word_data.get('lemma')}")
            print(f"Language: {word_data.get('language_code')}")
            if word_data.get('root_lemma'):
                print(f"Root: {word_data.get('root_lemma')} (ID: {word_data.get('root_word_id')})")
            if word_data.get('has_baybayin') and word_data.get('baybayin_form'):
                print(f"Baybayin: {word_data.get('baybayin_form')}")
                if word_data.get('romanized_form'): print(f"Romanized: {word_data.get('romanized_form')}")
            # Add other simple fields: is_proper_noun, is_abbreviation etc.
            for key in ['is_proper_noun', 'is_abbreviation', 'is_initialism', 'tags']:
                 if word_data.get(key):
                     print(f"{key.replace('_', ' ').title()}: {word_data.get(key)}")
            
            if word_data.get('definitions'):
                print("\n--- Definitions ---")
                current_pos = None
                for d in word_data['definitions']:
                    pos = d.get('pos_tl') or d.get('original_pos') or 'Uncategorized'
                    if pos != current_pos:
                        print(f"\n[{pos}]")
                        current_pos = pos
                    print(f"- {d.get('definition_text')}")
                    if d.get('examples'): print(f"  Ex: {d.get('examples')}")
                    if d.get('usage_notes'): print(f"  Notes: {d.get('usage_notes')}")

            if word_data.get('relations'):
                print("\n--- Related Words ---")
                current_type = None
                current_dir = None
                for r in word_data['relations']:
                    rel_type = r.get('relation_type', 'Unknown').title()
                    direction = r.get('direction')
                    if rel_type != current_type or direction != current_dir:
                        print(f"\n[{rel_type} ({direction})]")
                        current_type = rel_type
                        current_dir = direction
                    print(f"- {r.get('related_lemma')} (ID: {r.get('related_word_id')})")
            
            # Add simple text output for Etymologies, Pronunciations, Forms
            if word_data.get('etymologies'):
                print("\n--- Etymology ---")
                for e in word_data['etymologies']:
                    print(f"- {e.get('etymology_text')}")
                    if e.get('components'): print(f"  Components: {e.get('components')}")
                    if e.get('language_codes'): print(f"  Languages: {e.get('language_codes')}")

            if word_data.get('pronunciations'):
                 print("\n--- Pronunciations ---")
                 for p in word_data['pronunciations']:
                     print(f"- IPA: {p.get('ipa_transcription')}, Audio: {p.get('audio_url') or 'N/A'}, Notes: {p.get('notes') or 'None'}")

            if word_data.get('word_forms'):
                print("\n--- Word Forms ---")
                for wf in word_data['word_forms']:
                    print(f"- Form: {wf.get('form')} {'(Canonical)' if wf.get('is_canonical') else ''} {'(Primary)' if wf.get('is_primary') else ''}, POS: {wf.get('standardized_pos_id')}, Tags: {wf.get('tags') or 'None'}")

        elif format == 'rich': # Default Rich Output
            # Core Info Panel
            core_panel_content = Text()
            core_panel_content.append(f"Lemma: {word_data.get('lemma')}\n")
            core_panel_content.append(f"Language: {word_data.get('language_code')}\n")
            if word_data.get('root_lemma'):
                 core_panel_content.append(f"Root: {word_data.get('root_lemma')} (ID: {word_data.get('root_word_id')})\n")
            if word_data.get('has_baybayin') and word_data.get('baybayin_form'):
                core_panel_content.append(f"Baybayin: [magenta]{word_data.get('baybayin_form')}[/]\n")
                if word_data.get('romanized_form'): core_panel_content.append(f"Romanized: {word_data.get('romanized_form')}\n")
            # Add other flags
            flags = []
            if word_data.get('is_proper_noun'): flags.append("Proper Noun")
            if word_data.get('is_abbreviation'): flags.append("Abbreviation")
            if word_data.get('is_initialism'): flags.append("Initialism")
            if flags: core_panel_content.append(f"Flags: {', '.join(flags)}\n")
            if word_data.get('tags'): core_panel_content.append(f"Tags: [cyan]{word_data.get('tags')}[/]\n")
            
            console.print(Panel(core_panel_content, title=f"[bold cyan]Word Info - ID: {word_id}[/]"))

            # Definitions Panel
            if word_data.get('definitions'):
                def_panel_content = Text()
                current_pos = None
                for d in word_data['definitions']:
                    pos = d.get('pos_tl') or d.get('original_pos') or 'Uncategorized'
                    if pos != current_pos:
                        def_panel_content.append(f"\n[bold green]{pos}[/]\n")
                        current_pos = pos
                    def_panel_content.append(f"• {d.get('definition_text')}\n")
                    if d.get('examples'): def_panel_content.append(f"  [dim]Ex:[/] {d.get('examples')}\n")
                    if d.get('usage_notes'): def_panel_content.append(f"  [dim]Notes:[/] {d.get('usage_notes')}\n")
                    if d.get('tags'): def_panel_content.append(f"  [dim]Tags:[/] [cyan]{d.get('tags')}[/]\n")
                console.print(Panel(def_panel_content, title="[bold green]Definitions[/]", border_style="green"))
            
             # Word Forms Panel
            if word_data.get('word_forms'):
                forms_table = Table(title="[bold yellow]Word Forms[/]", box=box_ROUNDED, border_style="yellow")
                forms_table.add_column("Form")
                forms_table.add_column("Flags")
                forms_table.add_column("POS ID")
                forms_table.add_column("Tags")
                for wf in word_data['word_forms']:
                    flags = []
                    if wf.get('is_canonical'): flags.append("Canonical")
                    if wf.get('is_primary'): flags.append("Primary")
                    forms_table.add_row(
                        wf.get('form', ''),
                        ', '.join(flags) or '-',
                        str(wf.get('standardized_pos_id') or '-'),
                        wf.get('tags') or '-' 
                    )
                console.print(forms_table)

            # Relations Panel
            if word_data.get('relations'):
                rel_panel_content = Text()
                current_type = None
                current_dir = None
                for r in word_data['relations']:
                    rel_type = r.get('relation_type', 'Unknown').title()
                    direction = r.get('direction')
                    if rel_type != current_type or direction != current_dir:
                         rel_panel_content.append(f"\n[bold magenta]{rel_type} ({direction.capitalize()})[/]\n")
                         current_type = rel_type
                         current_dir = direction
                    rel_panel_content.append(f"• {r.get('related_lemma')} ([dim]ID: {r.get('related_word_id')}[/])\n")
                console.print(Panel(rel_panel_content, title="[bold magenta]Related Words[/]", border_style="magenta"))
                
            # Etymology Panel
            if word_data.get('etymologies'):
                etym_panel_content = Text()
                for e in word_data['etymologies']:
                    etym_panel_content.append(f"• {e.get('etymology_text')}\n")
                    if e.get('components'): etym_panel_content.append(f"  [dim]Components:[/] {e.get('components')}\n")
                    if e.get('language_codes'): etym_panel_content.append(f"  [dim]Languages:[/] {e.get('language_codes')}\n")
                console.print(Panel(etym_panel_content, title="[bold blue]Etymology[/]", border_style="blue"))

            # Pronunciations Panel
            if word_data.get('pronunciations'):
                pron_table = Table(title="[bold yellow]Pronunciations[/]", box=box_ROUNDED, border_style="yellow")
                pron_table.add_column("IPA")
                pron_table.add_column("Audio URL")
                pron_table.add_column("Notes")
                for p in word_data['pronunciations']:
                    pron_table.add_row(
                        p.get('ipa_transcription', '-'), 
                        p.get('audio_url') or '-', 
                        p.get('notes') or '-' 
                    )
                console.print(pron_table)
                
            # Metadata Panels (if present)
            if word_data.get('word_metadata') and word_data['word_metadata'] != {}:
                console.print(Panel(json.dumps(word_data['word_metadata'], indent=2), title="Word Metadata", border_style="dim"))
            
            # Optionally add other tables/panels for credits, etc.

            # Raw JSON dump if debug
            if debug:
                 console.print(Panel(json.dumps(word_data, indent=2, default=str), title="[DEBUG] Raw Word Data (JSON)", border_style="red"))
                 
        # Fallback for unknown format
        else:
            logger.warning(f"Unknown lookup format requested: {format}. Defaulting to rich.")
            # Recursive call with default format (avoid infinite loop potential?)
            if format != 'rich': # Prevent infinite loop if 'rich' is somehow unknown
                 lookup_by_id(cur, word_id, console, 'rich', debug) 

    except psycopg2.Error as db_err:
        logger.error(f"Database error looking up ID {word_id}: {db_err}", exc_info=True)
        if format == 'rich':
            console.print(f"[red]Database Error: {db_err.pgcode} - {db_err}[/]")
        elif format == 'text':
            print(f"Database Error: {db_err}")
        # JSON format handled by lack of data
    except Exception as e:
        logger.error(f"Unexpected error looking up ID {word_id}: {str(e)}", exc_info=True)
        if format == 'rich':
            console.print(f"[red]Error looking up word details: {str(e)}[/]")
        elif format == 'text':
            print(f"Error: {str(e)}")
        # JSON format handled by lack of data

# -------------------------------------------------------------------
# CLI Wrapper Functions
# -------------------------------------------------------------------
def create_argument_parser_cli() -> argparse.ArgumentParser:
    return create_argument_parser()


def migrate_data_cli(args):
    """CLI wrapper for migrating data."""
    logger.info(f"Migrate data called with args: {args}")
    try:
        migrate_data(args)
    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]Migration failed:[/] {str(e)}")
        if "database" in str(e).lower() and "exist" in str(e).lower():
            console.print(
                "\n[bold yellow]The database may not exist.[/] You can create it using:"
            )
            console.print("   [bold]python create_database.py[/]")
            console.print("Then run the migration again.")


def verify_database_cli(args):
    """CLI wrapper for verifying database."""
    logger.info(f"Verify database called with args: {args}")
    # Example: Accessing new args
    # checks_to_run = args.checks.split(',') if args.checks else ['all']
    # repair_tasks_to_run = args.repair_tasks.split(',') if args.repair_tasks else ['all']
    # logger.info(f"Running checks: {checks_to_run}, Repairing: {args.repair}, Repair tasks: {repair_tasks_to_run}")
    verify_database(args)


def purge_database_cli(args):
    """CLI wrapper for purging database."""
    logger.info(f"Purge database called with args: {args}")
    # Example: Accessing new args
    # tables_to_purge = args.tables.split(',') if args.tables else ['all']
    # logger.info(f"Force purge: {args.force}, Tables to purge: {tables_to_purge}")
    try:
        # Original logic, needs modification to use args.tables and args.force
        console = Console()
        if not args.force:
            if args.tables:
                confirm = input(f"Are you sure you want to purge data from tables: {args.tables}? (yes/no): ")
            else:
                confirm = input("Are you sure you want to purge ALL dictionary data? This cannot be undone. (yes/no): ")
            if confirm.lower() != 'yes':
                console.print("[yellow]Purge operation cancelled by user.[/]")
                return

        conn = get_connection()
        with conn.cursor() as cur:
            console.print("[bold blue]Starting dictionary purge process...[/]")
            
            tables_to_actually_purge = []
            if args.tables:
                tables_to_actually_purge = [t.strip() for t in args.tables.split(',')]
                logger.info(f"Purging specified tables: {tables_to_actually_purge}")
            else:
                # Default list of tables to purge if --tables is not specified
                tables_to_actually_purge = [
                    "definition_relations", "affixations", "relations", "etymologies",
                    "definition_examples", "definition_links", "definition_categories",
                    "word_forms", "word_templates", "pronunciations", "credits",
                    "definitions", "words", "parts_of_speech", "languages"
                ] # Added more tables
                logger.info("Purging all default dictionary tables.")

            for table in tables_to_actually_purge:
                try:
                    console.print(f"Purging {table}...")
                    cur.execute(f"DELETE FROM public.\"{table}\"") # Ensure public schema and quote table names
                    logger.info(f"Successfully purged {table}.")
                except psycopg2.Error as e:
                    logger.error(f"Error purging table {table}: {e}")
                    console.print(f"[red]Error purging table {table}: {e.pgcode}. Skipping.[/]")
                    conn.rollback() # Rollback changes for this table and continue
                except Exception as e_gen:
                    logger.error(f"Unexpected error purging table {table}: {e_gen}")
                    console.print(f"[red]Unexpected error purging table {table}. Skipping.[/]")

            conn.commit()
            console.print("[bold green]Dictionary purge completed.[/]")

    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed during purge: {e}")
        console.print(f"[bold red]Database connection error: {e}[/]")
    except Exception as e:
        logger.error(f"Error during purge: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error during purge: {str(e)}[/]")
    finally:
        if 'conn' in locals() and conn and not conn.closed:
            conn.close()


def lookup_word_cli(args):
    """CLI wrapper for looking up a word."""
    logger.info(f"Lookup word called with args: {args}")
    # Example: Accessing new args
    # term_is_id = args.id
    # language_filter = args.lang
    # logger.info(f"Looking up: {args.term}, Is ID: {term_is_id}, Language: {language_filter}")
    lookup_word(args)


def display_dictionary_stats_cli(args):
    """CLI wrapper for displaying dictionary stats."""
    logger.info(f"Display dictionary stats called with args: {args}")
    # Example: Accessing new args
    # specific_table = args.table
    # export_file = args.export
    # logger.info(f"Detailed: {args.detailed}, Table: {specific_table}, Export to: {export_file}")
    display_dictionary_stats(args) # Pass args to the main function


def display_leaderboard_cli(args):
    """CLI wrapper for displaying leaderboard."""
    logger.info(f"Display leaderboard called with args: {args}")
    console = Console()
    conn = None
    try:
        conn = get_connection()
        # Use DictCursor for the wrapped function
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur: 
            display_leaderboard(cur, console, args) # Pass the whole args object
    except psycopg2.OperationalError as db_err:
        logger.error(f"Database connection failed for leaderboard: {db_err}", exc_info=True)
        console.print(f"[bold red]Database connection error:[/]\n[dim]{db_err}[/]")
    except Exception as e:
        logger.error(f"Error in display_leaderboard_cli: {e}", exc_info=True)
        console.print(f"[red]Error displaying leaderboard: {str(e)}[/]")
    finally:
        if conn and not conn.closed: # Check if connection exists and is open
            conn.close()


def explore_dictionary_cli(args):
    """CLI wrapper for interactive explorer."""
    logger.info(f"Explore dictionary called with args: {args}")
    explore_dictionary() # This function doesn't take args currently


def display_help_cli(args):
    """CLI wrapper for displaying help."""
    logger.info(f"Display help called with args: {args}")
    display_help(args)

# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
# ... (main function needs to be updated to map to new CLI handlers if their names changed)
# ... and to map new commands like `update` and `migrate-relationships`

def main():
    parser = create_argument_parser_cli()
    args = parser.parse_args()

    # Map commands to their handler functions
    command_handlers = {
        "migrate": migrate_data_cli,
        "verify": verify_database_cli,
        "purge": purge_database_cli,
        "lookup": lookup_word_cli,
        "stats": display_dictionary_stats_cli,
        "leaderboard": display_leaderboard_cli,
        "help": display_help_cli, # display_help_cli is correct
        "explore": explore_dictionary_cli,
    }

    if args.command in command_handlers:
        command_handlers[args.command](args)
    else:
        # This case should ideally not be reached if subparsers are `required=True`
        # but as a fallback, or if `required` is removed.
        logger.warning(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    setup_logging() # Activate logging configuration
    main()
