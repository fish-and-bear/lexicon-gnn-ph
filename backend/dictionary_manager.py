#!/usr/bin/env python3
import argparse
import json
import logging
import os
import io
import traceback # Keep traceback for error logging in lookup_word
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

# Database connection and core helpers
from backend.dictionary_manager.db_helpers import (
    get_connection,
    get_cursor,
    with_transaction,
    create_or_update_tables,
    deduplicate_definitions,
    cleanup_relations,
    cleanup_dictionary_data,
    check_pg_version,
    check_extensions,
    check_data_access,
    check_tables_exist,
    check_query_performance,
    purge_database_tables,
    repair_database_issues,
    check_baybayin_consistency,
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
    subparsers = parser.add_subparsers(dest="command")
    migrate_parser = subparsers.add_parser(
        "migrate", help="Create/update schema and load data"
    )
    migrate_parser.add_argument(
        "--check-exists", action="store_true", help="Skip identical existing entries"
    )
    migrate_parser.add_argument(
        "--force", action="store_true", help="Force migration without confirmation"
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
    verify_parser = subparsers.add_parser("verify", help="Verify data integrity")
    verify_parser.add_argument(
        "--quick", action="store_true", help="Run quick verification"
    )
    verify_parser.add_argument(
        "--repair", action="store_true", help="Attempt to repair issues"
    )
    update_parser = subparsers.add_parser("update", help="Update DB with new data")
    update_parser.add_argument(
        "--file", type=str, required=True, help="JSON or JSONL file to use"
    )
    update_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    lookup_parser = subparsers.add_parser("lookup", help="Look up word information")
    lookup_parser.add_argument("word", help="Word to look up")
    lookup_parser.add_argument(
        "--debug", action="store_true", help="Show debug information"
    )
    lookup_parser.add_argument(
        "--format",
        choices=["text", "json", "rich"],
        default="rich",
        help="Output format",
    )
    stats_parser = subparsers.add_parser("stats", help="Display dictionary statistics")
    stats_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed statistics"
    )
    stats_parser.add_argument("--export", type=str, help="Export statistics to file")
    subparsers.add_parser("leaderboard", help="Display top contributors")
    subparsers.add_parser("help", help="Display help information")
    subparsers.add_parser("test", help="Run database connectivity tests")
    subparsers.add_parser("explore", help="Interactive dictionary explorer")
    purge_parser = subparsers.add_parser("purge", help="Safely delete all data")
    purge_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )
    subparsers.add_parser(
        "cleanup",
        help="Clean up dictionary data by removing duplicates and standardizing formats",
    )
    subparsers.add_parser(
        "migrate-relationships",
        help="Migrate existing relationships to the new RelationshipType system",
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
        console = Console()
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
        console.print("[bold]Setting up database schema...[/]")
        create_or_update_tables(conn)  # This function handles its own commit/rollback

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
    console = Console()
    console.print("\n[bold cyan]📖 Dictionary Manager CLI Help[/]", justify="center")
    console.print(
        "[dim]A comprehensive tool for managing Filipino dictionary data[/]\n",
        justify="center",
    )
    usage_panel = Panel(
        Text.from_markup("python dictionary_manager.py [command] [options]"),
        title="Basic Usage",
        border_style="blue",
    )
    console.print(usage_panel)
    console.print()
    commands = [
        {
            "name": "migrate",
            "description": "Create/update schema and load data from sources",
            "options": [
                ("--check-exists", "Skip identical existing entries"),
                ("--force", "Skip confirmation prompt"),
            ],
            "example": "python dictionary_manager.py migrate --check-exists",
            "icon": "🔄",
        },
        {
            "name": "lookup",
            "description": "Look up comprehensive information about a word",
            "options": [
                ("word", "The word to look up"),
                ("--format", "Output format (text/json/rich)"),
            ],
            "example": "python dictionary_manager.py lookup kamandag",
            "icon": "🔍",
        },
        {
            "name": "stats",
            "description": "Display comprehensive dictionary statistics",
            "options": [
                ("--detailed", "Show detailed statistics"),
                ("--export", "Export statistics to file"),
            ],
            "example": "python dictionary_manager.py stats --detailed",
            "icon": "📊",
        },
        {
            "name": "verify",
            "description": "Verify data integrity",
            "options": [
                ("--quick", "Run quick verification"),
                ("--repair", "Attempt to repair issues"),
            ],
            "example": "python dictionary_manager.py verify --repair",
            "icon": "✅",
        },
        {
            "name": "purge",
            "description": "Safely delete all data from the database",
            "options": [("--force", "Skip confirmation prompt")],
            "example": "python dictionary_manager.py purge --force",
            "icon": "🗑️",
        },
    ]
    data_commands = Table(
        title="Data Management Commands", box=box_ROUNDED, border_style="cyan"
    )
    data_commands.add_column("Command", style="bold yellow")
    data_commands.add_column("Description", style="white")
    data_commands.add_column("Options", style="cyan")
    data_commands.add_column("Example", style="green")
    query_commands = Table(
        title="Query Commands", box=box_ROUNDED, border_style="magenta"
    )
    query_commands.add_column("Command", style="bold yellow")
    query_commands.add_column("Description", style="white")
    query_commands.add_column("Options", style="cyan")
    query_commands.add_column("Example", style="green")
    for cmd in commands:
        options_text = (
            "\n".join([f"[cyan]{opt[0]}[/]: {opt[1]}" for opt in cmd["options"]]) or "-"
        )
        row = [
            f"{cmd['icon']} {cmd['name']}",
            cmd["description"],
            options_text,
            f"[dim]{cmd['example']}[/]",
        ]
        if cmd["name"] in ["migrate", "update", "purge"]:
            data_commands.add_row(*row)
        else:
            query_commands.add_row(*row)
    console.print(data_commands)
    console.print()
    console.print(query_commands)
    console.print()
    console.print(
        "\n[dim]For more detailed information, visit the documentation.[/]",
        justify="center",
    )
    console.print()


def lookup_word(args):
    """Look up a word and display its information."""
    word = args.word
    logger.info(f"Starting lookup for word: '{word}'")
    
    try:
        # First attempt with standard connection method
        conn = None
        try:
            logger.info("Attempting to use standard connection method...")
            conn = get_connection()
            cur = conn.cursor()
        except Exception as conn_err:
            logger.warning(f"Standard connection failed: {conn_err}. Trying direct psycopg2 connection.")
            # If standard connection fails, try direct psycopg2 connection using postgres defaults
            import psycopg2
            db_uri = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/fil_dict_db')
            logger.info(f"Connecting directly using: {db_uri.split('@')[0].split(':')[0]}:***@{db_uri.split('@')[1] if '@' in db_uri else db_uri}")
            conn = psycopg2.connect(db_uri)
            cur = conn.cursor()
            logger.info("Direct connection successful")
            
        with conn:
            with cur:
                normalized_word = normalize_lemma(word)
                logger.info(f"pg_trgm extension installed: {check_pg_trgm_installed(cur)}")
                
                # Try exact match first
                logger.info(f"Executing exact match query for '{word}'")
                cur.execute(
                    """
                    SELECT id, lemma, language_code, root_word_id
                    FROM words 
                    WHERE normalized_lemma = %s
                    """,
                    (normalized_word,),
                )
                results = cur.fetchall()
                logger.info(f"Found {len(results)} exact matches")
                
                # If no exact matches, try fuzzy search
                if not results:
                    logger.info(f"No exact matches, trying fuzzy search for '{word}'")
                    try:
                        # Only use similarity if pg_trgm extension is installed
                        if check_pg_trgm_installed(cur):
                            cur.execute(
                                """
                                SELECT id, lemma, language_code, root_word_id, 
                                       similarity(normalized_lemma, %s) as sim
                                FROM words 
                                WHERE similarity(normalized_lemma, %s) > 0.4
                                ORDER BY sim DESC
                                LIMIT 10
                                """,
                                (normalized_word, normalized_word),
                            )
                            results = cur.fetchall()
                            logger.info(f"Found {len(results)} fuzzy matches")
                        else:
                            logger.warning("pg_trgm extension not installed, skipping fuzzy search")
                    except Exception as e:
                        logger.warning(f"Error in fuzzy search: {e}")
                
                # If still no matches, try a simpler ILIKE search
                if not results:
                    logger.info(f"No fuzzy matches, falling back to ILIKE search")
                    logger.info(f"Trying ILIKE search for '{word}'")
                    cur.execute(
                        """
                        SELECT id, lemma, language_code, root_word_id 
                        FROM words 
                        WHERE lemma ILIKE %s OR normalized_lemma ILIKE %s
                        """,
                        (f"%{word}%", f"%{normalized_word}%"),
                    )
                    results = cur.fetchall()
                    logger.info(f"ILIKE search found {len(results)} matches")
                    
                # Run diagnostics if no results found
                if not results:
                    logger.info("Diagnostic: checking if word exists in any form")
                    cur.execute(
                        """
                        SELECT EXISTS(
                            SELECT 1 FROM words 
                            WHERE lemma = %s OR normalized_lemma = %s
                        )
                        """,
                        (word, normalized_word),
                    )
                    exists = cur.fetchone()[0]
                    logger.info(f"Diagnostic result: Word '{word}' exists (exact match): {exists}")
                    
                    # Check for partial matches as well
                    cur.execute(
                        """
                        SELECT EXISTS(
                            SELECT 1 FROM words 
                            WHERE lemma LIKE %s OR normalized_lemma LIKE %s
                        )
                        """,
                        (f"%{word}%", f"%{normalized_word}%"),
                    )
                    partial_exists = cur.fetchone()[0]
                    logger.info(f"Diagnostic result: Word '{word}' exists (partial match): {partial_exists}")
                    
                    # Verify the database schema
                    logger.info("Diagnostic: verifying database schema")
                    cur.execute(
                        """
                        SELECT column_name, data_type 
                        FROM information_schema.columns
                        WHERE table_name = 'words'
                        ORDER BY ordinal_position
                        """
                    )
                    columns = cur.fetchall()
                    logger.info(f"Diagnostic: words table has {len(columns)} columns")
                    for column_name, data_type in columns:
                        logger.info(f"  Column: {column_name}, Type: {data_type}")
                    
                    console = Console()
                    console.print(f"\nNo entries found for '{word}'")
                    return

                console = Console()
                if len(results) == 1:
                    word_id = results[0][0]
                    return lookup_by_id(cur, word_id, console)

                # Multiple results
                console.print("\n[bold]Multiple matches found:[/]")
                table = Table(show_header=True, header_style="bold")
                table.add_column("ID")
                table.add_column("Word")
                table.add_column("Language")
                table.add_column("Root")

                for word_id, lemma, lang_code, root_id in results:
                    # Get root word if available
                    root_word = None
                    if root_id:
                        cur.execute(
                            "SELECT lemma FROM words WHERE id = %s", (root_id,)
                        )
                        root_row = cur.fetchone()
                        if root_row:
                            root_word = root_row[0]

                    table.add_row(
                        str(word_id),
                        lemma,
                        lang_code or "unknown",
                        root_word or "N/A",
                    )

                console.print(table)
                choice = input("\nEnter ID to view details (or press Enter to exit): ")
                if choice.strip():
                    try:
                        word_id = int(choice.strip())
                        return lookup_by_id(cur, word_id, console)
                    except ValueError:
                        console.print("[red]Invalid ID. Please enter a number.[/]")
                    except Exception as e:
                        console.print(f"[red]Error looking up word: {str(e)}[/]")

    except Exception as e:
        logger.error(f"Error during word lookup: {str(e)}", exc_info=True)
        console = Console()
        console.print(f"[red]Error looking up word: {str(e)}[/]")
    finally:
        if 'conn' in locals() and conn is not None:
            try:
                conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


def display_dictionary_stats_cli(args):
    """Display dictionary statistics from the command line."""
    try:
        # Get a proper database connection and cursor
        conn = get_connection()
        with conn.cursor() as cur:
            display_dictionary_stats(cur)

        # Make sure to close the connection when done
        if conn:
            conn.close()
    except Exception as e:
        console = Console()
        console.print(f"[red]Error displaying dictionary stats: {str(e)}[/]")


@with_transaction(commit=False)
def display_dictionary_stats(cur):
    """Display comprehensive dictionary statistics."""
    console = Console()
    try:
        # Overall Statistics
        overall_table = Table(title="[bold blue]Overall Statistics[/]", box=box_ROUNDED)
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Count", justify="right", style="green")
        overall_table.add_column("Details", style="dim")

        # Check which columns exist in the words table
        cur.execute(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'words'
        """
        )
        available_columns = {row[0] for row in cur.fetchall()}

        # Basic counts with details - build dynamically based on available columns
        basic_queries = {
            "Total Words": ("SELECT COUNT(*) FROM words", None),
            "Total Definitions": ("SELECT COUNT(*) FROM definitions", None),
            "Total Relations": ("SELECT COUNT(*) FROM relations", None),
            "Total Etymologies": ("SELECT COUNT(*) FROM etymologies", None),
            "Total Pronunciations": ("SELECT COUNT(*) FROM pronunciations", None),
            "Total Credits": ("SELECT COUNT(*) FROM credits", None),
            "Words with Baybayin": (
                "SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE",
                None,
            ),
            "Words with Examples": (
                """
                SELECT COUNT(DISTINCT word_id) 
                FROM definitions 
                WHERE examples IS NOT NULL
            """,
                None,
            ),
            "Words with Etymology": (
                """
                SELECT COUNT(DISTINCT word_id) 
                FROM etymologies
            """,
                None,
            ),
            "Words with Pronunciation": (
                """
                SELECT COUNT(DISTINCT word_id) 
                FROM pronunciations
            """,
                None,
            ),
        }

        # Add optional columns if they exist
        if "is_proper_noun" in available_columns:
            basic_queries["Proper Nouns"] = (
                "SELECT COUNT(*) FROM words WHERE is_proper_noun = TRUE",
                None,
            )
        if "is_abbreviation" in available_columns:
            basic_queries["Abbreviations"] = (
                "SELECT COUNT(*) FROM words WHERE is_abbreviation = TRUE",
                None,
            )
        if "is_initialism" in available_columns:
            basic_queries["Initialisms"] = (
                "SELECT COUNT(*) FROM words WHERE is_initialism = TRUE",
                None,
            )
        if "root_word_id" in available_columns:
            basic_queries["Words with Root"] = (
                "SELECT COUNT(*) FROM words WHERE root_word_id IS NOT NULL",
                None,
            )

        for label, (query, detail_query) in basic_queries.items():
            try:
                cur.execute(query)
                count = cur.fetchone()[0]
                details = ""
                if detail_query:
                    cur.execute(detail_query)
                    details = cur.fetchone()[0]
                overall_table.add_row(label, f"{count:,}", details)
            except Exception as e:
                logger.warning(f"Error getting stats for {label}: {e}")
                overall_table.add_row(label, "N/A", f"Error: {str(e)}")

        # Language Statistics with more details
        try:
            cur.execute(
                """
                SELECT 
                    w.language_code,
                    COUNT(*) as word_count,
                    COUNT(DISTINCT d.id) as def_count,
                    COUNT(DISTINCT e.id) as etym_count,
                    COUNT(DISTINCT p.id) as pron_count,
                    COUNT(DISTINCT CASE WHEN w.has_baybayin THEN w.id END) as baybayin_count
                FROM words w
                LEFT JOIN definitions d ON w.id = d.word_id
                LEFT JOIN etymologies e ON w.id = e.word_id
                LEFT JOIN pronunciations p ON w.id = p.word_id
                GROUP BY w.language_code
                ORDER BY word_count DESC
            """
            )

            lang_table = Table(title="[bold blue]Words by Language[/]", box=box_ROUNDED)
            lang_table.add_column("Language", style="yellow")
            lang_table.add_column("Words", justify="right", style="green")
            lang_table.add_column("Definitions", justify="right", style="green")
            lang_table.add_column("Etymologies", justify="right", style="green")
            lang_table.add_column("Pronunciations", justify="right", style="green")
            lang_table.add_column("Baybayin", justify="right", style="green")

            total_words = 0
            results = cur.fetchall()
            for row in results:
                total_words += row[1]

            for lang_code, words, defs, etyms, prons, bayb in results:
                percentage = (words / total_words) * 100 if total_words > 0 else 0
                lang_table.add_row(
                    lang_code,
                    f"{words:,} ({percentage:.1f}%)",
                    f"{defs:,}",
                    f"{etyms:,}",
                    f"{prons:,}",
                    f"{bayb:,}",
                )

            console.print("\n[bold]Dictionary Statistics[/]")
            console.print(overall_table)
            console.print()
            console.print(lang_table)
        except Exception as e:
            logger.error(f"Error displaying language statistics: {str(e)}")
            console.print(f"[red]Error displaying language statistics: {str(e)}[/]")
            console.print(overall_table)

        # Parts of Speech Statistics with examples
        try:
            # Check if standardized_pos_id exists
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'definitions' AND column_name = 'standardized_pos_id'
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
                        COUNT(CASE WHEN d.examples IS NOT NULL THEN 1 END) as with_examples
                    FROM definitions d
                    JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                    GROUP BY p.name_tl
                    ORDER BY count DESC
                """
                )
            else:
                # Fallback to using part_of_speech text field
                cur.execute(
                    """
                    SELECT 
                        COALESCE(part_of_speech, 'Unknown'),
                        COUNT(*) as count,
                        COUNT(DISTINCT word_id) as unique_words,
                        COUNT(CASE WHEN examples IS NOT NULL THEN 1 END) as with_examples
                    FROM definitions
                    GROUP BY part_of_speech
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
                        f"{with_examples:,}",
                    )

                console.print()
                console.print(pos_table)
        except Exception as e:
            logger.error(f"Error displaying part of speech statistics: {str(e)}")
            console.print(
                f"[red]Error displaying part of speech statistics: {str(e)}[/]"
            )

        # Relationship Statistics by category
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
        except Exception as e:
            logger.error(f"Error displaying relationship statistics: {str(e)}")
            console.print(f"[red]Error displaying relationship statistics: {str(e)}[/]")

        # Source Statistics with more details
        try:
            # First check if source_info column exists
            if "source_info" in available_columns:
                # Get source statistics from source_info
                cur.execute(
                    """
                    SELECT 
                        COALESCE(source_info, 'Unknown') as source_name,
                        COUNT(*) as word_count
                    FROM words
                    GROUP BY source_name
                    ORDER BY word_count DESC
                """
                )

                source_results = cur.fetchall()
                if source_results:
                    source_table = Table(
                        title="[bold blue]Source Distribution[/]", box=box_ROUNDED
                    )
                    source_table.add_column("Source", style="yellow")
                    source_table.add_column("Words", justify="right", style="green")

                    for source, count in source_results:
                        source_table.add_row(source or "Unknown", f"{count:,}")

                    console.print()
                    console.print(source_table)

            # Also check definitions sources
            cur.execute(
                """
                SELECT 
                    COALESCE(sources, 'Unknown') as source_name,
                    COUNT(*) as def_count,
                    COUNT(DISTINCT word_id) as word_count,
                    COUNT(CASE WHEN examples IS NOT NULL THEN 1 END) as example_count
                FROM definitions
                GROUP BY sources
                ORDER BY def_count DESC
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
                def_source_table.add_column("Words", justify="right", style="green")
                def_source_table.add_column(
                    "With Examples", justify="right", style="green"
                )

                for source, def_count, word_count, example_count in def_source_results:
                    def_source_table.add_row(
                        source or "Unknown",
                        f"{def_count:,}",
                        f"{word_count:,}",
                        f"{example_count:,}",
                    )

                console.print()
                console.print(def_source_table)
        except Exception as e:
            logger.error(f"Error displaying source statistics: {str(e)}")
            console.print(f"[yellow]Could not generate source statistics: {str(e)}[/]")

        # Baybayin Statistics with details
        try:
            baybayin_table = Table(
                title="[bold blue]Baybayin Statistics[/]", box=box_ROUNDED
            )
            baybayin_table.add_column("Metric", style="yellow")
            baybayin_table.add_column("Count", justify="right", style="green")
            baybayin_table.add_column("Details", style="dim")

            baybayin_queries = {
                "Total Baybayin Forms": (
                    "SELECT COUNT(*) FROM words WHERE baybayin_form IS NOT NULL",
                    """SELECT COUNT(DISTINCT language_code) 
                    FROM words WHERE baybayin_form IS NOT NULL""",
                ),
                "With Romanization": (
                    "SELECT COUNT(*) FROM words WHERE romanized_form IS NOT NULL",
                    None,
                ),
                "Verified Forms": (
                    """SELECT COUNT(*) FROM words 
                    WHERE has_baybayin = TRUE 
                    AND baybayin_form IS NOT NULL""",
                    None,
                ),
            }

            # Only add Badlit stats if the column exists
            if "badlit_form" in available_columns:
                baybayin_queries["With Badlit"] = (
                    "SELECT COUNT(*) FROM words WHERE badlit_form IS NOT NULL",
                    None,
                )
                baybayin_queries["Complete Forms"] = (
                    """SELECT COUNT(*) FROM words 
                    WHERE baybayin_form IS NOT NULL 
                    AND romanized_form IS NOT NULL 
                    AND badlit_form IS NOT NULL""",
                    None,
                )

            for label, (query, detail_query) in baybayin_queries.items():
                try:
                    cur.execute(query)
                    count = cur.fetchone()[0]
                    details = ""
                    if detail_query:
                        cur.execute(detail_query)
                        details = f"across {cur.fetchone()[0]} languages"
                    baybayin_table.add_row(label, f"{count:,}", details)
                except Exception as e:
                    logger.warning(f"Error getting Baybayin stats for {label}: {e}")
                    baybayin_table.add_row(label, "N/A", f"Error: {str(e)}")

            console.print()
            console.print(baybayin_table)
        except Exception as e:
            logger.error(f"Error displaying Baybayin statistics: {str(e)}")
            console.print(
                f"[yellow]Could not generate Baybayin statistics: {str(e)}[/]"
            )

        # Print timestamp
        console.print(
            f"\n[dim]Statistics generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]"
        )

    except Exception as e:
        logger.error(f"Error displaying dictionary stats: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/]")


def display_leaderboard(cur, console):
    """Displays various contribution leaderboards."""
    # Get a fresh connection and cursor instead of using the one passed in
    # This ensures we start with a clean transaction state
    conn = None
    try:
        from contextlib import closing
        import psycopg2
        from psycopg2.extras import DictCursor

        # Get a fresh connection from the pool
        conn = get_connection()

        # Use a context manager to ensure proper closing
        with closing(conn.cursor(cursor_factory=DictCursor)) as fresh_cur:
            console.print(
                "\n[bold magenta underline]📊 Dictionary Contributors Leaderboard[/]\n"
            )

            overall_stats_table = Table(
                title="[bold blue]Overall Statistics[/]",
                box=box_ROUNDED,
                show_header=False,
            )
            overall_stats_table.add_column("Statistic", style="cyan")
            overall_stats_table.add_column("Value", justify="right", style="green")

            try:
                # Overall stats
                fresh_cur.execute("SELECT COUNT(*) FROM words")
                total_words = fresh_cur.fetchone()[0]
                overall_stats_table.add_row("Total Words", f"{total_words:,}")

                fresh_cur.execute("SELECT COUNT(*) FROM definitions")
                total_definitions = fresh_cur.fetchone()[0]
                overall_stats_table.add_row(
                    "Total Definitions", f"{total_definitions:,}"
                )

                fresh_cur.execute("SELECT COUNT(*) FROM relations")
                total_relations = fresh_cur.fetchone()[0]
                overall_stats_table.add_row("Total Relations", f"{total_relations:,}")

                fresh_cur.execute("SELECT COUNT(*) FROM etymologies")
                total_etymologies = fresh_cur.fetchone()[0]
                overall_stats_table.add_row(
                    "Total Etymologies", f"{total_etymologies:,}"
                )

                fresh_cur.execute(
                    "SELECT COUNT(DISTINCT standardized_pos_id) FROM definitions WHERE standardized_pos_id IS NOT NULL"
                )
                total_pos = fresh_cur.fetchone()[0]
                overall_stats_table.add_row("Unique Parts of Speech", str(total_pos))

                fresh_cur.execute(
                    "SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE OR baybayin_form IS NOT NULL"
                )
                words_with_baybayin = fresh_cur.fetchone()[0]
                overall_stats_table.add_row(
                    "Words w/ Baybayin", f"{words_with_baybayin:,}"
                )

                console.print(overall_stats_table)
                console.print()

            except Exception as e:
                logger.error(f"Error generating overall statistics: {str(e)}")
                console.print(
                    f"[yellow]Could not generate overall statistics: {str(e)}[/]"
                )
                # Roll back to clean state
                conn.rollback()

            # Definition Contributors
            try:
                fresh_cur.execute(
                    """
                    WITH source_stats AS (
                        SELECT
                            CASE
                                WHEN sources ILIKE '%project marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%kaikki-ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kaikki%' AND sources ILIKE '%ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kwf%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%kwf_dictionary%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%tagalog.com%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%root_words%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%diksiyonaryo.ph%' THEN 'diksiyonaryo.ph'
                                WHEN sources ILIKE '%tagalog-words%' THEN 'diksiyonaryo.ph'
                                ELSE COALESCE(sources, 'Unknown')
                            END AS source_name,
                            COUNT(*) AS def_count,
                            COUNT(DISTINCT word_id) AS unique_words,
                            COUNT(CASE WHEN examples IS NOT NULL AND examples != '' THEN 1 END) AS with_examples,
                            COUNT(DISTINCT standardized_pos_id) AS pos_count,
                            COUNT(CASE WHEN usage_notes IS NOT NULL AND usage_notes != '' THEN 1 END) AS with_notes
                        FROM definitions
                        GROUP BY source_name
                    )
                    SELECT
                        source_name,
                        def_count,
                        unique_words,
                        with_examples,
                        pos_count,
                        with_notes,
                        ROUND(100.0 * with_examples / NULLIF(def_count, 0), 1) as example_percentage,
                        ROUND(100.0 * with_notes / NULLIF(def_count, 0), 1) as notes_percentage
                    FROM source_stats
                    ORDER BY def_count DESC
                    """
                )

                def_results = fresh_cur.fetchall()
                if def_results:
                    def_table = Table(
                        title="[bold blue]Definition Contributors[/]", box=box_ROUNDED
                    )
                    def_table.add_column("Source", style="yellow")
                    def_table.add_column("Definitions", justify="right", style="green")
                    def_table.add_column("Words", justify="right", style="green")
                    def_table.add_column("Examples", justify="right", style="cyan")
                    def_table.add_column("POS Types", justify="right", style="cyan")
                    def_table.add_column("Notes", justify="right", style="cyan")
                    def_table.add_column("Coverage", style="dim")

                    for row in def_results:
                        source = (
                            row["source_name"] if "source_name" in row else "Unknown"
                        )
                        defs = row["def_count"] if "def_count" in row else 0
                        words = row["unique_words"] if "unique_words" in row else 0
                        examples = row["with_examples"] if "with_examples" in row else 0
                        pos = row["pos_count"] if "pos_count" in row else 0
                        notes = row["with_notes"] if "with_notes" in row else 0
                        ex_pct = (
                            row["example_percentage"]
                            if "example_percentage" in row
                            else 0.0
                        )
                        notes_pct = (
                            row["notes_percentage"]
                            if "notes_percentage" in row
                            else 0.0
                        )

                        coverage = (
                            f"Examples: {ex_pct or 0.0}%, Notes: {notes_pct or 0.0}%"
                        )
                        def_table.add_row(
                            source,
                            f"{defs:,}",
                            f"{words:,}",
                            f"{examples:,}",
                            str(pos),
                            f"{notes:,}",
                            coverage,
                        )

                    console.print(def_table)
                    console.print()
            except Exception as e:
                logger.error(f"Error generating definition statistics: {str(e)}")
                console.print(
                    f"[red]Error:[/][yellow] Could not generate definition statistics: {str(e)}[/]"
                )
                # Roll back to clean state
                conn.rollback()

            # Etymology Contributors
            try:
                fresh_cur.execute(
                    """
                    WITH etym_stats AS (
                        SELECT
                            CASE
                                WHEN sources ILIKE '%project marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%kaikki-ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kaikki%' AND sources ILIKE '%ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kwf%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%kwf_dictionary%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%tagalog.com%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%root_words%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%diksiyonaryo.ph%' THEN 'diksiyonaryo.ph'
                                WHEN sources ILIKE '%tagalog-words%' THEN 'diksiyonaryo.ph'
                                ELSE COALESCE(sources, 'Unknown')
                            END AS source_name,
                            COUNT(*) AS etym_count,
                            COUNT(DISTINCT word_id) AS unique_words,
                            COUNT(CASE WHEN normalized_components IS NOT NULL THEN 1 END) AS with_components,
                            COUNT(CASE WHEN language_codes IS NOT NULL THEN 1 END) AS with_lang_codes
                        FROM etymologies
                        GROUP BY source_name
                    )
                    SELECT *,
                        ROUND(100.0 * with_components / NULLIF(etym_count, 0), 1) as comp_percentage,
                        ROUND(100.0 * with_lang_codes / NULLIF(etym_count, 0), 1) as lang_percentage
                    FROM etym_stats
                    ORDER BY etym_count DESC
                    """
                )

                etym_results = fresh_cur.fetchall()
                if etym_results:
                    etym_table = Table(
                        title="[bold blue]Etymology Contributors[/]", box=box_ROUNDED
                    )
                    etym_table.add_column("Source", style="yellow")
                    etym_table.add_column("Etymologies", justify="right", style="green")
                    etym_table.add_column("Words", justify="right", style="green")
                    etym_table.add_column("Components", justify="right", style="cyan")
                    etym_table.add_column("Lang Codes", justify="right", style="cyan")
                    etym_table.add_column("Coverage", style="dim")

                    for row in etym_results:
                        source = (
                            row["source_name"] if "source_name" in row else "Unknown"
                        )
                        count = row["etym_count"] if "etym_count" in row else 0
                        words = row["unique_words"] if "unique_words" in row else 0
                        comps = (
                            row["with_components"] if "with_components" in row else 0
                        )
                        langs = (
                            row["with_lang_codes"] if "with_lang_codes" in row else 0
                        )
                        comp_pct = (
                            row["comp_percentage"] if "comp_percentage" in row else 0.0
                        )
                        lang_pct = (
                            row["lang_percentage"] if "lang_percentage" in row else 0.0
                        )

                        coverage = f"Components: {comp_pct or 0.0}%, Languages: {lang_pct or 0.0}%"
                        etym_table.add_row(
                            source,
                            f"{count:,}",
                            f"{words:,}",
                            f"{comps:,}",
                            f"{langs:,}",
                            coverage,
                        )

                    console.print(etym_table)
                    console.print()
            except Exception as e:
                logger.error(f"Error generating etymology statistics: {str(e)}")
                console.print(
                    f"[red]Error:[/][yellow] Could not generate etymology statistics: {str(e)}[/]"
                )
                # Roll back to clean state
                conn.rollback()

            # Relationship Contributors
            try:
                fresh_cur.execute(
                    """
                    WITH rel_stats AS (
                        SELECT
                            CASE
                                WHEN sources ILIKE '%project marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%kaikki-ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kaikki%' AND sources ILIKE '%ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kwf%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%kwf_dictionary%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%tagalog.com%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%root_words%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%diksiyonaryo.ph%' THEN 'diksiyonaryo.ph'
                                WHEN sources ILIKE '%tagalog-words%' THEN 'diksiyonaryo.ph'
                                ELSE COALESCE(sources, 'Unknown')
                            END AS source_name,
                            relation_type,
                            COUNT(*) AS rel_count
                        FROM relations
                        GROUP BY source_name, relation_type
                    )
                    SELECT source_name, relation_type, rel_count
                    FROM rel_stats
                    ORDER BY source_name, rel_count DESC
                    """
                )

                rel_results = fresh_cur.fetchall()
                if rel_results:
                    rel_table = Table(
                        title="[bold blue]Relationship Contributors[/]", box=box_ROUNDED
                    )
                    rel_table.add_column("Source", style="yellow")
                    rel_table.add_column("Relation Type", style="magenta")
                    rel_table.add_column("Count", justify="right", style="green")

                    current_source = None
                    for row in rel_results:
                        source = (
                            row["source_name"] if "source_name" in row else "Unknown"
                        )
                        rel_type = (
                            row["relation_type"]
                            if "relation_type" in row
                            else "Unknown"
                        )
                        count = row["rel_count"] if "rel_count" in row else 0

                        if source != current_source:
                            if current_source is not None:
                                rel_table.add_row("---", "---", "---")  # Separator
                            rel_table.add_row(source, rel_type, f"{count:,}")
                            current_source = source
                        else:
                            rel_table.add_row(
                                "", rel_type, f"{count:,}"
                            )  # Indent or leave source blank

                    console.print(rel_table)
                    console.print()
            except Exception as e:
                logger.error(
                    f"Error generating relationship statistics: {str(e)}", exc_info=True
                )
                console.print(
                    f"[red]Error:[/][yellow] Could not generate relationship statistics: {str(e)}[/]"
                )
                # Roll back to clean state
                conn.rollback()

            # Commit the transaction since we're done
            conn.commit()

            console.print(
                f"Leaderboard generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
    except Exception as outer_e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in display_leaderboard: {str(outer_e)}")
        console.print(f"[red]Error:[/] {str(outer_e)}")
        if conn and not conn.closed:
            conn.rollback()
    finally:
        # Always close the connection when done
        if conn and not conn.closed:
            conn.close()


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
def lookup_by_id(cur, word_id: int, console: Console):
    """Look up a word by its ID and display its information."""
    try:
        cur.execute(
            """
            SELECT lemma, language_code, has_baybayin, baybayin_form, romanized_form
            FROM words
            WHERE id = %s
        """,
            (word_id,),
        )

        result = cur.fetchone()

        if not result:
            console.print(f"[yellow]Word with ID {word_id} not found.[/]")
            return

        lemma, language_code, has_baybayin, baybayin_form, romanized_form = result

        console.print(f"\n[bold]Word Information - ID: {word_id}[/]")
        console.print(f"Lemma: {lemma}")
        console.print(f"Language: {'Tagalog' if language_code == 'tl' else 'Cebuano'}")

        if has_baybayin and baybayin_form:
            console.print(f"Baybayin Form: {baybayin_form}")
            if romanized_form:
                console.print(f"Romanized Form: {romanized_form}")

        cur.execute(
            """
            SELECT p.name_tl as pos, d.definition_text
            FROM definitions d
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE d.word_id = %s
            ORDER BY p.name_tl, d.created_at
        """,
            (word_id,),
        )

        definitions = cur.fetchall()

        if definitions:
            console.print("\n[bold]Definitions:[/]")
            current_pos = None
            for pos, definition in definitions:
                if pos != current_pos:
                    console.print(f"\n[cyan]{pos or 'Uncategorized'}[/]")
                    current_pos = pos
                console.print(f"• {definition}")

        cur.execute(
            """
            SELECT r.relation_type, w.lemma
            FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = %s
            ORDER BY r.relation_type, w.lemma
        """,
            (word_id,),
        )

        relations = cur.fetchall()

        if relations:
            console.print("\n[bold]Related Words:[/]")
            current_type = None
            for rel_type, rel_word in relations:
                if rel_type != current_type:
                    console.print(f"\n[magenta]{rel_type.title()}[/]")
                    current_type = rel_type
                console.print(f"• {rel_word}")

        input("\nPress Enter to continue...")

    except Exception as e:
        logger.error(f"Error looking up word ID {word_id}: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/]")


def test_database():
    """Run database connectivity tests."""
    console = Console()
    console.print("\n[bold cyan]🧪 Database Connection Tests[/]", justify="center")

    tests = [
        ("Database Connection", lambda: get_connection()),
        ("PostgreSQL Version", lambda: check_pg_version()),
        ("Tables Existence", lambda: check_tables_exist()),
        ("Extensions", lambda: check_extensions()),
        ("Data Access", lambda: check_data_access()),
        ("Query Performance", lambda: check_query_performance()),
    ]

    test_table = Table(box=box_ROUNDED)
    test_table.add_column("Test", style="cyan")
    test_table.add_column("Status", style="bold")
    test_table.add_column("Details", style="dim")

    conn = None

    try:
        for test_name, test_func in tests:
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"Running {test_name} test..."),
                    console=console,
                ) as progress:
                    task = progress.add_task("Testing", total=1)
                    result, details = test_func()
                    progress.update(task, completed=1)

                    if result:
                        test_table.add_row(test_name, "[green]PASS[/]", details)
                    else:
                        test_table.add_row(test_name, "[red]FAIL[/]", details)
            except Exception as e:
                test_table.add_row(test_name, "[red]ERROR[/]", str(e))
    finally:
        if conn:
            conn.close()

    console.print(test_table)

# -------------------------------------------------------------------
# CLI Wrapper Functions
# -------------------------------------------------------------------
def create_argument_parser_cli() -> argparse.ArgumentParser:
    return create_argument_parser()


def migrate_data_cli(args):
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
    verify_database(args)


def purge_database_cli(args):
    """Run purge operations on the dictionary database."""
    try:
        cur = get_cursor()
        console = Console()
        console.print("[bold blue]Starting dictionary purge process...[/]")

        console.print("[yellow]Purging all dictionary data...[/]")
        purge_database_tables(cur)

        console.print("[bold green]Dictionary purge completed successfully.[/]")
    except Exception as e:
        print(f"Error during purge: {str(e)}")


def lookup_word_cli(args):
    lookup_word(args)


def display_leaderboard_cli(args):
    """Display a leaderboard of dictionary contributors."""
    console = Console()
    display_leaderboard(get_cursor(), console)


def explore_dictionary_cli(args):
    explore_dictionary()


def test_database_cli(args):
    test_database()


def display_help_cli(args):
    display_help(args)


def cleanup_database_cli(args):
    """Run cleanup routines on the dictionary database."""
    try:
        cur = get_cursor()
        console = Console()
        console.print("[bold blue]Starting dictionary cleanup process...[/]")

        console.print("[yellow]Deduplicating definitions...[/]")
        deduplicate_definitions(cur)

        console.print("[yellow]Cleaning up relations...[/]")
        cleanup_relations(cur)

        console.print("[yellow]Standardizing formats...[/]")
        cleanup_dictionary_data(cur)

        console.print("[bold green]Dictionary cleanup completed successfully.[/]")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
def main():
    parser = create_argument_parser_cli()
    args = parser.parse_args()
    if args.command == "migrate":
        migrate_data_cli(args)
    elif args.command == "verify":
        verify_database_cli(args)
    elif args.command == "purge":
        purge_database_cli(args)
    elif args.command == "lookup":
        lookup_word_cli(args)
    elif args.command == "stats":
        display_dictionary_stats_cli(args)
    elif args.command == "leaderboard":
        display_leaderboard_cli(args)
    elif args.command == "help":
        display_help_cli(args)
    elif args.command == "test":
        test_database_cli(args)
    elif args.command == "explore":
        explore_dictionary_cli(args)
    elif args.command == "cleanup":
        cleanup_database_cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    setup_logging() # Activate logging configuration
    main()
