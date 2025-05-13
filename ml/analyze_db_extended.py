#!/usr/bin/env python3

"""
Extended database analysis script that provides more detailed insights
about the Filipino lexical database.
"""

import os
import sys
import logging
import argparse
import json
import psycopg2
import psycopg2.extras
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def get_db_connection(config=None):
    """Establish a read-only database connection using credentials."""

    conn = None
    tried_methods = []

    # Load environment variables from .env
    # First try the root directory
    root_env_path = Path(__file__).parent.parent / '.env'
    backend_env_path = Path(__file__).parent.parent / 'backend' / '.env'
    
    # Try root directory first, then backend
    if root_env_path.exists():
        env_path = root_env_path
        logger.info(f"Using .env file from project root directory: {env_path}")
    else:
        env_path = backend_env_path
        logger.info(f"Using .env file from backend directory: {env_path}")
    
    load_dotenv(dotenv_path=env_path)

    # Get credentials from environment variables or config
    if config and 'db_config' in config:
        db_host = config['db_config'].get('host', os.getenv("DB_HOST", "localhost"))
        db_port = config['db_config'].get('port', os.getenv("DB_PORT", "5432"))
        db_name = config['db_config'].get('dbname', os.getenv("DB_NAME", "fil_dict_db"))
        db_user = config['db_config'].get('user', os.getenv("DB_USER", "postgres"))
        db_password = config['db_config'].get('password', os.getenv("DB_PASSWORD", ""))
    else:
        # Get credentials from environment variables
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "fil_dict_db")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "")

    logger.info(f"Loading database configuration")

    try:
        # Method 1: Try with the configured parameters
        try:
            logger.info(f"Trying to connect to database {db_name} at {db_host}:{db_port}...")

            tried_methods.append("configured parameters")

            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                dbname=db_name,
                user=db_user,
                password=db_password,
                application_name="FilRelex-ReadOnlyAnalysis"
            )

        except psycopg2.OperationalError as e:
            logger.warning(f"Failed to connect with configured parameters: {e}")

            # Method 2: Try with trust authentication (no password)
            if "no password supplied" in str(e) or "password authentication failed" in str(e):
                logger.info("Trying trust authentication (no password)...")
                tried_methods.append("trust authentication")

                try:
                    conn = psycopg2.connect(
                        host=db_host,
                        port=db_port,
                        dbname=db_name,
                        user=db_user,
                        application_name="FilRelex-ReadOnlyAnalysis"
                    )
                except psycopg2.OperationalError as trust_err:
                    logger.warning(f"Trust authentication failed: {trust_err}")

                    # Method 3: Try default 'postgres' database
                    logger.info("Trying to connect to default 'postgres' database...")
                    tried_methods.append("default postgres database")

                    try:
                        conn = psycopg2.connect(
                            host=db_host,
                            port=db_port,
                            dbname="postgres",
                            user=db_user,
                            application_name="FilRelex-ReadOnlyAnalysis"
                        )
                    except psycopg2.OperationalError as default_err:
                        logger.warning(f"Default database connection failed: {default_err}")
                        raise Exception(f"All connection methods failed: {tried_methods}")
            else:
                # Some other connection error occurred
                raise

        # Set read-only mode
        if conn:
            conn.set_session(readonly=True, autocommit=False)
            logger.info(f"Connected to database {conn.info.dbname} in READ-ONLY mode")

        return conn

    except Exception as e:
        logger.error(f"Failed to connect to database after trying: {tried_methods}")
        logger.error(f"Error: {e}")
        if conn and not conn.closed:
            conn.close()
        raise

def analyze_word_counts(conn, verbose=False):
    """Analyze word counts by language and other attributes."""
    logger.info("Analyzing word counts...")
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # Total word count
        cur.execute("SELECT COUNT(*) FROM words")
        total_words = cur.fetchone()[0]
        logger.info(f"Total words in database: {total_words}")
        
        # Words by language
        logger.info("Analyzing words by language...")
        cur.execute("""
            SELECT language_code, COUNT(*) as count 
            FROM words 
            GROUP BY language_code 
            ORDER BY count DESC
        """)
        language_counts = cur.fetchall()
        
        if verbose:
            print("\n===== Words by Language =====")
            print(f"{'Language':<10} | {'Count':<10} | {'Percentage':<10}")
            print("-" * 35)
            for row in language_counts:
                lang = row['language_code'] or 'Unknown'
                count = row['count']
                percentage = (count / total_words) * 100
                print(f"{lang:<10} | {count:<10} | {percentage:.2f}%")
        
        # Words with definitions
        logger.info("Analyzing words with definitions...")
        cur.execute("""
            SELECT COUNT(DISTINCT w.id) 
            FROM words w
            JOIN definitions d ON w.id = d.word_id
        """)
        words_with_defs = cur.fetchone()[0]
        percentage = (words_with_defs / total_words) * 100
        logger.info(f"Words with definitions: {words_with_defs} ({percentage:.2f}%)")
        
        # Words with etymologies
        logger.info("Analyzing words with etymologies...")
        cur.execute("""
            SELECT COUNT(DISTINCT w.id) 
            FROM words w
            JOIN etymologies e ON w.id = e.word_id
        """)
        words_with_etym = cur.fetchone()[0]
        percentage = (words_with_etym / total_words) * 100
        logger.info(f"Words with etymologies: {words_with_etym} ({percentage:.2f}%)")
        
        # Words with relations
        logger.info("Analyzing words with relations...")
        cur.execute("""
            SELECT COUNT(DISTINCT w.id) 
            FROM words w
            JOIN relations r ON w.id = r.from_word_id OR w.id = r.to_word_id
        """)
        words_with_rel = cur.fetchone()[0]
        percentage = (words_with_rel / total_words) * 100
        logger.info(f"Words with relations: {words_with_rel} ({percentage:.2f}%)")
        
        return {
            'total_words': total_words,
            'language_counts': [dict(row) for row in language_counts],
            'words_with_definitions': words_with_defs,
            'words_with_etymologies': words_with_etym,
            'words_with_relations': words_with_rel
        }

def analyze_definitions(conn, verbose=False):
    """Analyze definition statistics."""
    logger.info("Analyzing definitions...")
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # Total definition count
        cur.execute("SELECT COUNT(*) FROM definitions")
        total_defs = cur.fetchone()[0]
        logger.info(f"Total definitions in database: {total_defs}")
        
        # Definitions by part of speech
        logger.info("Analyzing definitions by part of speech...")
        cur.execute("""
            SELECT COALESCE(p.name_tl, d.original_pos, 'Unknown') as pos, COUNT(*) as count 
            FROM definitions d
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            GROUP BY pos
            ORDER BY count DESC
        """)
        pos_counts = cur.fetchall()
        
        if verbose:
            print("\n===== Definitions by Part of Speech =====")
            print(f"{'POS':<15} | {'Count':<10} | {'Percentage':<10}")
            print("-" * 40)
            for row in pos_counts:
                pos = row['pos'] or 'Unknown'
                count = row['count']
                percentage = (count / total_defs) * 100
                print(f"{pos:<15} | {count:<10} | {percentage:.2f}%")
        
        # Definitions with examples
        logger.info("Analyzing definitions with examples...")
        cur.execute("""
            SELECT COUNT(*) 
            FROM definitions
            WHERE examples IS NOT NULL AND examples != ''
        """)
        defs_with_examples = cur.fetchone()[0]
        percentage = (defs_with_examples / total_defs) * 100
        logger.info(f"Definitions with examples: {defs_with_examples} ({percentage:.2f}%)")
        
        # Definitions by source
        logger.info("Analyzing definitions by source...")
        cur.execute("""
            SELECT COALESCE(sources, 'Unknown') as source, COUNT(*) as count 
            FROM definitions
            GROUP BY source
            ORDER BY count DESC
            LIMIT 10
        """)
        source_counts = cur.fetchall()
        
        if verbose:
            print("\n===== Top 10 Definition Sources =====")
            print(f"{'Source':<30} | {'Count':<10} | {'Percentage':<10}")
            print("-" * 55)
            for row in source_counts:
                source = row['source']
                if len(source) > 27:
                    source = source[:24] + "..."
                count = row['count']
                percentage = (count / total_defs) * 100
                print(f"{source:<30} | {count:<10} | {percentage:.2f}%")
                
        return {
            'total_definitions': total_defs,
            'pos_counts': [dict(row) for row in pos_counts],
            'definitions_with_examples': defs_with_examples,
            'source_counts': [dict(row) for row in source_counts]
        }

def analyze_relations(conn, verbose=False):
    """Analyze relation statistics."""
    logger.info("Analyzing relations...")
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # Total relation count
        cur.execute("SELECT COUNT(*) FROM relations")
        total_relations = cur.fetchone()[0]
        logger.info(f"Total relations in database: {total_relations}")
        
        # Relations by type
        logger.info("Analyzing relations by type...")
        cur.execute("""
            SELECT relation_type, COUNT(*) as count 
            FROM relations
            GROUP BY relation_type
            ORDER BY count DESC
        """)
        relation_type_counts = cur.fetchall()
        
        if verbose:
            print("\n===== Relations by Type =====")
            print(f"{'Relation Type':<20} | {'Count':<10} | {'Percentage':<10}")
            print("-" * 45)
            for row in relation_type_counts:
                rel_type = row['relation_type'] or 'Unknown'
                count = row['count']
                percentage = (count / total_relations) * 100
                print(f"{rel_type:<20} | {count:<10} | {percentage:.2f}%")
        
        # Self-loops (relations to the same word)
        logger.info("Analyzing self-relations...")
        cur.execute("""
            SELECT COUNT(*) 
            FROM relations
            WHERE from_word_id = to_word_id
        """)
        self_relations = cur.fetchone()[0]
        percentage = (self_relations / total_relations) * 100 if total_relations > 0 else 0
        logger.info(f"Self-relations: {self_relations} ({percentage:.2f}%)")
        
        # Average relations per word
        logger.info("Analyzing average relations per word...")
        cur.execute("""
            SELECT COUNT(DISTINCT w.id) as word_count
            FROM words w
            JOIN relations r ON w.id = r.from_word_id OR w.id = r.to_word_id
        """)
        words_with_relations = cur.fetchone()['word_count']
        avg_relations = total_relations / words_with_relations if words_with_relations > 0 else 0
        logger.info(f"Average relations per word: {avg_relations:.2f}")
        
        return {
            'total_relations': total_relations,
            'relation_type_counts': [dict(row) for row in relation_type_counts],
            'self_relations': self_relations,
            'avg_relations_per_word': avg_relations
        }

def analyze_etymologies(conn, verbose=False):
    """Analyze etymology statistics."""
    logger.info("Analyzing etymologies...")
    
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # Total etymology count
        cur.execute("SELECT COUNT(*) FROM etymologies")
        total_etyms = cur.fetchone()[0]
        logger.info(f"Total etymologies in database: {total_etyms}")
        
        # Etymologies with language codes
        logger.info("Analyzing etymologies with language codes...")
        cur.execute("""
            SELECT COUNT(*) 
            FROM etymologies
            WHERE language_codes IS NOT NULL AND language_codes != '{}'
        """)
        etyms_with_langs = cur.fetchone()[0]
        percentage = (etyms_with_langs / total_etyms) * 100 if total_etyms > 0 else 0
        logger.info(f"Etymologies with language codes: {etyms_with_langs} ({percentage:.2f}%)")
        
        # Check language_codes column type
        logger.info("Checking language_codes column type...")
        cur.execute("""
            SELECT data_type 
            FROM information_schema.columns 
            WHERE table_name = 'etymologies' AND column_name = 'language_codes'
        """)
        col_type = cur.fetchone()
        
        # Top source languages
        logger.info(f"Analyzing top etymology source languages (column type: {col_type['data_type'] if col_type else 'unknown'})...")
        
        try:
            # Try with string_to_array if it's a text type
            cur.execute("""
                SELECT lang_code, COUNT(*) as count
                FROM (
                    SELECT TRIM(UNNEST(string_to_array(REPLACE(REPLACE(language_codes, '{', ''), '}', ''), ','))) as lang_code
                    FROM etymologies
                    WHERE language_codes IS NOT NULL AND language_codes != '{}'
                ) AS lang_codes
                GROUP BY lang_code
                ORDER BY count DESC
                LIMIT 10
            """)
            lang_code_counts = cur.fetchall()
        except psycopg2.Error as e:
            logger.warning(f"Error using string_to_array approach: {e}")
            try:
                # Alternative approach: treat as text and extract with regex
                cur.execute("""
                    SELECT SUBSTRING(language_codes FROM '([^{},]+)') as lang_code, COUNT(*) as count
                    FROM etymologies
                    WHERE language_codes IS NOT NULL AND language_codes != '{}'
                    GROUP BY lang_code
                    ORDER BY count DESC
                    LIMIT 10
                """)
                lang_code_counts = cur.fetchall()
            except psycopg2.Error as e2:
                logger.error(f"Error extracting language codes with regex: {e2}")
                lang_code_counts = []
        
        if verbose and lang_code_counts:
            print("\n===== Top 10 Etymology Source Languages =====")
            print(f"{'Language':<10} | {'Count':<10} | {'Percentage':<10}")
            print("-" * 35)
            for row in lang_code_counts:
                lang = row['lang_code'] or 'Unknown'
                count = row['count']
                percentage = (count / etyms_with_langs) * 100 if etyms_with_langs > 0 else 0
                print(f"{lang:<10} | {count:<10} | {percentage:.2f}%")
        
        return {
            'total_etymologies': total_etyms,
            'etymologies_with_language_codes': etyms_with_langs,
            'top_source_languages': [dict(row) for row in lang_code_counts] if lang_code_counts else []
        }

def generate_visualizations(results, output_dir='./visualizations'):
    """Generate visualizations from analysis results."""
    logger.info("Generating visualizations...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Language distribution pie chart
        if 'language_counts' in results and results['language_counts']:
            plt.figure(figsize=(10, 8))
            langs = [item['language_code'] or 'Unknown' for item in results['language_counts']]
            counts = [item['count'] for item in results['language_counts']]
            
            # Combine small slices into 'Other'
            threshold = 0.02  # 2%
            total = sum(counts)
            small_indices = [i for i, count in enumerate(counts) if count/total < threshold]
            
            if small_indices:
                other_count = sum(counts[i] for i in small_indices)
                filtered_langs = [lang for i, lang in enumerate(langs) if i not in small_indices]
                filtered_counts = [count for i, count in enumerate(counts) if i not in small_indices]
                
                filtered_langs.append('Other')
                filtered_counts.append(other_count)
                
                plt.pie(filtered_counts, labels=filtered_langs, autopct='%1.1f%%', startangle=90)
            else:
                plt.pie(counts, labels=langs, autopct='%1.1f%%', startangle=90)
                
            plt.axis('equal')
            plt.title('Word Distribution by Language')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'language_distribution.png'))
            plt.close()
            logger.info(f"Language distribution chart saved to {output_dir}/language_distribution.png")
        
        # Relations by type bar chart
        if 'relation_type_counts' in results and results['relation_type_counts']:
            plt.figure(figsize=(12, 6))
            relation_types = [item['relation_type'] or 'Unknown' for item in results['relation_type_counts']]
            relation_counts = [item['count'] for item in results['relation_type_counts']]
            
            # Limit to top 10
            if len(relation_types) > 10:
                relation_types = relation_types[:10]
                relation_counts = relation_counts[:10]
                
            y_pos = range(len(relation_types))
            plt.barh(y_pos, relation_counts)
            plt.yticks(y_pos, relation_types)
            plt.xlabel('Count')
            plt.title('Top Relation Types')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'relation_types.png'))
            plt.close()
            logger.info(f"Relation types chart saved to {output_dir}/relation_types.png")
            
        # Database coverage statistics
        if all(key in results for key in ['total_words', 'words_with_definitions', 
                                         'words_with_etymologies', 'words_with_relations']):
            plt.figure(figsize=(10, 6))
            categories = ['Definitions', 'Etymologies', 'Relations']
            coverage = [
                results['words_with_definitions'] / results['total_words'] * 100,
                results['words_with_etymologies'] / results['total_words'] * 100,
                results['words_with_relations'] / results['total_words'] * 100
            ]
            
            plt.bar(categories, coverage)
            plt.ylabel('Coverage (%)')
            plt.title('Database Coverage Statistics')
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'database_coverage.png'))
            plt.close()
            logger.info(f"Database coverage chart saved to {output_dir}/database_coverage.png")
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extended Database Analysis Tool')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Display detailed output')
    
    parser.add_argument('--output', '-o', type=str, default='./analysis_results',
                        help='Directory to save analysis results')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    parser.add_argument('--db-config', type=str,
                        help='Path to database configuration JSON file')
    
    parser.add_argument('--analyze-words', action='store_true',
                        help='Analyze word statistics')
    
    parser.add_argument('--analyze-definitions', action='store_true',
                        help='Analyze definition statistics')
    
    parser.add_argument('--analyze-relations', action='store_true',
                        help='Analyze relation statistics')
    
    parser.add_argument('--analyze-etymologies', action='store_true',
                        help='Analyze etymology statistics')
    
    parser.add_argument('--analyze-all', action='store_true',
                        help='Analyze all data types')
    
    parser.add_argument('--export-json', action='store_true',
                        help='Export results to JSON file')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    return parser.parse_args()

def main():
    """Main function to run the database analysis."""
    args = parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    if args.verbose:
        logger.debug("Verbose logging enabled")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load database config if provided
    config = None
    if args.db_config:
        try:
            with open(args.db_config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded database configuration from {args.db_config}")
        except Exception as e:
            logger.error(f"Error loading database config file: {e}")
            return
    
    try:
        # Connect to database
        conn = get_db_connection(config)
        
        # Initialize results dict
        all_results = {}
        
        # Analyze words if requested or if analyze_all
        if args.analyze_words or args.analyze_all:
            word_results = analyze_word_counts(conn, args.verbose)
            all_results.update(word_results)
            
        # Analyze definitions if requested or if analyze_all
        if args.analyze_definitions or args.analyze_all:
            definition_results = analyze_definitions(conn, args.verbose)
            all_results.update(definition_results)
            
        # Analyze relations if requested or if analyze_all
        if args.analyze_relations or args.analyze_all:
            relation_results = analyze_relations(conn, args.verbose)
            all_results.update(relation_results)
            
        # Analyze etymologies if requested or if analyze_all
        if args.analyze_etymologies or args.analyze_all:
            etymology_results = analyze_etymologies(conn, args.verbose)
            all_results.update(etymology_results)
            
        # Generate visualizations if requested
        if args.visualize and all_results:
            visualizations_dir = os.path.join(args.output, 'visualizations')
            generate_visualizations(all_results, visualizations_dir)
            
        # Export results to JSON if requested
        if args.export_json and all_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_file = os.path.join(args.output, f'analysis_results_{timestamp}.json')
            
            with open(json_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"Analysis results exported to {json_file}")
        
        # If no specific analysis was requested, show basic stats
        if not any([args.analyze_words, args.analyze_definitions, 
                   args.analyze_relations, args.analyze_etymologies, args.analyze_all]):
            logger.info("No specific analysis requested, showing basic stats")
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM words")
                total_words = cur.fetchone()[0]
                logger.info(f"Total words in database: {total_words}")
                
                cur.execute("SELECT COUNT(*) FROM definitions")
                total_defs = cur.fetchone()[0]
                logger.info(f"Total definitions: {total_defs}")
                
                cur.execute("SELECT COUNT(*) FROM relations")
                total_relations = cur.fetchone()[0]
                logger.info(f"Total relations: {total_relations}")
                
                cur.execute("SELECT COUNT(*) FROM etymologies")
                total_etyms = cur.fetchone()[0]
                logger.info(f"Total etymologies: {total_etyms}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            logger.info("Database connection closed")
    
if __name__ == "__main__":
    main() 