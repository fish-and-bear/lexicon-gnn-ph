import os
import json
import re
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_params():
    """Loads database connection parameters from .env file."""
    load_dotenv()  # Load environment variables from .env file

    # Map environment variable names to psycopg2 parameter names
    env_to_psycopg_map = {
        'DB_NAME': 'dbname',
        'DB_USER': 'user',
        'DB_PASSWORD': 'password',
        'DB_HOST': 'host',
        'DB_PORT': 'port'
    }
    required_vars = list(env_to_psycopg_map.keys())

    db_params = {}
    missing_vars = []

    for env_var in required_vars:
        value = os.getenv(env_var)
        psycopg_param = env_to_psycopg_map[env_var]

        if value is None:
            # Allow host and port to be optional, default later if needed
            if env_var not in ['DB_HOST', 'DB_PORT']:
                missing_vars.append(env_var)
        else:
            db_params[psycopg_param] = value

    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Set defaults for host and port if not provided
    if 'host' not in db_params:
        db_params['host'] = 'localhost'
        logger.info("DB_HOST not found in .env, defaulting to 'localhost'.")
    if 'port' not in db_params:
        db_params['port'] = '5432'
        logger.info("DB_PORT not found in .env, defaulting to '5432'.")

    return db_params

def fetch_schema_data(conn):
    """Fetches tables, columns, constraints, indexes, and basic stats from the database."""
    cursor = conn.cursor(cursor_factory=DictCursor)
    schema_data = {
        'tables': {},
        'columns': {},
        'constraints': {},
        'indexes': {}, # Will store {table_name: {index_definition: index_row_dict}}
        'stats': {}
    }

    # Columns to get distinct/example values for
    # Key: table_name, Value: list of column_names
    categorical_columns_to_analyze = {
        'languages': ['code', 'region', 'family', 'status'],
        'words': ['language_code', 'is_proper_noun', 'is_abbreviation', 'is_initialism'], # Added booleans
        'definitions': ['original_pos'],
        'parts_of_speech': ['code', 'name_en', 'name_tl'],
        'relations': ['relation_type'],
        'affixations': ['affix_type'],
        'pronunciations': ['type'],
        'definition_relations': ['relation_type'],
        'definition_categories': ['category_kind'],
        'definition_examples': ['example_type'], # Added
        'word_forms': ['is_canonical', 'is_primary'], # Added booleans
        'word_templates': ['template_name']
    }
    MAX_DISTINCT_VALUES_TO_LIST = 25 # Keep listing all values if count <= this
    NUM_EXAMPLE_VALUES = 10 # Increase number of examples shown

    # Fetch Tables and Comments
    cursor.execute("""
        SELECT
            t.table_name,
            pg_catalog.obj_description(c.oid, 'pg_class') as table_comment
        FROM
            information_schema.tables t
        JOIN
            pg_catalog.pg_class c ON c.relname = t.table_name
        JOIN
            pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE
            t.table_schema = 'public'
            AND t.table_type = 'BASE TABLE'
            AND n.nspname = 'public';
    """)
    table_names = []
    for row in cursor.fetchall():
        table_name = row['table_name']
        table_names.append(table_name)
        schema_data['tables'][table_name] = {'comment': row['table_comment']}
        schema_data['stats'][table_name] = {'columns': {}} # Init stats for table

    # Fetch Columns
    cursor.execute("""
        SELECT
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            datetime_precision
        FROM
            information_schema.columns
        WHERE
            table_schema = 'public'
        ORDER BY
            table_name, ordinal_position;
    """)
    for row in cursor.fetchall():
        table_name = row['table_name']
        if table_name not in schema_data['columns']:
            schema_data['columns'][table_name] = []
        schema_data['columns'][table_name].append(dict(row))

    # Fetch Constraints (PK, FK, UNIQUE, CHECK)
    cursor.execute("""
        SELECT
            tc.table_name,
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name,
            rc.update_rule AS on_update,
            rc.delete_rule AS on_delete,
            chk.check_clause
        FROM
            information_schema.table_constraints AS tc
        JOIN
            information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
        LEFT JOIN
            information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
        LEFT JOIN
            information_schema.referential_constraints AS rc
            ON tc.constraint_name = rc.constraint_name AND tc.constraint_schema = rc.constraint_schema
        LEFT JOIN
            information_schema.check_constraints AS chk
            ON tc.constraint_name = chk.constraint_name AND tc.constraint_schema = chk.constraint_schema
        WHERE
            tc.table_schema = 'public'
        ORDER BY
            tc.table_name, tc.constraint_name;
    """)
    for row in cursor.fetchall():
        table_name = row['table_name']
        constraint_name = row['constraint_name']
        if table_name not in schema_data['constraints']:
            schema_data['constraints'][table_name] = {}
        if constraint_name not in schema_data['constraints'][table_name]:
            schema_data['constraints'][table_name][constraint_name] = {
                'type': row['constraint_type'],
                'columns': [],
                'foreign_table_name': row['foreign_table_name'],
                'foreign_column_name': row['foreign_column_name'],
                'on_update': row['on_update'],
                'on_delete': row['on_delete'],
                'check_clause': row['check_clause']
            }
        if row['column_name'] not in schema_data['constraints'][table_name][constraint_name]['columns']:
             schema_data['constraints'][table_name][constraint_name]['columns'].append(row['column_name'])

    # Fetch Indexes
    cursor.execute("""
        SELECT
            ix.schemaname AS schema_name,
            ix.tablename AS table_name,
            ix.indexname AS index_name,
            pg_get_indexdef(idx.indexrelid) AS index_definition, -- Get the full CREATE INDEX command
            idx.indisunique AS is_unique,
            am.amname AS index_method
        FROM
            pg_catalog.pg_indexes ix
        JOIN
            pg_catalog.pg_class ixc ON ix.indexname = ixc.relname
        JOIN
            pg_catalog.pg_index idx ON ixc.oid = idx.indexrelid
        JOIN
            pg_catalog.pg_am am ON ixc.relam = am.oid
        WHERE
            ix.schemaname = 'public' -- Adjust schema if needed
        ORDER BY
            ix.tablename, ix.indexname;
    """)
    raw_indexes = cursor.fetchall()
    processed_indexes = {} # {table_name: {index_definition: index_dict}}
    index_name_map = {} # {index_definition: first_encountered_index_name}

    for index in raw_indexes:
        table_name = index['table_name']
        index_def = index['index_definition']

        if table_name not in processed_indexes:
            processed_indexes[table_name] = {}
            schema_data['indexes'][table_name] = [] # Initialize list for this table

        if index_def not in processed_indexes[table_name]:
            # Store the first name encountered for this definition
            if index_def not in index_name_map:
                 index_name_map[index_def] = index['index_name']

            index_data = {
                "name": index_name_map[index_def], # Use consistent name
                "unique": index['is_unique'],
                "method": index['index_method'],
                "columns": [], # Column names are part of the definition string
                "definition": index_def
            }
            processed_indexes[table_name][index_def] = index_data
            schema_data['indexes'][table_name].append(index_data)
        # else: This definition is already recorded for this table, skip duplicate row from pg_indexes

    # Fetch Stats (Row Counts and Distinct/Example Values)
    logger.info("Fetching table row counts and column statistics...")
    for table_name in schema_data['tables']:
        if table_name not in schema_data['stats']:
            schema_data['stats'][table_name] = {'columns': {}, 'row_count': 0}

        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM public.{table_name};")
            count_result = cursor.fetchone()
            schema_data['stats'][table_name]['row_count'] = count_result[0] if count_result else 0
        except Exception as e:
            logger.error(f"Could not get row count for {table_name}: {e}")
            schema_data['stats'][table_name]['row_count'] = -1 # Indicate error

        # Get distinct/example values for specified columns
        if table_name in categorical_columns_to_analyze:
            for col_name in categorical_columns_to_analyze[table_name]:
                # Check if column exists in fetched schema for the table
                if col_name not in [c['column_name'] for c in schema_data['columns'].get(table_name, [])]:
                    logger.warning(f"Column '{col_name}' specified for stats analysis not found in table '{table_name}'. Skipping stats for this column.")
                    continue

                col_stats = {}
                try:
                    # Get distinct count
                    cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM public."{table_name}";')
                    distinct_count_result = cursor.fetchone()
                    distinct_count = distinct_count_result[0] if distinct_count_result else 0
                    col_stats['distinct_count'] = distinct_count

                    # Get distinct values or examples
                    if distinct_count > 0:
                        if distinct_count <= MAX_DISTINCT_VALUES_TO_LIST:
                            # Fetch all distinct values
                            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM public."{table_name}" ORDER BY "{col_name}" LIMIT {MAX_DISTINCT_VALUES_TO_LIST + 1};')
                            values = [row[0] for row in cursor.fetchall()]
                            col_stats['distinct_values'] = values
                        else:
                            # Fetch example values (including NULL if present)
                            example_query = f'''
                                (
                                    SELECT DISTINCT "{col_name}"
                                    FROM public."{table_name}"
                                    WHERE "{col_name}" IS NOT NULL
                                    ORDER BY "{col_name}" -- Order helps get consistent examples
                                    LIMIT {NUM_EXAMPLE_VALUES}
                                )
                                UNION ALL
                                (
                                    SELECT NULL
                                    WHERE EXISTS (SELECT 1 FROM public."{table_name}" WHERE "{col_name}" IS NULL)
                                    LIMIT 1 -- Only need to know if NULL exists
                                )
                                LIMIT {NUM_EXAMPLE_VALUES};
                            '''
                            cursor.execute(example_query)
                            values = [row[0] for row in cursor.fetchall()]
                            # Check if NULL exists but wasn't picked
                            if None not in values and distinct_count > 0:
                                cursor.execute(f'SELECT EXISTS (SELECT 1 FROM public."{table_name}" WHERE "{col_name}" IS NULL);')
                                null_exists_result = cursor.fetchone()
                                if null_exists_result and null_exists_result[0]:
                                    pass # UNION ALL query should include NULL if it exists within the LIMIT

                            col_stats['example_values'] = values

                except psycopg2.Error as pg_err: # Catch specific psycopg2 errors
                    logger.error(f'Could not get stats for {table_name}."{col_name}": [{pg_err.pgcode}] {pg_err}')
                    conn.rollback() # Rollback transaction on error
                    col_stats['error'] = f"PostgreSQL Error: {pg_err.pgcode}"
                except Exception as e:
                    logger.error(f"Unexpected error getting stats for {table_name}.{col_name}: {e}")
                    col_stats['error'] = str(e)

                schema_data['stats'][table_name]['columns'][col_name] = col_stats
    logger.info("Finished fetching statistics.")

    cursor.close()
    return schema_data

def build_json_schema(data):
    """Builds a JSON structure from the fetched schema data."""
    output_schema = {}
    for table_name, table_info in data['tables'].items():
        output_schema[table_name] = {
            "comment": table_info.get('comment'),
            "stats": data['stats'].get(table_name, {}), # Include stats
            "columns": data['columns'].get(table_name, []),
            "constraints": data['constraints'].get(table_name, []),
            "indexes": data['indexes'].get(table_name, [])
        }
    return output_schema


if __name__ == "__main__":
    logger.info("Starting schema generation...")
    connection = None
    try:
        db_parameters = get_db_params()
        logger.info("Connecting to the database...")
        connection = psycopg2.connect(**db_parameters)
        logger.info("Database connection successful.")

        logger.info("Fetching schema data...")
        schema_details = fetch_schema_data(connection)
        logger.info("Schema data fetched.")

        logger.info("Building JSON schema...")
        json_schema = build_json_schema(schema_details)
        logger.info("JSON schema built.")

        output_filename = "actual_schema.json"
        logger.info(f"Writing schema to {output_filename}...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_schema, f, indent=2, ensure_ascii=False)
        logger.info(f"Schema successfully written to {output_filename}.")

    except EnvironmentError as env_err:
        logger.error(f"Configuration error: {env_err}")
    except psycopg2.Error as db_err:
        logger.error(f"Database error: {db_err}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if connection:
            connection.close()
            logger.info("Database connection closed.")
        logger.info("Schema generation script finished.") 