import os
import json
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import sys

def get_db_params():
    """Loads database connection parameters from .env file in the parent directory."""
    # Construct the path to the .env file in the parent directory
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if not os.path.exists(dotenv_path):
        print(f"Error: .env file not found at {dotenv_path}")
        print("Please ensure the .env file exists in the project root directory.")
        sys.exit(1)

    load_dotenv(dotenv_path=dotenv_path)

    params = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432')
    }
    if not all(params.values()):
        print("Error: Missing database connection details in .env file.")
        print("Required: DB_NAME, DB_USER, DB_PASSWORD. Optional: DB_HOST, DB_PORT.")
        sys.exit(1)
    return params

def fetch_schema_data(conn):
    """Fetches tables, columns, constraints, and indexes from the database."""
    cursor = conn.cursor(cursor_factory=DictCursor)
    schema_data = {
        'tables': {},
        'columns': {},
        'constraints': {},
        'indexes': {}
    }

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
    for row in cursor.fetchall():
        schema_data['tables'][row['table_name']] = {'comment': row['table_comment']}

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
            pg_get_indexdef(i.indexrelid) AS index_definition,
            idx.indisunique AS is_unique,
            am.amname as index_method -- Get index method (btree, gin, gist, etc.)
        FROM
            pg_indexes ix
        JOIN
            pg_class c ON c.relname = ix.tablename
        JOIN
            pg_index idx ON idx.indrelid = c.oid
        JOIN
            pg_class i ON i.oid = idx.indexrelid
        JOIN
            pg_am am ON am.oid = i.relam -- Join with pg_am for method name
        WHERE
            ix.schemaname = 'public'
        ORDER BY
            ix.tablename, ix.indexname;
    """)
    for row in cursor.fetchall():
        table_name = row['table_name']
        if table_name not in schema_data['indexes']:
            schema_data['indexes'][table_name] = []
        schema_data['indexes'][table_name].append(dict(row))

    cursor.close()
    return schema_data

def build_json_schema(data):
    """Constructs the final JSON schema from fetched data."""
    output_schema = {}
    for table_name, table_info in data['tables'].items():
        output_schema[table_name] = {
            'comment': table_info.get('comment'),
            'columns': [],
            'constraints': [],
            'indexes': []
        }

        # Add columns
        if table_name in data['columns']:
            for col in data['columns'][table_name]:
                col_data = {
                    'name': col['column_name'],
                    'type': col['data_type'],
                    'nullable': col['is_nullable'] == 'YES',
                    'default': col['column_default'],
                    'primary_key': False # Will be updated by constraint info
                }
                if col['character_maximum_length'] is not None:
                    col_data['length'] = col['character_maximum_length']
                if col['numeric_precision'] is not None:
                    col_data['precision'] = col['numeric_precision']
                if col['numeric_scale'] is not None:
                    col_data['scale'] = col['numeric_scale']
                if col['datetime_precision'] is not None:
                    col_data['datetime_precision'] = col['datetime_precision']
                output_schema[table_name]['columns'].append(col_data)

        # Add constraints
        if table_name in data['constraints']:
            for name, constraint in data['constraints'][table_name].items():
                constraint_data = {
                    'name': name,
                    'type': constraint['type'],
                    'columns': sorted(constraint['columns']) # Ensure consistent order
                }
                if constraint['type'] == 'PRIMARY KEY':
                    # Mark corresponding columns as primary keys
                    for col in output_schema[table_name]['columns']:
                        if col['name'] in constraint['columns']:
                            col['primary_key'] = True
                elif constraint['type'] == 'FOREIGN KEY':
                    constraint_data['referred_table'] = constraint['foreign_table_name']
                    # Assuming FK constraints link single columns in this simple case
                    constraint_data['referred_columns'] = [constraint['foreign_column_name']]
                    constraint_data['on_update'] = constraint['on_update']
                    constraint_data['on_delete'] = constraint['on_delete']
                elif constraint['type'] == 'CHECK':
                     constraint_data['check_clause'] = constraint['check_clause']

                output_schema[table_name]['constraints'].append(constraint_data)

        # Add indexes
        if table_name in data['indexes']:
            for index in data['indexes'][table_name]:
                # Try to parse columns from definition (simple case)
                # This is a basic parser, might need refinement for complex index defs
                columns = []
                try:
                    col_part = index['index_definition'].split('ON')[1].split('USING')[0]
                    col_part = col_part[col_part.find('(')+1:col_part.rfind(')')]
                    columns = [c.strip().strip('"') for c in col_part.split(',')]
                except Exception:
                    print(f"Warning: Could not parse columns for index {index['index_name']} from definition: {index['index_definition']}")

                output_schema[table_name]['indexes'].append({
                    'name': index['index_name'],
                    'unique': index['is_unique'],
                    'method': index['index_method'],
                    'columns': columns, # Parsed columns
                    'definition': index['index_definition'] # Include full definition
                })

    return json.dumps(output_schema, indent=2)

if __name__ == "__main__":
    print("Attempting to connect to database specified in .env...")
    db_params = get_db_params()
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        print("Database connection successful.")
        raw_data = fetch_schema_data(conn)
        print("Schema data fetched. Generating JSON...")
        json_output = build_json_schema(raw_data)
        print("\n--- DATABASE SCHEMA (JSON) ---")
        print(json_output)
        print("--- END SCHEMA ---")

        # Save to file
        output_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'actual_schema.json')
        with open(output_filename, 'w') as f:
            f.write(json_output)
        print(f"\nSchema successfully saved to: {output_filename}")

    except psycopg2.OperationalError as e:
        print(f"\nDatabase connection failed: {e}")
        print("Please ensure the database server is running and accessible,")
        print("and that the connection details in .env are correct.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.") 