import os
import psycopg2
import sqlite3
from dotenv import load_dotenv
import json
import datetime
import uuid as uuid_module
import re

def get_pg_connection():
    """Establishes connection to PostgreSQL database."""
    load_dotenv()
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    return conn

def get_table_schema(pg_cursor, table_name):
    """Fetches schema for a given table from PostgreSQL."""
    pg_cursor.execute(f"""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position;
    """)
    return pg_cursor.fetchall()

def create_sqlite_table(sqlite_cursor, table_name, schema):
    """Creates a table in SQLite based on PostgreSQL schema."""
    columns_ddl = []
    for col_name, data_type, is_nullable, col_default in schema:
        sqlite_type = map_pg_to_sqlite_type(data_type)
        ddl_part = f'"{col_name}" {sqlite_type}'
        if is_nullable == 'NO':
            ddl_part += " NOT NULL"
        
        if col_default:
            if "nextval" in col_default and ("integer" in sqlite_type.lower() or "int" in sqlite_type.lower()):
                if not columns_ddl:
                    ddl_part = f'"{col_name}" INTEGER PRIMARY KEY AUTOINCREMENT'
                else:
                    pass
            
            elif sqlite_type == "TEXT":
                default_text_val = str(col_default)
                m = re.match(r"^'(.*)'::[\w\s\[\]\".-]+$", default_text_val, re.DOTALL)
                if m:
                    extracted_literal = m.group(1)
                    ddl_part += f" DEFAULT '{extracted_literal}'"
                elif default_text_val.upper() == 'CURRENT_TIMESTAMP':
                    ddl_part += " DEFAULT CURRENT_TIMESTAMP"
                elif (default_text_val.startswith("'{}'::") or default_text_val.startswith("'[]'::")) and "[]" in default_text_val :
                     first_quote = default_text_val.find("'")
                     second_quote = default_text_val.find("'", first_quote + 1)
                     if first_quote != -1 and second_quote != -1:
                        extracted_array_literal = default_text_val[first_quote+1:second_quote]
                        ddl_part += f" DEFAULT '{extracted_array_literal.replace("'", "''")}'"
                     else:
                        ddl_part += f" DEFAULT '{default_text_val.replace("'", "''")}'"
                else:
                    ddl_part += f" DEFAULT '{default_text_val.replace("'", "''")}'"
            
            elif sqlite_type in ["INTEGER", "REAL"] and "numeric" not in data_type.lower():
                try:
                    float(col_default)
                    ddl_part += f" DEFAULT {col_default}"
                except ValueError:
                    if col_default.lower() == 'false':
                        ddl_part += " DEFAULT 0"
                    elif col_default.lower() == 'true':
                        ddl_part += " DEFAULT 1"
                    else:
                        ddl_part += f" DEFAULT '{str(col_default).replace("'", "''")}'"

            elif "boolean" in data_type.lower():
                if col_default.lower() == 'true':
                    ddl_part += " DEFAULT 1"
                elif col_default.lower() == 'false':
                    ddl_part += " DEFAULT 0"
                else:
                    ddl_part += f" DEFAULT {col_default}"
            
        columns_ddl.append(ddl_part)

    create_table_sql = f"CREATE TABLE IF NOT EXISTS \"{table_name}\" ({', '.join(columns_ddl)})"
    print(f"Creating table \"{table_name}\" with SQL: {create_table_sql}")
    sqlite_cursor.execute(create_table_sql)

def map_pg_to_sqlite_type(pg_type):
    """Maps PostgreSQL data types to SQLite data types."""
    pg_type_lower = pg_type.lower()
    if "int" in pg_type_lower or "serial" in pg_type_lower:
        return "INTEGER"
    elif "char" in pg_type_lower or "text" in pg_type_lower or pg_type_lower == "name":
        return "TEXT"
    elif "uuid" == pg_type_lower:
        return "TEXT" 
    elif "numeric" in pg_type_lower or "decimal" in pg_type_lower or "double precision" in pg_type_lower or "real" in pg_type_lower:
        return "REAL"
    elif "timestamp" in pg_type_lower or "date" == pg_type_lower or "time" == pg_type_lower:
        return "TEXT"
    elif "boolean" in pg_type_lower:
        return "INTEGER"
    elif "bytea" in pg_type_lower:
        return "BLOB"
    elif "json" in pg_type_lower:
        return "TEXT"
    elif "array" in pg_type_lower or pg_type_lower.startswith('_'):
        return "TEXT"
    elif "tsvector" in pg_type_lower:
        return "TEXT"
        
    print(f"Warning: Unmapped PostgreSQL type: {pg_type}. Defaulting to TEXT.")
    return "TEXT"

def copy_data(pg_cursor, sqlite_conn, sqlite_cursor, table_name, schema):
    """Copies data from a PostgreSQL table to an SQLite table, handling type conversions."""
    pg_cursor.execute(f'SELECT * FROM "{table_name}"')
    rows = pg_cursor.fetchall()

    if not rows:
        print(f"No data in table \"{table_name}\" or table is empty.")
        return

    num_columns = len(schema)
    column_pg_types = [s[1] for s in schema]
    column_names = [f'"{s[0]}"' for s in schema]

    placeholders = ', '.join(['?'] * num_columns)
    insert_sql = f"INSERT INTO \"{table_name}\" ({', '.join(column_names)}) VALUES ({placeholders})"


    processed_rows = []
    for row_idx, row_tuple in enumerate(rows):
        processed_row_list = []
        for i, value in enumerate(row_tuple):
            if value is None:
                processed_row_list.append(None)
                continue

            pg_type = column_pg_types[i].lower()
            col_name = schema[i][0]

            try:
                if 'json' in pg_type or pg_type.startswith('_'):
                    if isinstance(value, (dict, list, str, int, float, bool)):
                        processed_row_list.append(json.dumps(value))
                    else:
                        print(f"Warning: Column '{col_name}' in table '{table_name}' (row {row_idx+1}) has pg_type {pg_type} and Python type {type(value)}. Attempting json.dumps().")
                        processed_row_list.append(json.dumps(str(value)))
                elif pg_type == 'uuid':
                    if isinstance(value, uuid_module.UUID):
                        processed_row_list.append(str(value))
                    elif isinstance(value, str):
                         processed_row_list.append(value)
                    else:
                        print(f"Warning: Column '{col_name}' in table '{table_name}' (row {row_idx+1}) has pg_type uuid but Python type {type(value)}. Attempting str().")
                        processed_row_list.append(str(value))
                elif pg_type == 'bytea':
                    if isinstance(value, memoryview):
                        processed_row_list.append(value.tobytes())
                    elif isinstance(value, bytes):
                        processed_row_list.append(value)
                    elif isinstance(value, str) and value.startswith('\\\\x'):
                         processed_row_list.append(bytes.fromhex(value[2:]))
                    else:
                        print(f"Warning: Column '{col_name}' in table '{table_name}' (row {row_idx+1}) has pg_type bytea but Python type {type(value)}. Attempting encode('utf-8').")
                        processed_row_list.append(str(value).encode('utf-8'))
                elif 'timestamp' in pg_type or 'date' == pg_type or 'time' == pg_type:
                    if hasattr(value, 'isoformat'):
                        processed_row_list.append(value.isoformat())
                    elif isinstance(value, str):
                        processed_row_list.append(value)
                    else:
                        print(f"Warning: Column '{col_name}' in table '{table_name}' (row {row_idx+1}) has pg_type datetime but Python type {type(value)}. Attempting str().")
                        processed_row_list.append(str(value))
                elif pg_type == 'tsvector':
                    processed_row_list.append(str(value))
                else:
                    processed_row_list.append(value)
            except Exception as e:
                print(f"Error processing value for column '{col_name}' (pg_type: {pg_type}, py_type: {type(value)}, value: '{str(value)[:100]}...') in table '{table_name}' row {row_idx+1}: {e}")
                raise
        processed_rows.append(tuple(processed_row_list))

    print(f"Inserting data into \"{table_name}\" ({len(processed_rows)} rows)...")
    try:
        sqlite_cursor.executemany(insert_sql, processed_rows)
        sqlite_conn.commit()
        print(f"Data copied for table \"{table_name}\".")
    except sqlite3.Error as e:
        print(f"SQLite Error during data insertion for table \"{table_name}\": {e}")
        if processed_rows:
            for row_num, p_row in enumerate(processed_rows):
                try:
                    sqlite_cursor.execute(insert_sql, p_row)
                except sqlite3.Error as single_row_e:
                    print(f"Error on row {row_num} for table \"{table_name}\". SQLite error: {single_row_e}")
                    print(f"Problematic row data (Python types and first 100 chars of str value):")
                    for i, (val, pg_t, col_s_name) in enumerate(zip(p_row, column_pg_types, [s[0] for s in schema])):
                        print(f"  Col {i} ('{col_s_name}', pg_type: {pg_t}): {type(val)} - '{str(val)[:100]}'")
                    break
        sqlite_conn.rollback()
        raise

def main():
    sqlite_db_file = 'fil_relex_colab.sqlite'
    if os.path.exists(sqlite_db_file):
        try:
            os.remove(sqlite_db_file)
            print(f"Removed existing SQLite file: {sqlite_db_file}")
        except OSError as e:
            print(f"Error removing existing SQLite file {sqlite_db_file}: {e}. Please check permissions or close applications using it.")
            return


    pg_conn = None
    sqlite_conn = None

    try:
        pg_conn = get_pg_connection()
        pg_cursor = pg_conn.cursor()

        sqlite_conn = sqlite3.connect(sqlite_db_file)
        sqlite_cursor = sqlite_conn.cursor()

        pg_cursor.execute("""
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname != 'pg_catalog' AND 
                  schemaname != 'information_schema' AND
                  tablename NOT LIKE 'pg_stat_%' AND 
                  tablename NOT LIKE 'sql_%'; 
        """)
        tables = [row[0] for row in pg_cursor.fetchall()]

        print(f"Found user tables: {tables}")

        for table_name in tables:
            print(f"\nProcessing table: \"{table_name}\"")
            schema = get_table_schema(pg_cursor, table_name)
            if not schema:
                print(f"Could not retrieve schema for table \"{table_name}\". Skipping.")
                continue

            create_sqlite_table(sqlite_cursor, table_name, schema)
            copy_data(pg_cursor, sqlite_conn, sqlite_cursor, table_name, schema)

        print(f"\nDatabase export to {sqlite_db_file} complete.")

    except psycopg2.Error as e:
        print(f"PostgreSQL Connection/Query Error: {e}")
    except sqlite3.Error as e:
        print(f"SQLite Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pg_conn:
            pg_conn.close()
        if sqlite_conn:
            sqlite_conn.close()

if __name__ == '__main__':
    main()