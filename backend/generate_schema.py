import json
from sqlalchemy import inspect, MetaData
from sqlalchemy.dialects import postgresql
from flask import Flask

# Assume your Flask app creation logic is in app.py or similar
# and it initializes the db object and imports models.
try:
    # Adjust this import based on your actual app structure
    from app import create_app, db
except ImportError as e:
    print(f"Error importing Flask app or db object: {e}")
    print("Please ensure this script is run from the 'backend' directory")
    print("and that your main Flask app file (e.g., app.py) correctly")
    print("initializes the SQLAlchemy 'db' object and imports models.")
    exit(1)

def get_column_info(column):
    """Extracts relevant information from a SQLAlchemy Column object."""
    col_info = {
        'name': column.name,
        'type': str(column.type),
        'nullable': column.nullable,
        'default': str(column.default) if column.default else None,
        'primary_key': column.primary_key,
    }
    # Add specific type attributes if useful (e.g., length for VARCHAR)
    if hasattr(column.type, 'length'):
        col_info['length'] = column.type.length
    return col_info

def get_constraint_info(constraint):
    """Extracts relevant information from a SQLAlchemy Constraint object."""
    constraint_type = type(constraint).__name__
    info = {'type': constraint_type, 'name': constraint.name}

    if constraint_type == 'ForeignKeyConstraint':
        info['columns'] = [col.name for col in constraint.columns]
        info['referred_table'] = constraint.elements[0].target_fullname.split('.')[0]
        info['referred_columns'] = [el.target_fullname.split('.')[1] for el in constraint.elements]
        info['onupdate'] = constraint.onupdate
        info['ondelete'] = constraint.ondelete
    elif constraint_type == 'PrimaryKeyConstraint':
        info['columns'] = [col.name for col in constraint.columns]
    elif constraint_type == 'UniqueConstraint':
        info['columns'] = [col.name for col in constraint.columns]
    elif constraint_type == 'CheckConstraint':
        info['sqltext'] = str(constraint.sqltext)
    # Add more constraint types if needed (e.g., Index)

    return info

def get_index_info(index):
    """Extracts relevant information from a SQLAlchemy Index object."""
    return {
        'name': index.name,
        'columns': [col.name for col in index.columns],
        'unique': index.unique,
        # Add postgresql_using if it's a PG index
        'using': getattr(index, 'dialect_options', {}).get('postgresql', {}).get('using'),
        # Add expression info if it's an expression-based index
        'expressions': [str(expr) for expr in index.expressions] if hasattr(index, 'expressions') else []
    }


def generate_schema_json(app: Flask, database):
    """Generates a JSON representation of the database schema."""
    schema_info = {}
    with app.app_context():
        metadata = database.metadata
        inspector = inspect(database.engine)

        table_names = sorted(metadata.tables.keys())

        for table_name in table_names:
            table = metadata.tables[table_name]
            table_info = {
                'columns': [],
                'constraints': [],
                'indexes': [],
                'comment': table.comment
            }

            # Get Columns
            table_info['columns'] = [get_column_info(column) for column in table.columns]

            # Get Constraints (FK, PK, Unique, Check) from metadata
            table_info['constraints'] = [get_constraint_info(constraint) for constraint in table.constraints]

            # Get Indexes using inspector (more reliable for different index types)
            try:
                indexes = inspector.get_indexes(table_name)
                table_info['indexes'] = [get_index_info(index) for index in indexes]
            except Exception as e:
                print(f"Warning: Could not inspect indexes for table {table_name}: {e}")


            schema_info[table_name] = table_info

    return json.dumps(schema_info, indent=2)

if __name__ == "__main__":
    flask_app = create_app()
    schema_json = generate_schema_json(flask_app, db)
    print(schema_json) 