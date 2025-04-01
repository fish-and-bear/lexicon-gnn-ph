"""add updated_at to definition_relations

Revision ID: zzz_add_updated_at_to_definition_relations
Revises: 
Create Date: 2025-04-01 13:40:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = 'zzz_add_updated_at_to_definition_relations'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Add updated_at column to definition_relations table with server_default
    op.add_column('definition_relations', 
                 sa.Column('updated_at', 
                          sa.DateTime(), 
                          server_default=sa.text('CURRENT_TIMESTAMP'),
                          nullable=False))
    
    # Create trigger to automatically update updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        CREATE TRIGGER update_definition_relations_updated_at
            BEFORE UPDATE ON definition_relations
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)

def downgrade():
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS update_definition_relations_updated_at ON definition_relations")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column")
    
    # Remove updated_at column from definition_relations table
    op.drop_column('definition_relations', 'updated_at') 