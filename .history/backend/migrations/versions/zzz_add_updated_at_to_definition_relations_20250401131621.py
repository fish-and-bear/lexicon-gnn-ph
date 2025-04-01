"""Add updated_at column to definition_relations table

Revision ID: zzz
Revises: yyy
Create Date: 2025-04-01 13:15:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = 'zzz'
down_revision = 'yyy'
branch_labels = None
depends_on = None

def upgrade():
    # Add updated_at column to definition_relations table
    op.add_column('definition_relations', sa.Column('updated_at', sa.DateTime(), nullable=True))
    
    # Set updated_at to created_at for existing records
    op.execute("""
        UPDATE definition_relations 
        SET updated_at = created_at 
        WHERE updated_at IS NULL
    """)
    
    # Make updated_at not nullable after setting values
    op.alter_column('definition_relations', 'updated_at',
                    existing_type=sa.DateTime(),
                    nullable=False)

def downgrade():
    # Remove updated_at column from definition_relations table
    op.drop_column('definition_relations', 'updated_at') 