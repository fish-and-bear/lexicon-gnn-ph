"""Add language metadata tables and standardization

Revision ID: xxx
Revises: previous_revision
Create Date: 2024-01-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from language_systems import LanguageSystem
import json

# Initialize language system
language_system = LanguageSystem()

def upgrade():
    # Create language metadata table
    op.create_table(
        'language_metadata',
        sa.Column('language_code', sa.String(), primary_key=True),
        sa.Column('family_tree', ARRAY(sa.String())),
        sa.Column('writing_systems', JSONB),
        sa.Column('regions', ARRAY(sa.String())),
        sa.Column('standardized_name', sa.String(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'))
    )
    
    # Create updated_at trigger
    op.execute("""
        CREATE OR REPLACE FUNCTION update_language_metadata_timestamp()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        CREATE TRIGGER update_language_metadata_timestamp
        BEFORE UPDATE ON language_metadata
        FOR EACH ROW
        EXECUTE FUNCTION update_language_metadata_timestamp();
    """)
    
    # Create auto-population trigger
    op.execute("""
        CREATE OR REPLACE FUNCTION update_language_metadata() 
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.language_code IS NOT NULL THEN
                INSERT INTO language_metadata (
                    language_code,
                    standardized_name,
                    family_tree,
                    writing_systems,
                    regions
                )
                VALUES (
                    NEW.language_code,
                    NEW.language_code,
                    ARRAY[]::text[],
                    '{}'::jsonb,
                    ARRAY[]::text[]
                )
                ON CONFLICT (language_code) DO NOTHING;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        CREATE TRIGGER words_language_metadata_trigger
        AFTER INSERT ON words
        FOR EACH ROW
        EXECUTE FUNCTION update_language_metadata();
    """)
    
    # Create indexes
    op.create_index('idx_language_metadata_family_tree', 'language_metadata', ['family_tree'], postgresql_using='gin')
    op.create_index('idx_language_metadata_regions', 'language_metadata', ['regions'], postgresql_using='gin')

def downgrade():
    # Drop in reverse order
    op.execute("DROP TRIGGER IF EXISTS update_language_metadata_timestamp ON language_metadata")
    op.execute("DROP FUNCTION IF EXISTS update_language_metadata_timestamp()")
    op.execute("DROP TRIGGER IF EXISTS words_language_metadata_trigger ON words")
    op.execute("DROP FUNCTION IF EXISTS update_language_metadata()")
    op.drop_index('idx_language_metadata_family_tree')
    op.drop_index('idx_language_metadata_regions')
    op.drop_table('language_metadata') 