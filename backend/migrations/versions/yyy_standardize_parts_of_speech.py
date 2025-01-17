"""Standardize parts of speech in the database

Revision ID: yyy
Revises: xxx
Create Date: 2024-01-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# Comprehensive mapping of parts of speech with bilingual labels
POS_MAPPING = {
    # Base forms
    'noun': {'en': 'Noun', 'tl': 'Pangngalan'},
    'adjective': {'en': 'Adjective', 'tl': 'Pang-uri'},
    'verb': {'en': 'Verb', 'tl': 'Pandiwa'},
    'adverb': {'en': 'Adverb', 'tl': 'Pang-abay'},
    'pronoun': {'en': 'Pronoun', 'tl': 'Panghalip'},
    'preposition': {'en': 'Preposition', 'tl': 'Pang-ukol'},
    'conjunction': {'en': 'Conjunction', 'tl': 'Pangatnig'},
    'interjection': {'en': 'Interjection', 'tl': 'Pandamdam'},
    'affix': {'en': 'Affix', 'tl': 'Panlapi'},
    
    # Tagalog forms
    'Pangngalan': {'en': 'Noun', 'tl': 'Pangngalan'},
    'Pang-uri': {'en': 'Adjective', 'tl': 'Pang-uri'},
    'Pandiwa': {'en': 'Verb', 'tl': 'Pandiwa'},
    'Pang-abay': {'en': 'Adverb', 'tl': 'Pang-abay'},
    'Panghalip': {'en': 'Pronoun', 'tl': 'Panghalip'},
    'Pang-ukol': {'en': 'Preposition', 'tl': 'Pang-ukol'},
    'Pangatnig': {'en': 'Conjunction', 'tl': 'Pangatnig'},
    'Pandamdam': {'en': 'Interjection', 'tl': 'Pandamdam'},
    'Panlapi': {'en': 'Affix', 'tl': 'Panlapi'},
    
    # Abbreviations
    'pnd': {'en': 'Noun', 'tl': 'Pangngalan'},
    'png': {'en': 'Noun', 'tl': 'Pangngalan'},
    'pnr': {'en': 'Pronoun', 'tl': 'Panghalip'},
    'pnl': {'en': 'Affix', 'tl': 'Panlapi'},
    'pnu': {'en': 'Adjective', 'tl': 'Pang-uri'},
    'pnw': {'en': 'Verb', 'tl': 'Pandiwa'},
    'pny': {'en': 'Adverb', 'tl': 'Pang-abay'},
    
    # Special cases
    'Baybayin': {'en': 'Baybayin Script', 'tl': 'Baybayin'},
    '': {'en': 'Uncategorized', 'tl': 'Hindi Tiyak'}
}

def upgrade():
    # First, add new JSON column for bilingual POS
    op.add_column('definitions', sa.Column('pos_data', JSONB))
    
    conn = op.get_bind()
    
    # Clean up existing data
    conn.execute(sa.text("""
        UPDATE definitions
        SET part_of_speech = TRIM(BOTH ',' FROM TRIM(part_of_speech))
        WHERE part_of_speech LIKE ',%' OR part_of_speech LIKE '%,' 
           OR part_of_speech LIKE ' %' OR part_of_speech LIKE '% '
    """))
    
    # Update pos_data based on existing part_of_speech
    for old_pos, bilingual in POS_MAPPING.items():
        if old_pos:  # Skip empty string to avoid matching everything
            conn.execute(
                sa.text("""
                    UPDATE definitions
                    SET pos_data = :bilingual::jsonb
                    WHERE LOWER(part_of_speech) = LOWER(:old_pos)
                """),
                {"bilingual": bilingual, "old_pos": old_pos}
            )
    
    # Handle multiple parts of speech
    conn.execute(sa.text("""
        CREATE OR REPLACE FUNCTION standardize_pos(pos text) RETURNS jsonb AS $$
        DECLARE
            pos_parts text[];
            result jsonb;
            part text;
        BEGIN
            pos_parts := string_to_array(pos, ',');
            result := '{"en": [], "tl": []}'::jsonb;
            
            FOR part IN SELECT unnest(pos_parts) LOOP
                part := TRIM(part);
                IF part <> '' THEN
                    -- Logic to map part to bilingual form would go here
                    -- This is a simplified version
                    result := jsonb_set(
                        result,
                        '{en}',
                        (result->'en') || to_jsonb(COALESCE(part, 'Uncategorized'))
                    );
                END IF;
            END LOOP;
            
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
        
        UPDATE definitions
        SET pos_data = standardize_pos(part_of_speech)
        WHERE part_of_speech LIKE '%,%';
    """))

def downgrade():
    conn = op.get_bind()
    conn.execute(sa.text("DROP FUNCTION IF EXISTS standardize_pos(text)"))
    op.drop_column('definitions', 'pos_data') 