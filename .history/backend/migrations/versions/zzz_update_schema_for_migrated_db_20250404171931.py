"""Update schema to match migrated database

Revision ID: zzz_update_schema_for_migrated_db
Revises: zzz_add_updated_at_to_definition_relations
Create Date: 2025-04-01 07:15:00.000000

# IMPORTANT NAMING CONFLICT NOTE:
# 'metadata' is a reserved attribute name in SQLAlchemy's declarative models.
# Any model that needs a 'metadata' column in the database should:
# 1. Use a different attribute name in the model (e.g., 'extra_data' or 'pronunciation_metadata')
# 2. Map that attribute to the 'metadata' column using: db.Column('metadata', db.JSON)
# 3. If needed, use @property methods to expose it as 'metadata' in the API
#
# Example:
#     # In model:
#     extra_data = db.Column('metadata', db.JSON)  # Maps attribute to database column
#     
#     # If needed, create property to access as 'metadata'
#     @property
#     def metadata(self):
#         return self.extra_data
#     
#     @metadata.setter
#     def metadata(self, value):
#         self.extra_data = value
#         
#     # In to_dict method:
#     def to_dict(self):
#         return {
#             ...
#             'metadata': self.extra_data,  # Return as 'metadata' in API
#             ...
#         }

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision = 'zzz_update_schema_for_migrated_db'
down_revision = 'zzz_add_updated_at_to_definition_relations'
branch_labels = None
depends_on = None


def upgrade():
    # Update JSON fields to use JSONB if they aren't already
    op.execute("""
        DO $$
        BEGIN
            -- Check if column already exists in words table
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'word_metadata'
            ) THEN
                ALTER TABLE words ADD COLUMN word_metadata JSONB DEFAULT '{}'::jsonb;
            END IF;

            -- Change pronunciations.pronunciation_metadata to metadata if needed
            IF EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'pronunciations' AND column_name = 'pronunciation_metadata'
            ) AND NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'pronunciations' AND column_name = 'metadata'
            ) THEN
                -- Rename the column in the database, but use pronunciation_metadata as attribute in models
                -- to avoid conflicts with SQLAlchemy's reserved 'metadata' attribute
                ALTER TABLE pronunciations RENAME COLUMN pronunciation_metadata TO metadata;
            END IF;

            -- Add metadata to relations if needed
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'relations' AND column_name = 'metadata'
            ) THEN
                ALTER TABLE relations ADD COLUMN metadata JSONB DEFAULT '{}'::jsonb;
            END IF;

            -- Ensure proper constraints on baybayin fields
            IF EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'baybayin_form'
            ) THEN
                -- Add constraint if not exists
                IF NOT EXISTS (
                    SELECT FROM pg_constraint
                    WHERE conname = 'baybayin_form_check'
                ) THEN
                    ALTER TABLE words ADD CONSTRAINT baybayin_form_check 
                    CHECK ((has_baybayin = false AND baybayin_form IS NULL) 
                        OR (has_baybayin = true AND baybayin_form IS NOT NULL));
                END IF;

                IF NOT EXISTS (
                    SELECT FROM pg_constraint
                    WHERE conname = 'baybayin_form_regex'
                ) THEN
                    ALTER TABLE words ADD CONSTRAINT baybayin_form_regex
                    CHECK (baybayin_form ~ '^[\u1700-\u171F[:space:]]*$' OR baybayin_form IS NULL);
                END IF;
            END IF;

            -- Add standard parts of speech if table exists
            IF EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'parts_of_speech'
            ) THEN
                -- Ensure 'lig' code exists
                IF NOT EXISTS (
                    SELECT FROM parts_of_speech WHERE code = 'lig'
                ) THEN
                    INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                    VALUES ('lig', 'Ligature', 'Pang-angkop', 'Word that links modifiers to modified words');
                END IF;

                -- Ensure 'part' code exists
                IF NOT EXISTS (
                    SELECT FROM parts_of_speech WHERE code = 'part'
                ) THEN
                    INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                    VALUES ('part', 'Particle', 'Kataga', 'Function word that doesn''t fit other categories');
                END IF;

                -- Ensure 'num' code exists
                IF NOT EXISTS (
                    SELECT FROM parts_of_speech WHERE code = 'num'
                ) THEN
                    INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                    VALUES ('num', 'Number', 'Pamilang', 'Word representing a number');
                END IF;

                -- Ensure 'expr' code exists
                IF NOT EXISTS (
                    SELECT FROM parts_of_speech WHERE code = 'expr'
                ) THEN
                    INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                    VALUES ('expr', 'Expression', 'Pahayag', 'Common phrase or expression');
                END IF;

                -- Ensure 'punc' code exists
                IF NOT EXISTS (
                    SELECT FROM parts_of_speech WHERE code = 'punc'
                ) THEN
                    INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                    VALUES ('punc', 'Punctuation', 'Bantas', 'Punctuation mark');
                END IF;
            END IF;

        END $$;
    """)

    # Create required indexes if they don't exist
    op.execute("""
        DO $$
        BEGIN
            -- Create words_lemma_idx if not exists
            IF NOT EXISTS (
                SELECT FROM pg_indexes WHERE indexname = 'words_lemma_idx'
            ) THEN
                CREATE INDEX IF NOT EXISTS words_lemma_idx ON words(lemma);
            END IF;

            -- Create words_normalized_idx if not exists
            IF NOT EXISTS (
                SELECT FROM pg_indexes WHERE indexname = 'words_normalized_idx'
            ) THEN
                CREATE INDEX IF NOT EXISTS words_normalized_idx ON words(normalized_lemma);
            END IF;

            -- Create words_baybayin_idx if not exists
            IF NOT EXISTS (
                SELECT FROM pg_indexes WHERE indexname = 'words_baybayin_idx'
            ) THEN
                CREATE INDEX IF NOT EXISTS words_baybayin_idx ON words(baybayin_form) WHERE has_baybayin = TRUE;
            END IF;

            -- Create idx_words_romanized if not exists
            IF NOT EXISTS (
                SELECT FROM pg_indexes WHERE indexname = 'idx_words_romanized'
            ) THEN
                CREATE INDEX IF NOT EXISTS idx_words_romanized ON words(romanized_form);
            END IF;
        END $$;
    """)


def downgrade():
    # No downgrade needed as this is a compatibility update
    pass 