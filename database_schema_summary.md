# Filipino Dictionary Database Schema

This document provides a detailed summary of the database schema used in the Filipino dictionary application, derived from the `actual_schema.json` file. The information herein reflects the state of the database metadata as extracted by the `backend/generate_actual_schema.py` script, including basic population statistics and details confirmed by `backend/dictionary_manager/db_helpers.py`.

**Note on Foreign Keys:** The `generate_actual_schema.py` script primarily uses `information_schema.table_constraints` to identify foreign keys. Some tables (`definition_categories`, `definition_links`, `word_forms`, `word_templates`) have columns defined with `REFERENCES words(id) ON DELETE CASCADE` or `REFERENCES definitions(id) ON DELETE CASCADE` in their `CREATE TABLE` statements (see `db_helpers.py`), establishing relationships and cascade behavior. However, these might not be reported as formal `FOREIGN KEY` constraints by the script if they were created without the explicit `CONSTRAINT ... FOREIGN KEY` syntax. Application logic in `db_helpers.py` appears to assume these relationships by requiring valid `word_id` or `definition_id` during insertion.

**Note on Indexes:** The `actual_schema.json` lists all index objects found in the database (`pg_indexes`). This includes indexes created explicitly (`CREATE INDEX`) and those implicitly created by Primary Key (`PK`) and Unique constraints. This summary consolidates the index information, describing the functional indexes present. Constraint-related indexes are noted alongside the relevant constraint.

## Core Tables

### 1. `languages`
Stores information about languages supported in the dictionary.
- **Row Count:** 0

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `code` (varchar(10), not null, unique) - *Distinct Count: 0*
- `name_en` (varchar(100), nullable)
- `name_tl` (varchar(100), nullable)
- `region` (varchar(100), nullable) - *Distinct Count: 0*
- `family` (varchar(100), nullable) - *Distinct Count: 0*
- `status` (varchar(50), nullable) - *Distinct Count: 0*
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `languages_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `languages_pkey`*
- `languages_code_key`: UNIQUE (`code`) - *Implicitly creates unique index `languages_code_key`*

**Indexes (Explicitly Created):**
- `idx_languages_code`: btree (`code`) - *Note: Covered by UNIQUE constraint `languages_code_key`.*

### 2. `words`
Contains the main lexical entries in the dictionary.
- **Row Count:** 167,315

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `lemma` (varchar(255), not null)
- `normalized_lemma` (varchar(255), not null)
- `has_baybayin` (boolean, nullable, default: false)
- `baybayin_form` (varchar(255), nullable)
- `romanized_form` (varchar(255), nullable)
- `language_code` (varchar(16), not null) - *Distinct Count: 120, Examples: ['aby', 'agt', 'agu', 'akl', 'am', 'apa', 'ar', 'ata', 'bag', 'baj']*
- `root_word_id` (integer, nullable) - *References `words.id` (ON DELETE SET NULL)*
- `preferred_spelling` (varchar(255), nullable)
- `tags` (text, nullable)
- `idioms` (jsonb, nullable, default: '[]'::jsonb)
- `pronunciation_data` (jsonb, nullable)
- `source_info` (jsonb, nullable, default: '{}'::jsonb)
- `word_metadata` (jsonb, nullable, default: '{}'::jsonb)
- `data_hash` (text, nullable)
- `badlit_form` (text, nullable)
- `hyphenation` (jsonb, nullable)
- `is_proper_noun` (boolean, nullable, default: false) - *Distinct Count: 2, Values: [false, true]*
- `is_abbreviation` (boolean, nullable, default: false) - *Distinct Count: 1, Values: [false]*
- `is_initialism` (boolean, nullable, default: false) - *Distinct Count: 1, Values: [false]*
- `search_text` (tsvector, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `words_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `words_pkey`*
- `words_lang_lemma_uniq`: UNIQUE (`normalized_lemma`, `language_code`) - *Implicitly creates unique index `words_lang_lemma_uniq`*
- `words_root_word_id_fkey`: FOREIGN KEY (`root_word_id`) REFERENCES `words` (`id`) ON DELETE SET NULL

**Indexes (Explicitly Created):**
- `idx_words_baybayin`: btree (`baybayin_form`) WHERE has_baybayin = true
- `idx_words_language`: btree (`language_code`)
- `idx_words_lemma`: btree (`lemma`)
- `idx_words_metadata`: gin (`word_metadata`)
- `idx_words_normalized`: btree (`normalized_lemma`) - *Note: Covered by UNIQUE constraint `words_lang_lemma_uniq`.*
- `idx_words_romanized`: btree (`romanized_form`)
- `idx_words_root`: btree (`root_word_id`)
- `idx_words_search`: gin (`search_text`)

### 3. `definitions`
Stores definitions for words.
- **Row Count:** 235,180

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `word_id` (integer, not null) - *References `words.id` (ON DELETE CASCADE)*
- `definition_text` (text, not null)
- `original_pos` (text, nullable) - *Distinct Count: 20, Values: ['abbr', 'adj', 'adv', 'affix', 'conj', 'det', 'interj', 'noun', 'num', 'part', 'phrase', 'prefix', 'prep', 'pron', 'propn', 'punct', 'suffix', 'sym', 'unc', 'verb', null]*
- `standardized_pos_id` (integer, nullable) - *References `parts_of_speech.id` (ON DELETE SET NULL)*
- `examples` (text, nullable)
- `usage_notes` (text, nullable)
- `tags` (text, nullable)
- `sources` (text, nullable)
- `definition_metadata` (jsonb, nullable, default: '{}'::jsonb)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `definitions_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `definitions_pkey`*
- `definitions_unique`: UNIQUE (`word_id`, `definition_text`, `standardized_pos_id`) - *Implicitly creates unique index `definitions_unique`*
- `definitions_standardized_pos_id_fkey`: FOREIGN KEY (`standardized_pos_id`) REFERENCES `parts_of_speech` (`id`) ON DELETE SET NULL
- `definitions_word_id_fkey`: FOREIGN KEY (`word_id`) REFERENCES `words` (`id`) ON DELETE CASCADE

**Indexes (Explicitly Created):**
- `idx_definitions_pos`: btree (`standardized_pos_id`)
- `idx_definitions_text`: gin (`to_tsvector('english'::regconfig, definition_text)`)
- `idx_definitions_word`: btree (`word_id`) - *Note: Covered by UNIQUE constraint `definitions_unique`.*

### 4. `parts_of_speech`
Stores standardized parts of speech information.
- **Row Count:** 25

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `code` (varchar(32), not null, unique) - *Distinct Count: 25, Values: ['abbr', 'adj', 'adv', 'affix', 'char', 'class', 'conj', 'contraction', 'det', 'interj', 'noun', 'num', 'part', 'phrase', 'prefix', 'prep', 'pron', 'propn', 'punct', 'rom', 'root', 'suffix', 'sym', 'unc', 'verb']*
- `name_en` (varchar(64), not null) - *Distinct Count: 25, Values: ['Abbreviation', 'Adjective', 'Adverb', 'Affix', 'Character', 'Classifier', 'Conjunction', 'Contraction', 'Determiner', 'Interjection', 'Noun', 'Numeral', 'Particle', 'Phrase', 'Prefix', 'Preposition', 'Pronoun', 'Proper Noun', 'Punctuation', 'Romanization', 'Root', 'Suffix', 'Symbol', 'Uncategorized', 'Verb']*
- `name_tl` (varchar(64), not null) - *Distinct Count: 24, Values: ['Bantas', 'Daglat', 'Hindi Tiyak', 'Hulapi', 'Kataga', 'Pagpapaikli', 'Pamilang', 'Pandamdam', 'Pandiwa', 'Pang-abay', 'Pang-ukol', 'Pang-uri', 'Pangatnig', 'Panghalip', 'Pangngalan', 'Pangngalang Pantangi', 'Panlapi', 'Pantukoy', 'Parirala', 'Romanisasyon', 'Salitang Ugat', 'Simbolo', 'Titik', 'Unlapi']*
- `description` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `parts_of_speech_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `parts_of_speech_pkey`*
- `parts_of_speech_code_uniq`: UNIQUE (`code`) - *Implicitly creates unique index `parts_of_speech_code_uniq`*

**Indexes (Explicitly Created):**
- `idx_parts_of_speech_code`: btree (`code`) - *Note: Covered by UNIQUE constraint `parts_of_speech_code_uniq`.*
- `idx_parts_of_speech_name`: btree (`name_en`, `name_tl`)

## Relations and Associations

### 5. `relations`
Represents semantic relationships between words (e.g., synonyms, antonyms).
- **Row Count:** 304,171

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `from_word_id` (integer, not null) - *References `words.id` (ON DELETE CASCADE)*
- `to_word_id` (integer, not null) - *References `words.id` (ON DELETE CASCADE)*
- `relation_type` (varchar(64), not null) - *Distinct Count: 12, Values: ['antonym', 'cognate_of', 'derived', 'derived_from', 'doublet_of', 'has_translation', 'related', 'root_of', 'see_also', 'synonym', 'translation_of', 'variant']*
- `sources` (text, nullable)
- `metadata` (jsonb, nullable, default: '{}'::jsonb)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `relations_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `relations_pkey`*
- `relations_unique`: UNIQUE (`from_word_id`, `to_word_id`, `relation_type`) - *Implicitly creates unique index `relations_unique`*
- `relations_from_word_id_fkey`: FOREIGN KEY (`from_word_id`) REFERENCES `words` (`id`) ON DELETE CASCADE
- `relations_to_word_id_fkey`: FOREIGN KEY (`to_word_id`) REFERENCES `words` (`id`) ON DELETE CASCADE

**Indexes (Explicitly Created):**
- `idx_relations_from`: btree (`from_word_id`) - *Note: Covered by UNIQUE constraint `relations_unique`.*
- `idx_relations_metadata`: gin (`metadata`)
- `idx_relations_metadata_strength`: btree (`((metadata ->> 'strength'::text)))`)
- `idx_relations_to`: btree (`to_word_id`) - *Note: Covered by UNIQUE constraint `relations_unique`.*
- `idx_relations_type`: btree (`relation_type`) - *Note: Covered by UNIQUE constraint `relations_unique`.*

### 6. `etymologies`
Tracks the origin and history of words.
- **Row Count:** 92,816

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `word_id` (integer, not null) - *References `words.id` (ON DELETE CASCADE)*
- `etymology_text` (text, not null)
- `normalized_components` (text, nullable)
- `etymology_structure` (text, nullable)
- `language_codes` (text, nullable)
- `sources` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `etymologies_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `etymologies_pkey`*
- `etymologies_wordid_etymtext_uniq`: UNIQUE (`word_id`, `etymology_text`) - *Implicitly creates unique index `etymologies_wordid_etymtext_uniq`*
- `etymologies_word_id_fkey`: FOREIGN KEY (`word_id`) REFERENCES `words` (`id`) ON DELETE CASCADE

**Indexes (Explicitly Created):**
- `idx_etymologies_langs`: gin (`to_tsvector('simple'::regconfig, language_codes)`)
- `idx_etymologies_word`: btree (`word_id`) - *Note: Covered by UNIQUE constraint `etymologies_wordid_etymtext_uniq`.*

### 7. `affixations`
Records relationships between root words and their derived forms through affixation.
- **Row Count:** 0

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `root_word_id` (integer, not null) - *References `words.id` (ON DELETE CASCADE)*
- `affixed_word_id` (integer, not null) - *References `words.id` (ON DELETE CASCADE)*
- `affix_type` (varchar(64), not null) - *Distinct Count: 0*
- `sources` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `affixations_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `affixations_pkey`*
- `affixations_unique`: UNIQUE (`root_word_id`, `affixed_word_id`, `affix_type`) - *Implicitly creates unique index `affixations_unique`*
- `affixations_root_word_id_fkey`: FOREIGN KEY (`root_word_id`) REFERENCES `words` (`id`) ON DELETE CASCADE
- `affixations_affixed_word_id_fkey`: FOREIGN KEY (`affixed_word_id`) REFERENCES `words` (`id`) ON DELETE CASCADE

**Indexes (Explicitly Created):**
- `idx_affixations_root`: btree (`root_word_id`) - *Note: Covered by UNIQUE constraint `affixations_unique`.*
- `idx_affixations_affixed`: btree (`affixed_word_id`) - *Note: Covered by UNIQUE constraint `affixations_unique`.*

### 8. `pronunciations`
Stores pronunciation information for words.
- **Row Count:** 95,005

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `word_id` (integer, not null) - *References `words.id` (ON DELETE CASCADE)*
- `type` (varchar(20), not null, default: 'ipa') - *Distinct Count: 4, Values: ['ipa', 'kwf_raw', 'respelling_guide', 'text']*
- `value` (text, not null)
- `tags` (jsonb, nullable)
- `pronunciation_metadata` (jsonb, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `pronunciations_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `pronunciations_pkey`*
- `pronunciations_unique`: UNIQUE (`word_id`, `type`, `value`) - *Implicitly creates unique index `pronunciations_unique`*
- `pronunciations_word_id_fkey`: FOREIGN KEY (`word_id`) REFERENCES `words` (`id`) ON DELETE CASCADE

**Indexes (Explicitly Created):**
- `idx_pronunciations_type`: btree (`type`) - *Note: Covered by UNIQUE constraint `pronunciations_unique`.*
- `idx_pronunciations_value`: btree (`value`) - *Note: Covered by UNIQUE constraint `pronunciations_unique`.*
- `idx_pronunciations_word`: btree (`word_id`) - *Note: Covered by UNIQUE constraint `pronunciations_unique`.*

### 9. `credits`
Stores credit information for word entries.
- **Row Count:** 2,348

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `word_id` (integer, not null) - *References `words.id` (ON DELETE CASCADE)*
- `credit` (text, not null)
- `sources` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `credits_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `credits_pkey`*
- `credits_unique`: UNIQUE (`word_id`, `credit`) - *Implicitly creates unique index `credits_unique`*
- `credits_word_id_fkey`: FOREIGN KEY (`word_id`) REFERENCES `words` (`id`) ON DELETE CASCADE

**Indexes (Explicitly Created):**
- `idx_credits_word`: btree (`word_id`) - *Note: Covered by UNIQUE constraint `credits_unique`.*

## Supporting Tables (Definition-related)

### 10. `definition_examples`
Provides usage examples for definitions.
- **Row Count:** 14,575

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `definition_id` (integer, not null) - *References `definitions.id` (ON DELETE CASCADE)*
- `example_text` (text, not null)
- `translation` (text, nullable)
- `example_type` (varchar(50), nullable) - *Distinct Count: 2, Values: ['example', 'quotation', null]*
- `reference` (text, nullable)
- `metadata` (jsonb, nullable)
- `sources` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `definition_examples_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `definition_examples_pkey`*
- `definition_examples_unique`: UNIQUE (`definition_id`, `example_text`) - *Implicitly creates unique index `definition_examples_unique`*
- `definition_examples_definition_id_fkey`: FOREIGN KEY (`definition_id`) REFERENCES `definitions` (`id`) ON DELETE CASCADE

**Indexes (Explicitly Created):**
- `idx_def_examples_def`: btree (`definition_id`) - *Note: Covered by UNIQUE constraint `definition_examples_unique`.*

### 11. `definition_links`
Stores external links or references related to definitions.
- **Row Count:** 1,379,566

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `definition_id` (integer, not null) - *References `definitions.id` (ON DELETE CASCADE)*
- `link_text` (text, not null)
- `link_url` (text, nullable)
- `tags` (text, nullable)
- `link_metadata` (jsonb, nullable, default: '{}'::jsonb)
- `sources` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `definition_links_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `definition_links_pkey`*
- `definition_links_unique`: UNIQUE (`definition_id`, `link_text`) - *Implicitly creates unique index `definition_links_unique`*
- *Foreign Key Implied:* `definition_id` REFERENCES `definitions` (`id`) ON DELETE CASCADE (No formal FK constraint found by script)

**Indexes (Explicitly Created):**
- `idx_def_links_def`: btree (`definition_id`) - *Note: Covered by UNIQUE constraint `definition_links_unique`.*
- `idx_def_links_type_trgm`: gin (`link_text` gin_trgm_ops) - *Note: Depends on `pg_trgm` extension.*

### 12. `definition_relations`
Represents relationships between definitions and words.
- **Row Count:** 0

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `definition_id` (integer, nullable) - *References `definitions.id` (ON DELETE CASCADE)*
- `word_id` (integer, nullable) - *References `words.id` (ON DELETE CASCADE)*
- `relation_type` (varchar(64), not null) - *Distinct Count: 0*
- `sources` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `definition_relations_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `definition_relations_pkey`*
- `definition_relations_unique`: UNIQUE (`definition_id`, `word_id`, `relation_type`) - *Implicitly creates unique index `definition_relations_unique`*
- `definition_relations_definition_id_fkey`: FOREIGN KEY (`definition_id`) REFERENCES `definitions` (`id`) ON DELETE CASCADE
- `definition_relations_word_id_fkey`: FOREIGN KEY (`word_id`) REFERENCES `words` (`id`) ON DELETE CASCADE

**Indexes (Explicitly Created):**
- `idx_def_relations_def`: btree (`definition_id`) - *Note: Covered by UNIQUE constraint `definition_relations_unique`.*
- `idx_def_relations_word`: btree (`word_id`) - *Note: Covered by UNIQUE constraint `definition_relations_unique`.*

### 13. `definition_categories`
Categorizes definitions.
- **Row Count:** 1,651,196

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `definition_id` (integer, nullable) - *References `definitions.id` (ON DELETE CASCADE)*
- `category_name` (text, not null)
- `category_kind` (text, nullable) - *Distinct Count: 4, Values: ['lifeform', 'other', 'place', 'topical']*
- `parents` (jsonb, nullable)
- `sources` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `category_metadata` (jsonb, nullable, default: '{}'::jsonb)

**Constraints:**
- `definition_categories_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `definition_categories_pkey`*
- `definition_categories_definition_id_category_name_key`: UNIQUE (`definition_id`, `category_name`) - *Implicitly creates unique index `definition_categories_definition_id_category_name_key`*
- *Foreign Key Implied:* `definition_id` REFERENCES `definitions` (`id`) ON DELETE CASCADE (No formal FK constraint found by script)

**Indexes (Explicitly Created):**
- `idx_def_categories_def`: btree (`definition_id`) - *Note: Covered by UNIQUE constraint `definition_categories_definition_id_category_name_key`.*
- `idx_def_categories_name`: btree (`category_name`)
- `idx_definition_categories_definition_id`: btree (`definition_id`) - *Note: Duplicate of `idx_def_categories_def`.*

## Supporting Tables (Word-related)

### 14. `word_forms`
Records different forms of words (inflections, spelling variants, etc.).
- **Row Count:** 1,223,492

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `word_id` (integer, nullable) - *References `words.id` (ON DELETE CASCADE)*
- `form` (text, not null)
- `is_canonical` (boolean, nullable, default: false) - *Distinct Count: 2, Values: [false, true]*
- `is_primary` (boolean, nullable, default: false) - *Distinct Count: 1, Values: [false]*
- `tags` (jsonb, nullable)
- `sources` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `word_forms_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `word_forms_pkey`*
- `word_forms_word_id_form_key`: UNIQUE (`word_id`, `form`) - *Implicitly creates unique index `word_forms_word_id_form_key`*
- *Foreign Key Implied:* `word_id` REFERENCES `words` (`id`) ON DELETE CASCADE (No formal FK constraint found by script)

**Indexes (Explicitly Created):**
- `idx_word_forms_form`: btree (`form`) - *Note: Covered by UNIQUE constraint `word_forms_word_id_form_key`.*
- `idx_word_forms_word`: btree (`word_id`) - *Note: Covered by UNIQUE constraint `word_forms_word_id_form_key`.*

### 15. `word_templates`
Stores templates for word formation patterns.
- **Row Count:** 575,479

**Columns:**
- `id` (integer, PK, not null, default: sequence)
- `word_id` (integer, nullable) - *References `words.id` (ON DELETE CASCADE)*
- `template_name` (text, not null) - *Distinct Count: 114, Examples: ['ceb-adj', 'ceb-adv', 'ceb-head', 'ceb-infl-gi-an', 'ceb-infl-i', 'ceb-infl-ma', 'ceb-infl-ma-an', 'ceb-infl-mag', 'ceb-infl-magka', 'ceb-infl-maka']*
- `args` (jsonb, nullable)
- `expansion` (text, nullable)
- `sources` (text, nullable)
- `created_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)
- `updated_at` (timestamptz, nullable, default: CURRENT_TIMESTAMP)

**Constraints:**
- `word_templates_pkey`: PRIMARY KEY (`id`) - *Implicitly creates unique index `word_templates_pkey`*
- `word_templates_word_id_template_name_key`: UNIQUE (`word_id`, `template_name`) - *Implicitly creates unique index `word_templates_word_id_template_name_key`*
- *Foreign Key Implied:* `word_id` REFERENCES `words` (`id`) ON DELETE CASCADE (No formal FK constraint found by script)

**Indexes (Explicitly Created):**
- `idx_word_templates_name`: btree (`template_name`) - *Note: Covered by UNIQUE constraint `word_templates_word_id_template_name_key`.*
- `idx_word_templates_word`: btree (`word_id`) - *Note: Covered by UNIQUE constraint `word_templates_word_id_template_name_key`.*

## Key Relationships Overview

This section describes relationships as defined by Foreign Key constraints found in `actual_schema.json` or implied by `REFERENCES` clauses in `CREATE TABLE` statements.

- `words` is a central table.
- `definitions` links to `words.id` (FK) and `parts_of_speech.id` (FK).
- `etymologies` links to `words.id` (FK).
- `affixations` links to `words.id` (FK for both `root_word_id` and `affixed_word_id`).
- `pronunciations` links to `words.id` (FK).
- `credits` links to `words.id` (FK).
- `relations` links `words.id` to `words.id` (FK for both `from_word_id` and `to_word_id`).
- `definition_examples` links to `definitions.id` (FK).
- `definition_relations` links to `definitions.id` (FK) and `words.id` (FK).
- `definition_links` links to `definitions.id` (Implied FK via `REFERENCES`).
- `definition_categories` links to `definitions.id` (Implied FK via `REFERENCES`).
- `word_forms` links to `words.id` (Implied FK via `REFERENCES`).
- `word_templates` links to `words.id` (Implied FK via `REFERENCES`).

## Indexing Strategy Notes (Based on `actual_schema.json`)

The `actual_schema.json` file lists index objects with their names (as found in `pg_indexes`), uniqueness, method, and full SQL definition. This summary attempts to list the primary functional indexes, noting where explicitly created indexes might overlap with those implicitly created by constraints.

- **B-tree indexes** are prevalent for primary keys, unique constraints, foreign keys, and many other queried columns.
- **GIN indexes** are utilized for:
    - Full-text search (e.g., `words.search_text`, `definitions.definition_text`, `etymologies.language_codes`).
    - JSONB querying (e.g., `words.word_metadata`, `relations.metadata`).
    - Trigram similarity search (e.g., `definition_links.link_text` using `gin_trgm_ops`).
- **Partial indexes** are present (e.g., `idx_words_baybayin` on `words.baybayin_form` WHERE `has_baybayin` = true).
- **Multi-column indexes** support unique constraints and optimize specific query patterns.

## Data Integrity Notes (Based on `actual_schema.json`)

- **Primary Keys** are defined for each table, ensuring entity integrity.
- **Foreign Keys**, where listed as constraints or implied by `REFERENCES`, help enforce referential integrity. Common actions include `ON UPDATE NO ACTION` and `ON DELETE CASCADE` or `ON DELETE SET NULL`.
- **Unique Constraints** are defined to prevent logical duplicates on specified column sets.
- **NOT NULL** constraints ensure essential data is present in columns.
- **Default Values** are specified for some columns (e.g., timestamps, booleans, JSONB fields).
- **CHECK constraints** were not part of the data fetched into `actual_schema.json` (except where manually added to the markdown, like for `words.baybayin_form`).

*(Removed previous general note about inconsistencies)* 