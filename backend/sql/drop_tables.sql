-- Drop tables in the correct order to avoid foreign key constraint issues

-- First drop triggers
DROP TRIGGER IF EXISTS update_words_timestamp ON words;
DROP TRIGGER IF EXISTS update_definitions_timestamp ON definitions;

-- Drop tables with foreign key constraints first
DROP TABLE IF EXISTS definition_relations CASCADE;
DROP TABLE IF EXISTS affixations CASCADE;
DROP TABLE IF EXISTS relations CASCADE;
DROP TABLE IF EXISTS etymologies CASCADE;
DROP TABLE IF EXISTS definitions CASCADE;

-- Drop main tables
DROP TABLE IF EXISTS words CASCADE;
DROP TABLE IF EXISTS parts_of_speech CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS update_timestamp(); 