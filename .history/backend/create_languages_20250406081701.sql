CREATE TABLE languages (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    name_en VARCHAR(100),
    name_tl VARCHAR(100),
    region VARCHAR(100),
    family VARCHAR(100),
    status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_languages_code ON languages (code);

-- Optional: Trigger to update 'updated_at' timestamp automatically
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ language 'plpgsql';

-- Check if trigger exists before creating to avoid errors
DO $$
BEGIN
   IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_languages_modtime') THEN
      CREATE TRIGGER update_languages_modtime
      BEFORE UPDATE ON languages
      FOR EACH ROW
      EXECUTE FUNCTION update_modified_column();
   END IF;
END;
$$; 