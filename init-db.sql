-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create people table
CREATE TABLE IF NOT EXISTS people (
    person_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create embeddings table with vector column
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(255) NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
    embedding vector(192) NOT NULL,
    source_image VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS embeddings_vector_idx
ON embeddings USING hnsw (embedding vector_l2_ops);

-- Create index on person_id for fast lookups
CREATE INDEX IF NOT EXISTS embeddings_person_id_idx
ON embeddings(person_id);

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to auto-update updated_at
CREATE TRIGGER update_people_updated_at BEFORE UPDATE ON people
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
