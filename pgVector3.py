import psycopg2
import requests
import random

# ------------------------------
# Configuration & API Parameters
# ------------------------------

PG_CONN_INFO = "dbname=vectorPractice user=postgres password=aak101010 host=localhost port=5432"
GEMINI_API_KEY = "AIzaSyBuGdO93ZamLCYqNgPgulzlcxowUd5dxkU"
GEMINI_ENDPOINT = "https://gemini.googleapis.com/v1/embeddings"
EMBEDDING_DIM = 768

# ------------------------------
# Embedding Functions
# ------------------------------

def get_embedding(text):
    """
    Calls the Google Gemini API to get an embedding for the input text.
    (This is a placeholder; adjust parameters according to the actual API.)
    """
    payload = {
        "model": "text-embedding-004",
        "input": text,
    }
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(GEMINI_ENDPOINT, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        embedding = data.get("embedding")
        if embedding is None:
            raise Exception("No embedding returned from API.")
        return embedding
    else:
        raise Exception(f"Error from Gemini API: {response.status_code} {response.text}")

def simulate_embedding(text, dimension=EMBEDDING_DIM):
    """
    Simulates an embedding by generating a deterministic random vector.
    Useful for testing when the real API call is not available.
    """
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(dimension)]

# ------------------------------
# Database Setup & Sample Data
# ------------------------------

def setup_database():
    """
    Connects to PostgreSQL, creates the pgVector extension and the documents table.
    """
    conn = psycopg2.connect(PG_CONN_INFO)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT,
            description TEXT,
            embedding vector({EMBEDDING_DIM})
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Database and table setup completed.")

def insert_sample_data():
    """
    Inserts sample documents into the database if none exist.
    """
    sample_data = [
        ("Algorithm Basics", "An introduction to algorithms, complexity, and computational theory."),
        ("Machine Learning", "Overview of supervised and unsupervised learning techniques in computer science."),
        ("Neural Networks", "Exploring deep learning and artificial neural networks, including backpropagation."),
        ("Data Structures", "Fundamental concepts of data organization, including arrays, linked lists, and trees."),
        ("Operating Systems", "Understanding processes, threads, memory management, and scheduling in OS."),
    ]
    conn = psycopg2.connect(PG_CONN_INFO)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents;")
    count = cur.fetchone()[0]
    if count == 0:
        for title, description in sample_data:
            cur.execute("INSERT INTO documents (title, description) VALUES (%s, %s)", (title, description))
        conn.commit()
        print("Sample data inserted.")
    else:
        print("Data already exists. Skipping sample data insertion.")
    cur.close()
    conn.close()

def update_embeddings(use_simulation=True):
    """
    Updates documents with a generated embedding for each description.
    """
    conn = psycopg2.connect(PG_CONN_INFO)
    cur = conn.cursor()
    cur.execute("SELECT id, description FROM documents WHERE embedding IS NULL;")
    rows = cur.fetchall()
    for doc_id, description in rows:
        embedding = simulate_embedding(description) if use_simulation else get_embedding(description)
        cur.execute("UPDATE documents SET embedding = %s WHERE id = %s", (embedding, doc_id))
        print(f"Updated document id {doc_id} with embedding.")
    conn.commit()
    cur.close()
    conn.close()

# ------------------------------
# Similarity Search Functionality
# ------------------------------

def search_similar_documents(query_text, use_simulation=True, limit=3):
    """
    Generates an embedding for the query text and performs a similarity search.
    Returns the top matching documents.
    """
    query_embedding = simulate_embedding(query_text) if use_simulation else get_embedding(query_text)
    conn = psycopg2.connect(PG_CONN_INFO)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, description, embedding <-> %s::vector AS distance
        FROM documents
        ORDER BY distance
        LIMIT %s;
    """, (query_embedding, limit))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# ------------------------------
# Main Flow
# ------------------------------

def main():
    setup_database()
    insert_sample_data()
    update_embeddings(use_simulation=True)
    
    query_text = "Learn about deep learning and backpropagation in neural networks."
    results = search_similar_documents(query_text, use_simulation=True, limit=3)
    
    print("\nSimilarity search results:")
    for doc_id, title, description, distance in results:
        print(f"ID: {doc_id}, Title: {title}, Distance: {distance:.4f}")

if __name__ == "__main__":
    main()
