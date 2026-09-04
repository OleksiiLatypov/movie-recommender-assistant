import os
import polars as pl
from elasticsearch import Elasticsearch, helpers

# Import settings from your config
from config import INDEX_NAME, ELASTICSEARCH_URL, DATA_PATH, BATCH_SIZE


# Initialize Elasticsearch
es = Elasticsearch(
    ELASTICSEARCH_URL,
    request_timeout=120,
    retry_on_timeout=True
)

def run_ingestion(parquet_path: str):
    # 1. LOAD DATA
    if not os.path.exists(parquet_path):
        print(f"❌ Error: Data file not found at {parquet_path}")
        return

    print(f"📂 Loading data from: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    
    total_docs = len(df)
    
    print(f"✅ Data prepared. Total movies to index: {total_docs}")

    # 2. CHUNKED INGESTION LOOP
    print(f"🚀 Starting ingestion in chunks of {BATCH_SIZE}...")

    for start in range(0, total_docs, BATCH_SIZE):
        # Calculate end of the current chunk
        end = min(start + BATCH_SIZE, total_docs)
        
        # Take a slice of the Polars DataFrame for this batch
        # This is memory efficient because it doesn't duplicate the whole DF
        chunk = df.slice(start, BATCH_SIZE).to_dicts()
        
        actions = []
        for row in chunk:
            # Map the flat Parquet row to the Elasticsearch document structure
            actions.append({
                "_index": INDEX_NAME,
                "_source": {
                    "title":          row.get("title"),
                    "overview":       row.get("overview"),
                    "search_context": row.get("input"), 
                    "keywords":       row.get("keywords"),
                    "genres":         row.get("genres"),
                    "director":       row.get("director"),
                    "writers":        row.get("writers"),
                    "cast":           row.get("cast"),
                    "language":       row.get("original_language"),
                    "country":        row.get("production_countries"),
                    "status":         row.get("status"),
                    "runtime":        row.get("runtime"),
                    "release_year":   row.get("release_year"),
                    "popularity":     row.get("popularity"),
                    "vote_average":   row.get("vote_average"),
                    "imdb_rating":    row.get("imdb_rating"),
                    "vote_count":     row.get("vote_count"),
                    "poster_url":     row.get("poster_url"),
                    "movie_link":     row.get("movie_link"),
                    
                    # THE FIX: Ensure key is 'embedding' (not 'embeddding')
                    "embeddings":      row.get("embeddings") 
                }
            })

        # Bulk upload the chunk to Elasticsearch
        if actions:
            helpers.bulk(es, actions, stats_only=True)
        
        print(f"📦 Progress: {end}/{total_docs} movies indexed...")

    # 3. FINALIZE
    es.indices.refresh(index=INDEX_NAME)
    final_count = es.count(index=INDEX_NAME)["count"]
    print(f"\n✨ Ingestion Successful!")
    print(f"🏁 Total documents now in '{INDEX_NAME}': {final_count}")

if __name__ == "__main__":
    # Set PYTHONPATH=. in your terminal to ensure imports work
    run_ingestion(DATA_PATH)