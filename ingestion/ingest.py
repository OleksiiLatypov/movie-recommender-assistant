import os

import polars as pl
from elasticsearch import Elasticsearch, helpers

from config import (
    INDEX_NAME,
    ELASTICSEARCH_URL,
    DATA_PATH,
    BATCH_SIZE,
    get_index_mapping,
)


# ELASTICSEARCH

es = Elasticsearch(
    ELASTICSEARCH_URL,
    request_timeout=120,
    retry_on_timeout=True,
)

EMBEDDING_DIM = 768


# CREATE INDEX

def create_index():
    """
    Create Elasticsearch index with the predefined mapping
    if it does not already exist.
    """

    if es.indices.exists(index=INDEX_NAME):
        print(f"ℹ️ Index '{INDEX_NAME}' already exists.")
        return

    mapping = get_index_mapping(EMBEDDING_DIM)

    es.indices.create(
        index=INDEX_NAME,
        body=mapping,
    )

    print(f"✅ Index '{INDEX_NAME}' created.")


# INGESTION

def run_ingestion(parquet_path: str):
    """
    Load movies from Parquet and bulk index them into Elasticsearch.
    """

    # 1. CHECK DATA

    if not os.path.exists(parquet_path):
        print(f"❌ Error: Data file not found at {parquet_path}")
        return

    print(f"📂 Loading data from: {parquet_path}")

    df = pl.read_parquet(parquet_path)

    total_docs = len(df)

    print(f"✅ Data prepared. Total movies to index: {total_docs}")


    # 2. BULK INGESTION
    print(
        f"🚀 Starting ingestion in batches of {BATCH_SIZE}..."
    )

    for start in range(0, total_docs, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total_docs)

        chunk = (
            df
            .slice(start, BATCH_SIZE)
            .to_dicts()
        )

        actions = []

        for row in chunk:

            actions.append(
                {
                    "_index": INDEX_NAME,
                    "_source": {
                        # Core content
                        "title": row.get("title"),
                        "overview": row.get("overview"),
                        "search_context": row.get("input"),
                        "keywords": row.get("keywords"),

                        # Categories
                        "genres": row.get("genres"),
                        "language": row.get("original_language"),
                        "country": row.get("production_countries"),
                        "status": row.get("status"),

                        # People
                        "director": row.get("director"),
                        "writers": row.get("writers"),
                        "cast": row.get("cast"),

                        # Movie metadata
                        "runtime": row.get("runtime"),
                        "release_year": row.get("release_year"),
                        "popularity": row.get("popularity"),
                        "vote_average": row.get("vote_average"),
                        "imdb_rating": row.get("imdb_rating"),
                        "vote_count": row.get("vote_count"),

                        # URLs
                        "poster_url": row.get("poster_url"),
                        "movie_link": row.get("movie_link"),

                        # Vector embedding
                        "embeddings": row.get("embeddings"),
                    },
                }
            )

        # 3. SEND BATCH TO ELASTICSEARCH

        if actions:
            success, failed = helpers.bulk(
                es,
                actions,
                stats_only=True,
            )

            print(
                f"📦 Progress: {end}/{total_docs} "
                f"movies indexed | "
                f"success: {success} | failed: {failed}"
            )


    # 4. REFRESH INDEX

    es.indices.refresh(index=INDEX_NAME)


    
    # 5. VERIFY RESULT

    final_count = es.count(
        index=INDEX_NAME
    )["count"]

    print("\n✨ Ingestion completed!")
    print(
        f"🏁 Total documents in "
        f"'{INDEX_NAME}': {final_count}"
    )



if __name__ == "__main__":

    # Create index with predefined mapping
    create_index()

    # Ingest Parquet data
    run_ingestion(DATA_PATH)