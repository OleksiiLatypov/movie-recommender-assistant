from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL
from rag.llm_query_rewriting import rewrite_query
from rag.vector import vector_search, bi_encoder, cross_encoder
from rag.bm_25 import bm25_search
from rag.rrf import reciprocal_rank_fusion
import numpy as np


# # To this:
# es = Elasticsearch(
#     ELASTICSEARCH_URL,
#     request_timeout=60,       # Increased to 60 seconds
#     retry_on_timeout=True, 
#     max_retries=3
# )



def rerank_movies(query, movies):
    if not movies:
        return []

    pairs = []
    for m in movies:
        context = (
                        f"Title: {m.get('title') or ''}. "
                        f"Genres: {m.get('genres') or ''}. "
                        f"Plot: {m.get('overview') or ''}. "
                        f"Keywords: {m.get('keywords') or ''}. "
                        f"Cast: {m.get('cast') or ''}. "
                        f"Director: {m.get('director') or ''}"
                    )
        pairs.append([query, context])
    
    scores = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
    
    for movie, score in zip(movies, scores):
        movie["cross_score"] = float(score)
        # Apply your popularity boost here if desired
        pop = movie.get("popularity", 0) or 0
        movie["final_score"] = movie["cross_score"]# + (np.log(pop + 1) * 0.85)

    movies.sort(key=lambda x: x["final_score"], reverse=True)
    return movies


