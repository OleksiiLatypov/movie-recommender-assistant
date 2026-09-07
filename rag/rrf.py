from rag.llm_query_rewriting import rewrite_query
from rag.vector import vector_search
from rag.bm_25 import bm25_search




def reciprocal_rank_fusion(bm25_results, vector_results, k=60, top_n=150):
   scores = {}
   documents = {}
   # BM25
   for rank, hit in enumerate(bm25_results, start=1):
       doc_id = hit["_id"]
       scores[doc_id] = scores.get(doc_id, 0) + (0.6 / (k + rank))
       documents[doc_id] = hit["_source"]
   # Vector
   for rank, hit in enumerate(vector_results, start=1):
       doc_id = hit["_id"]
       scores[doc_id] = scores.get(doc_id, 0) + (0.4 / (k + rank))
       documents[doc_id] = hit["_source"]
   ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
   results = []
   for doc_id, rrf_score in ranked[:top_n]:
       movie = documents[doc_id].copy()
       movie["rrf_score"] = rrf_score
       movie["_id"] = doc_id
       results.append(movie)
   return results





