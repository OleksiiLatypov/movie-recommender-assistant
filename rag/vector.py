import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import torch

# Modular imports
from rag.llm_query_rewriting import rewrite_query
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL



load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Initialize Elasticsearch
es = Elasticsearch(ELASTICSEARCH_URL, request_timeout=60, retry_on_timeout=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


def vector_search(query, retrieve_k=100):
   query_for_embedding = (
                            "Represent this sentence for searching relevant movie plots: " + query
                            )
   query_vector = bi_encoder.encode(query_for_embedding, normalize_embeddings=True).tolist()
   body = {
       "size": retrieve_k,
       "knn": {
           "field": "embeddings",
           "query_vector": query_vector,
           "k": retrieve_k,
           "num_candidates": 200
       },
       "_source": [
           "title",
           "overview",
           "genres",
           "director",
           "writers",
           "cast",
           "keywords",
           "popularity",
           "vote_average",
           "poster_url",
           "movie_link",
           "release_year",
           "runtime"
       ]
   }
   response = es.search(index=INDEX_NAME, body=body)
   return response["hits"]["hits"]



