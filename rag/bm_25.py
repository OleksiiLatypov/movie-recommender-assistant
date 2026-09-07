from elasticsearch import Elasticsearch
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL


# Initialize Elasticsearch
es = Elasticsearch(ELASTICSEARCH_URL, request_timeout=60, retry_on_timeout=True)


def bm25_search(query, retrieve_k=100):
   body = {
       "size": retrieve_k,
       "query": {
           "multi_match": {
               "query": query,
               "fields": [
                   "genres.text",
                   "overview^2",
                   "cast^3",
                   "director",
                   "search_context^3"
               ],
               "type": "best_fields"
           }
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


