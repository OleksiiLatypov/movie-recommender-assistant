# 🎬 Movie Recommender Assistant

An AI-powered movie recommendation assistant that combines **hybrid search, semantic embeddings, Reciprocal Rank Fusion (RRF), CrossEncoder reranking, and LLM-generated recommendations**.

The system is designed to understand natural-language movie requests such as:

> "I want a psychological thriller where a woman disappears and her husband becomes the prime suspect."

Instead of relying only on keyword matching, the application combines lexical and semantic retrieval to find relevant movies and uses an LLM to generate the final recommendation.

---

## 🏗️ Architecture

```text
                        User Query
                            │
                            ▼
                    ┌───────────────┐
                    │   FastAPI API  │
                    └───────┬───────┘
                            │
                            ▼
                    Query Processing
                            │
                            ▼
              ┌─────────────────────────┐
              │     Hybrid Retrieval    │
              │                         │
              │  BM25 + Vector Search   │
              └────────────┬────────────┘
                           │
                           ▼
                    RRF Fusion
                           │
                           ▼
                 Candidate Movies
                           │
                           ▼
              ┌─────────────────────────┐
              │    CrossEncoder         │
              │       Reranking         │
              └────────────┬────────────┘
                           │
                           ▼
                    Top Candidates
                           │
                           ▼
              ┌─────────────────────────┐
              │       LLM               │
              │ Recommendation Generator│
              └────────────┬────────────┘
                           │
                           ▼
                 Final Recommendation
                           │
                           ▼
             PostgreSQL → Grafana
                    Monitoring
```

---

## ✨ Features

- 🔎 **Hybrid movie search**
  - BM25 lexical search
  - Dense vector semantic search
  - Reciprocal Rank Fusion (RRF)

- 🧠 **Semantic search**
  - BGE embeddings
  - Cosine similarity
  - Elasticsearch `dense_vector`

- 🎯 **CrossEncoder reranking**
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`

- 🤖 **LLM-powered recommendations**
  - Natural-language explanations
  - Groq API
  - `openai/gpt-oss-120b`

- ⚡ **FastAPI REST API**

- 🗄️ **PostgreSQL monitoring database**
  - Conversations
  - Feedback
  - Response metadata

- 📊 **Grafana dashboard**
  - Request statistics
  - Response time
  - Feedback
  - LLM usage/cost metrics

- 🐳 **Docker Compose**
  - FastAPI
  - Elasticsearch
  - PostgreSQL
  - Grafana

- 📈 **Search evaluation**
  - HitRate@K
  - MRR@K
  - LLM-as-a-Judge evaluation

---

## 🧠 Search Pipeline

The recommendation system uses several retrieval stages.

### 1. Query

The user sends a natural-language query:

```text
movies similar to Batman
```

or:

```text
psychological thriller where the wife disappears
and the husband becomes the prime suspect
```

### 2. BM25 Search

Elasticsearch performs lexical retrieval using fields such as:

- title
- keywords
- genres
- overview
- director
- cast

Field boosting is used to give more importance to relevant fields.

### 3. Vector Search

The query is converted into an embedding using:

```text
BAAI/bge-base-en-v1.5
```

The movie dataset contains precomputed embeddings, which are stored in Elasticsearch as `dense_vector`.

Vector search uses cosine similarity.

### 4. Reciprocal Rank Fusion

BM25 and vector search produce separate rankings.

RRF combines these rankings into a single candidate list.

Conceptually:

```text
BM25 results
       +
Vector results
       ↓
      RRF
       ↓
Unified ranking
```

This allows the system to benefit from both exact keyword matching and semantic similarity.

### 5. CrossEncoder Reranking

The retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The CrossEncoder directly evaluates the relationship between:

```text
query ↔ movie
```

and produces a relevance score.

### 6. LLM Recommendation

The most relevant movies are passed to the LLM.

The LLM generates a natural-language recommendation based on the user's query and retrieved movies.

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI |
| Search engine | Elasticsearch 8.13.4 |
| Lexical search | BM25 |
| Vector search | Elasticsearch kNN |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Reranking | CrossEncoder |
| LLM | Groq / GPT OSS |
| Database | PostgreSQL 15 |
| Monitoring | Grafana |
| Containerization | Docker / Docker Compose |
| Package manager | UV |
| Data processing | Polars |
| Language | Python 3.12 |

---

## 📂 Project Structure

```text
movie-recommender-assistant/
│
├── main.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── README.md
│
├── ingestion/
│   ├── config.py
│   └── ingest.py
│
├── rag/
│   ├── hybrid_search.py
│   ├── vector_search.py
│   ├── llm_query_rewriting.py
│   └── llm_recommendation.py
│
├── monitoring/
│   └── db.py
│
├── evaluation/
│   ├── ...
│   └── ground_truth.csv
│
├── grafana/
│   ├── dashboards/
│   └── provisioning/
│
├── data/
│   └── movies_with_embeddings.parquet
│
└── tests/
    └── ...
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/movie-recommender-assistant.git
cd movie-recommender-assistant
```

---

## 2. Install dependencies

The project uses [UV](https://docs.astral.sh/uv/).

```bash
uv sync
```

Activate the environment if needed:

```bash
source .venv/bin/activate
```

---

## 3. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key

POSTGRES_DB=movie_db
POSTGRES_USER=admin
POSTGRES_PASSWORD=password

GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
```

Do not commit `.env` to GitHub.

---

# 📦 Dataset

The project uses a movie dataset containing approximately **100,000 movies** with metadata and precomputed embeddings.

The processed dataset contains information such as:

- title
- overview
- genres
- keywords
- director
- writers
- cast
- runtime
- release year
- ratings
- popularity
- movie links
- poster URLs
- embeddings

The embeddings are generated using:

```text
BAAI/bge-base-en-v1.5
```

---

# 🔥 Elasticsearch Ingestion

Start Elasticsearch first:

```bash
docker compose up -d elasticsearch
```

Check that Elasticsearch is available:

```bash
curl http://localhost:9200
```

Expected response contains:

```json
{
  "version": {
    "number": "8.13.4"
  }
}
```

Then run the ingestion script:

```bash
uv run python ingestion/ingest.py
```

The script creates the `movies` index and uploads the movie dataset in batches.

Check the index:

```bash
curl http://localhost:9200/_cat/indices?v
```

You should see:

```text
movies
```

---

# 🐳 Run with Docker Compose

Start the complete application:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

The stack contains:

```text
FastAPI        → localhost:8000
Elasticsearch  → localhost:9200
PostgreSQL     → localhost:5433
Grafana        → localhost:3000
```

---

## 🔍 Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_device": "cpu"
}
```

---

# 🎬 Recommendation API

Send a movie recommendation request:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"movies similar to Batman"}'
```

Example natural-language query:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"psychological thrillers where someone disappears and the investigation changes the relationship between the main characters"}'
```

The response contains:

```json
{
  "conversation_id": "...",
  "recommendation": "...",
  "top_movies": [
    {
      "title": "...",
      "year": 2014,
      "genres": "...",
      "director": "...",
      "cast": "...",
      "runtime": 120,
      "imdb_rating": 7.8,
      "vote_average": 7.9,
      "final_score": 8.42
    }
  ],
  "response_time": 1.84
}
```

---

# 👍 Feedback API

Users can provide positive or negative feedback for a recommendation.

Positive feedback:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id":"YOUR_CONVERSATION_ID",
    "feedback":1
  }'
```

Negative feedback:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id":"YOUR_CONVERSATION_ID",
    "feedback":-1
  }'
```

---

# 📊 Monitoring

The application stores recommendation metadata and user feedback in PostgreSQL.

Grafana connects to PostgreSQL and provides dashboards for monitoring the application.

Open Grafana:

```text
http://localhost:3000
```

Default credentials:

```text
Username: admin
Password: admin
```

unless overridden through environment variables.

Inside the Docker network, Grafana connects to PostgreSQL using:

```text
Host: postgres
Port: 5432
```

The host machine exposes PostgreSQL on:

```text
localhost:5433
```

---

# 📈 Evaluation

The retrieval system was evaluated using standard information-retrieval metrics.

### HitRate@K

Measures whether at least one relevant movie appears in the top K results.

```text
HitRate@10
```

asks:

> Did the system retrieve a relevant movie within the first 10 results?

### MRR@K

Mean Reciprocal Rank measures how high the first relevant result appears.

For example:

```text
Relevant result at position 1 → 1.0
Relevant result at position 2 → 0.5
Relevant result at position 5 → 0.2
```

### LLM-as-a-Judge

An additional evaluation layer uses an LLM to assess the relevance of generated recommendations.

The judge evaluates retrieved movies against the user's query and ground-truth expectations.

---

# 🧪 Example Evaluation Results

Example evaluation:

```text
Average Relevance: 2.50 / 3
Success Rate:      80%
```

Retrieval evaluation can also be performed using:

```text
HitRate@10
MRR@10
Recall@10
```

These metrics help compare different retrieval strategies, embedding models, and reranking configurations.

---

# ⚙️ Configuration

Important Elasticsearch configuration:

```env
ELASTICSEARCH_URL=http://elasticsearch:9200
```

When running the application inside Docker, **do not use**:

```text
localhost:9200
```

because `localhost` refers to the FastAPI container itself.

Docker service names are used for internal communication:

```text
FastAPI → elasticsearch:9200
FastAPI → postgres:5432
Grafana → postgres:5432
```

From the host machine:

```text
localhost:9200
localhost:5433
localhost:8000
localhost:3000
```

---

# 🧠 Models

### Embedding Model

```text
BAAI/bge-base-en-v1.5
```

Used for semantic retrieval.

### CrossEncoder

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Used for candidate reranking.

### LLM

```text
openai/gpt-oss-120b
```

Used to generate the final natural-language recommendation through the Groq API.

---

# 💡 Why Hybrid Search?

Pure keyword search can fail when the user describes a movie without using the exact words appearing in its metadata.

For example:

```text
"movie about a man investigating his wife's disappearance"
```

may not share many exact keywords with the movie description.

Semantic search can recognize the underlying meaning, while BM25 remains useful for:

- exact titles
- actor names
- directors
- specific keywords
- genres

Combining both approaches with RRF makes the retrieval system more robust.

---

# 🔄 Recommendation Pipeline

The complete pipeline can be summarized as:

```text
User Query
    ↓
Query Rewriting
    ↓
BM25 Search ──────────┐
                      │
Vector Search ────────┤
                      ↓
                     RRF
                      ↓
              Candidate Retrieval
                      ↓
              CrossEncoder Rerank
                      ↓
                 Top Movies
                      ↓
                    LLM
                      ↓
          Natural Language Response
                      ↓
             PostgreSQL Logging
                      ↓
                   Grafana
```

---

# 🛠️ Development

Run the API locally:

```bash
uv run uvicorn main:app --reload
```

Run tests:

```bash
uv run pytest
```

Check Docker logs:

```bash
docker compose logs app
```

Follow application logs:

```bash
docker compose logs -f app
```

Stop the stack:

```bash
docker compose down
```

> Do not use `docker compose down -v` unless you intentionally want to remove the PostgreSQL, Elasticsearch, and Grafana volumes.

---

# 📌 Future Improvements

Potential improvements include:

- [ ] Improve hybrid retrieval weights
- [ ] Optimize RRF parameters
- [ ] Improve query rewriting
- [ ] Optimize LLM context size
- [ ] Add caching for repeated queries
- [ ] Add automated evaluation to CI
- [ ] Add model/version tracking
- [ ] Add more detailed observability
- [ ] Add recommendation diversity
- [ ] Add personalized recommendations based on user feedback
- [ ] Improve cold-start handling
- [ ] Add automated model evaluation
- [ ] Add production deployment

---

# 👨‍💻 Project Goals

This project was built to explore practical applications of **LLM-powered information retrieval and recommendation systems**.

The main focus areas are:

- Retrieval-Augmented Generation
- Hybrid Search
- Vector Search
- Embeddings
- Reranking
- Elasticsearch
- LLM evaluation
- API development
- Docker
- PostgreSQL
- Monitoring
- MLOps

---

## 📄 License

This project is intended for educational and portfolio purposes.