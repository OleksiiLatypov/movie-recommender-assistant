import os
import json
import logging
import time
from pathlib import Path

import polars as pl
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

from rag.hybrid_search import search_rrf_pipeline as search_movies


load_dotenv()


# ============================================================
# LOGGING
# ============================================================

Path("logs").mkdir(exist_ok=True)

logger = logging.getLogger("llm_judge_evaluation")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(
    "logs/llm_judge.log",
    mode="a",
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ============================================================
# OPENAI / GROQ CLIENT
# ============================================================

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


JUDGE_MODEL = "openai/gpt-oss-120b"


# ============================================================
# JSON PROMPT
# ============================================================

JUDGE_PROMPT = """
You are a search quality judge. Evaluate the relevance of the TOP 3 search results for a movie query.

USER QUERY: {query}
EXPECTED MOVIE: {expected}

RESULTS:
{results_text}

SCORING RUBRIC:
3: EXCELLENT - Perfect match or exactly what was requested.
2: GOOD - Highly relevant, same vibe/genre/actors.
1: PARTIAL - Slight connection (e.g., same director but wrong movie).
0: IRRELEVANT - No connection.

You MUST return a JSON object with this structure:
{{
  "evaluations": [
    {{ "rank": 1, "score": number, "reason": "string" }},
    {{ "rank": 2, "score": number, "reason": "string" }},
    {{ "rank": 3, "score": number, "reason": "string" }}
  ]
}}
"""


# ============================================================
# LLM JUDGE
# ============================================================

def get_top_n_judgments(query, expected, movies):
    """
    Evaluate a list of movies in one LLM call.
    """

    results_text = ""

    for i, movie in enumerate(movies, 1):
        results_text += f"""
RANK {i}
Title: {movie.get('title', '')}
Genres: {movie.get('genres', '')}
Director: {movie.get('director', '')}
Cast: {movie.get('cast', '')}
Overview: {movie.get('overview', '')}
Keywords: {movie.get('keywords', '')}
"""

    prompt = JUDGE_PROMPT.format(
        query=query,
        expected=expected,
        results_text=results_text,
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        data = json.loads(
            response.choices[0].message.content
        )

        evaluations = data.get("evaluations", [])

        logger.info(
            "Judge completed | query=%s | evaluations=%d",
            query,
            len(evaluations),
        )

        return evaluations

    except Exception as e:
        logger.exception(
            "LLM Judge error | query=%s | error=%s",
            query,
            e,
        )
        return []


# ============================================================
# EVALUATION
# ============================================================

def run_judge_evaluation(
    gt_path,
    output_path,
    limit=None,
):
    df = pl.read_csv(gt_path)

    if limit:
        df = df.head(limit)

    total_queries = len(df)

    all_results = []

    logger.info("=" * 60)
    logger.info("Starting Top-3 LLM Judge Evaluation")
    logger.info(
        "queries=%d | model=%s",
        total_queries,
        JUDGE_MODEL,
    )
    logger.info("=" * 60)

    start_eval_time = time.time()

    for index, row in enumerate(
        tqdm(
            df.iter_rows(named=True),
            total=total_queries,
            desc="LLM Judge",
        ),
        start=1,
    ):
        query = row["query"]
        expected = row["expected_title"]

        logger.info(
            "Processing query %d/%d | expected=%s | query=%s",
            index,
            total_queries,
            expected,
            query,
        )

        # --------------------------------------------
        # Search Top 3
        # --------------------------------------------

        try:
            search_hits = search_movies(
                query,
                top_n=3,
            )
        except Exception as e:
            logger.exception(
                "Search error | query=%s | error=%s",
                query,
                e,
            )
            continue

        if not search_hits:
            logger.warning(
                "No search results | query=%s",
                query,
            )
            continue

        logger.info(
            "Search completed | query=%s | results=%d",
            query,
            len(search_hits),
        )

        # --------------------------------------------
        # LLM Judge
        # --------------------------------------------

        evaluations = get_top_n_judgments(
            query,
            expected,
            search_hits,
        )

        if not evaluations:
            logger.warning(
                "No judge evaluations | query=%s",
                query,
            )
            continue

        # --------------------------------------------
        # Combine search + judge results
        # --------------------------------------------

        for i, movie in enumerate(search_hits):

            eval_data = next(
                (
                    item
                    for item in evaluations
                    if item.get("rank") == i + 1
                ),
                {
                    "score": 0,
                    "reason": "N/A",
                },
            )

            score = eval_data.get("score", 0)
            reason = eval_data.get("reason", "N/A")

            all_results.append({
                "query": query,
                "expected": expected,
                "retrieved_title": movie["title"],
                "rank": i + 1,
                "llm_score": score,
                "llm_reason": reason,
            })

            logger.info(
                "Result | rank=%d | title=%s | score=%s",
                i + 1,
                movie["title"],
                score,
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    if not all_results:
        logger.error("No evaluation results were produced.")
        return

    results_df = pl.DataFrame(all_results)

    # Average score @ Rank 1
    avg_rank1 = (
        results_df
        .filter(pl.col("rank") == 1)["llm_score"]
        .mean()
    )

    # Average score across all Top 3
    avg_overall = results_df["llm_score"].mean()

    # LLM-verified Hit Rate@3
    perfect_hits = (
        results_df
        .filter(pl.col("llm_score") == 3)
        .select("query")
        .unique()
        .height
    )

    hit_rate_at_3 = (
        perfect_hits / total_queries
    ) * 100

    duration = time.time() - start_eval_time

    # ========================================================
    # LOG FINAL RESULTS
    # ========================================================

    logger.info("=" * 60)
    logger.info("TOP-3 LLM JUDGE RESULTS")
    logger.info("-" * 60)
    logger.info(
        "Avg Score @ Rank 1: %.2f",
        avg_rank1,
    )
    logger.info(
        "Avg Score (All Top 3): %.2f",
        avg_overall,
    )
    logger.info(
        "LLM-Verified Hit Rate @ 3: %.1f%%",
        hit_rate_at_3,
    )
    logger.info(
        "Duration: %.2fs",
        duration,
    )
    logger.info(
        "Results saved to: %s",
        output_path,
    )
    logger.info("=" * 60)

    # Save detailed results
    results_df.write_csv(output_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    GT_PATH = "data/ground_truth_45.csv"

    OUTPUT_PATH = (
        "data/judge_top3_results.csv"
    )

    run_judge_evaluation(
        GT_PATH,
        OUTPUT_PATH,
    )