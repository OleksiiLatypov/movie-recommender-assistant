import logging
import time
from pathlib import Path

import polars as pl

from rag.bm_25 import bm25_search
from rag.vector import vector_search
from rag.rrf import reciprocal_rank_fusion


# Create logs directory
Path("logs").mkdir(exist_ok=True)


# Evaluation logger
logger = logging.getLogger("rrf_evaluation")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(
    "logs/rrf_search.log",
    mode="a", 
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def calculate_metrics(
    ground_truth_path: str,
    top_n: int = 20,
    retrieve_k: int = 100,
):
    df_gt = pl.read_csv(ground_truth_path)
    total = len(df_gt)

    results_data = []

    logger.info(
        "Starting RRF evaluation | queries=%d | retrieve_k=%d | top_k=%d",
        total,
        retrieve_k,
        top_n,
    )

    start_eval_time = time.time()

    for row in df_gt.iter_rows(named=True):
        query = row["query"]
        expected_title = row["expected_title"]

        # --------------------------------
        # BM25 retrieval
        # --------------------------------
        bm25_results = bm25_search(
            query,
            retrieve_k=retrieve_k,
        )

        # --------------------------------
        # Vector retrieval
        # Original query — NO rewriting
        # --------------------------------
        vector_results = vector_search(
            query,
            retrieve_k=retrieve_k,
        )

        # --------------------------------
        # Reciprocal Rank Fusion
        # --------------------------------
        rrf_results = reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            top_n=top_n,
        )

        top_titles = [
            movie["title"]
            for movie in rrf_results
        ]

        # --------------------------------
        # Metrics
        # --------------------------------
        if expected_title in top_titles:
            rank = top_titles.index(expected_title) + 1
            reciprocal_rank = 1 / rank

            logger.info(
                "PASS | rank=%d | expected=%s | query=%s",
                rank,
                expected_title,
                query,
            )

        else:
            rank = None
            reciprocal_rank = 0.0

            logger.warning(
                "FAIL | expected=%s | query=%s",
                expected_title,
                query,
            )

        results_data.append({
            "query": query,
            "expected": expected_title,
            "rank": rank,
            "rr": reciprocal_rank,
        })

    results_df = pl.DataFrame(results_data)

    # Hit Rate@K
    hit_rate = (
        results_df.filter(
            pl.col("rank").is_not_null()
        ).height / total
    )

    # MRR@K
    mrr = results_df["rr"].mean()

    duration = time.time() - start_eval_time

    logger.info(
        "Evaluation finished | duration=%.2fs",
        duration,
    )

    logger.info(
        "Results | hit_rate=%.2f%% | mrr=%.3f",
        hit_rate * 100,
        mrr,
    )

    return hit_rate, mrr


if __name__ == "__main__":
    GT_PATH = "data/ground_truth.csv"

    TOP_K = 20
    RETRIEVE_K = 100

    hr, mrr = calculate_metrics(
        GT_PATH,
        top_n=TOP_K,
        retrieve_k=RETRIEVE_K,
    )

    logger.info("=" * 40)
    logger.info(
        "RRF RETRIEVAL PERFORMANCE (K=%d)",
        TOP_K,
    )
    logger.info("-" * 40)
    logger.info(
        "HIT RATE: %.2f%%",
        hr * 100,
    )
    logger.info(
        "MRR:      %.3f",
        mrr,
    )
    logger.info("=" * 40)