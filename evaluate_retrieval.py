"""Evaluate retrieval accuracy of chunks from Qdrant vector store.

Loads test queries with one chunk per row, retrieves chunks, and computes
precision/recall/MRR metrics. Logs results for debugging MissionOS chatbot retrieval performance.
"""

import logging
import os
import pandas as pd
import time
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from setup import set_google_credentials, get_embeddings, get_vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("retrieval_debug.log")
    ]
)
logger = logging.getLogger(__name__)

def load_test_set(csv_path: str) -> pd.DataFrame:
    """Load test queries and ground-truth chunks from CSV.

    Expects one row per chunk, with columns: query,chunk_id,chunk_content,source_url.
    Groups rows by query for evaluation.

    Args:
        csv_path: Path to CSV file.

    Returns:
        DataFrame with test data, grouped by query.
    """
    try:
        df = pd.read_csv(csv_path)
        # Ensure chunk_id is string, handle NaN
        df["chunk_id"] = df["chunk_id"].fillna("").astype(str)
        # Verify required columns
        required_columns = ["query", "chunk_id", "chunk_content", "source_url"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing columns in CSV: {missing}")
        logger.info(f"Loaded test set with {len(df)} chunk entries from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading test set: {str(e)}")
        raise

def evaluate_retrieval(test_df: pd.DataFrame, vector_store: QdrantVectorStore, k: int = 4) -> tuple[dict, float, float]:
    """Evaluate retrieval accuracy for test queries with multiple ground truth chunks.

    Args:
        test_df: DataFrame with query,chunk_id,chunk_content,source_url (one row per chunk)
        vector_store: QdrantVectorStore instance
        k: Number of chunks to retrieve

    Returns:
        Tuple of (results dict, total execution time, total retrieval time)
    """
    start_total = time.time()
    total_retrieval_time = 0.0
    results = []
    precisions = []
    recalls = []
    ranks = []

    # Group by query to collect ground truth chunks
    grouped = test_df.groupby("query")

    for query, group in grouped:
        # Collect ground truth chunk IDs for this query
        ground_truth_ids = [cid.strip() for cid in group["chunk_id"] if cid.strip()]
        ground_truth_content = "; ".join(group["chunk_content"].astype(str))

        # Log ground truth
        logger.info(f"Query: {query}")
        logger.info(f"Ground truth IDs: {ground_truth_ids}")

        # Retrieve chunks
        try:
            start_retrieval = time.time()
            retrieved_docs = vector_store.similarity_search_with_score(query, k=k)
            retrieval_time = time.time() - start_retrieval
            total_retrieval_time += retrieval_time

            retrieved_ids = []
            retrieved_contents = []

            for i, (doc, score) in enumerate(retrieved_docs, 1):
                doc_id = doc.metadata.get("point_id", f"doc_{i}")  # Qdrant point ID or fallback
                retrieved_ids.append(doc_id)
                retrieved_contents.append(doc.page_content[:100] + "...")  # Truncate for logging
                logger.info(f"  Retrieved chunk {i}: ID={doc_id}, Score={score:.4f}, Content={retrieved_contents[-1]}")

            # Compute metrics
            relevant = [1 if doc_id in ground_truth_ids else 0 for doc_id in retrieved_ids]
            precision = sum(relevant) / k if k > 0 else 0
            recall = sum(relevant) / len(ground_truth_ids) if ground_truth_ids else 0

            # MRR: First relevant chunk's rank
            mrr = 0
            for rank, is_relevant in enumerate(relevant, 1):
                if is_relevant:
                    mrr = 1 / rank
                    break

            precisions.append(precision)
            recalls.append(recall)
            ranks.append(mrr)

            results.append({
                "query": query,
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
                "retrieved_ids": retrieved_ids,
                "ground_truth_ids": ground_truth_ids
            })
            logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, MRR: {mrr:.4f}, Retrieval Time: {retrieval_time:.2f}s")

        except Exception as e:
            logger.error(f"Error retrieving for query '{query}': {str(e)}")
            results.append({
                "query": query,
                "precision": 0,
                "recall": 0,
                "mrr": 0,
                "retrieved_ids": [],
                "ground_truth_ids": ground_truth_ids
            })

    # Aggregate metrics
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_mrr = sum(ranks) / len(ranks) if ranks else 0

    total_time = time.time() - start_total

    logger.info(f"Average Precision@{k}: {avg_precision:.4f}")
    logger.info(f"Average Recall@{k}: {avg_recall:.4f}")
    logger.info(f"Average MRR: {avg_mrr:.4f}")
    logger.info(f"Total Execution Time: {total_time:.2f}s")
    logger.info(f"Total Retrieval Time: {total_retrieval_time:.2f}s")

    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_mrr": avg_mrr,
        "per_query_results": results
    }, total_time, total_retrieval_time

def main():
    """Run retrieval evaluation."""
    # Initialize vector store using setup.py functions
    set_google_credentials()
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    # Load test set
    test_csv = "retrieval_test_set.csv"
    test_df = load_test_set(test_csv)

    # Evaluate
    results, total_time, retrieval_time = evaluate_retrieval(test_df, vector_store, k=4)

    # Print summary
    print(f"Average Precision@4: {results['avg_precision']:.4f}")
    print(f"Average Recall@4: {results['avg_recall']:.4f}")
    print(f"Average MRR: {results['avg_mrr']:.4f}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Retrieval Time: {retrieval_time:.2f} seconds")
    print("\nPer-query results:")
    for res in results["per_query_results"]:
        print(f"Query: {res['query']}")
        print(f"  Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}, MRR: {res['mrr']:.4f}")
        print(f"  Retrieved IDs: {res['retrieved_ids']}")
        print(f"  Ground Truth IDs: {res['ground_truth_ids']}")

if __name__ == "__main__":
    main()