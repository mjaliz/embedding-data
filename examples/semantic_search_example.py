"""
Example usage of the semantic search functionality
"""

import asyncio

from loguru import logger

from src.semantic_search import fetch_and_search_queries, search_queries_from_db


async def main():
    """Example of how to use the semantic search functions"""

    # Example 1: Basic semantic search
    logger.info("=== Example 1: Basic Semantic Search ===")
    try:
        results = await search_queries_from_db(
            query_text="laptop computer price",
            limit=5,
            score_threshold=0.7,  # Only return results with similarity > 0.7
        )

        logger.info(f"Search completed in {results.search_time_ms:.2f}ms")
        logger.info(
            f"Found {results.total_results} results for: '{results.query_text}'"
        )

        for i, result in enumerate(results.results, 1):
            logger.info(f"{i}. ID: {result.id}, Score: {result.score:.4f}")
            logger.info(f"   Query: {result.query[:100]}...")
            print()

    except Exception as e:
        logger.error(f"Example 1 failed: {str(e)}")

    # Example 2: Fetch and search queries without BGE M3 candidate flag
    logger.info("=== Example 2: Fetch and Search (BGE M3 Candidates) ===")
    try:
        results_list = await fetch_and_search_queries(
            db_limit=5,  # Fetch 5 queries from DB (where has_bge_m3_candidate is False)
            search_limit=3,  # Return top 3 similar results for each query
            score_threshold=0.6,
        )

        logger.info(f"Processed {len(results_list)} queries from database")

        for query_idx, results in enumerate(results_list, 1):
            logger.info(
                f"\n--- Results for Query {query_idx}: '{results.query_text[:50]}...' ---"
            )
            logger.info(f"Search completed in {results.search_time_ms:.2f}ms")
            logger.info(f"Found {results.total_results} similar results:")

            for i, result in enumerate(results.results, 1):
                logger.info(f"  {i}. ID: {result.id}, Score: {result.score:.4f}")
                logger.info(f"     Similar Query: {result.query[:80]}...")

    except Exception as e:
        logger.error(f"Example 2 failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
