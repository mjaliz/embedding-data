import time

from loguru import logger

from src.config import config
from src.database import get_async_session_context
from src.embedding import Embedder
from src.milvus_repo import MilvusRepo
from src.models import SearchResponse, SearchResult
from src.postgres_repo import PostgresRepo


class SemanticSearchService:
    """Service for performing semantic search on queries"""

    def __init__(self):
        self.embedder = Embedder("BAAI/bge-m3")  # BGE model for embeddings
        self.milvus_repo = MilvusRepo(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)
        self.collection_name = "bge_m3_embeddings"  # Using project name from memory

    async def search_similar_queries(
        self, query_text: str, limit: int = 10, score_threshold: float = 0.0
    ) -> SearchResponse:
        """
        Search for similar queries in the database using semantic search

        Args:
            query_text: The text to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            SearchResponse containing the search results
        """
        start_time = time.time()

        try:
            # Generate embedding for the query text
            logger.info(f"Generating embedding for query: {query_text[:100]}...")
            query_embedding = self.embedder.embed(
                [query_text], show_progress_bar=False
            )[0]

            # Search in Milvus
            logger.info(f"Searching in Milvus collection: {self.collection_name}")
            search_results = self.milvus_repo.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                output_fields=["id", "query"],
            )

            # Process search results
            results = []
            if search_results and len(search_results[0]) > 0:
                async with get_async_session_context() as session:
                    postgres_repo = PostgresRepo(session)

                    for hit in search_results[0]:
                        # Filter by score threshold
                        if hit["distance"] < score_threshold:
                            continue

                        # Get the full query from PostgreSQL
                        query_obj = await postgres_repo.get_query_by_id(hit["id"])
                        if query_obj:
                            result = SearchResult(
                                id=hit["id"],
                                query=query_obj.query,
                                score=hit[
                                    "distance"
                                ],  # Milvus returns distance, higher = more similar
                                distance=hit["distance"],
                            )
                            results.append(result)

            search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            logger.info(
                f"Search completed in {search_time:.2f}ms, found {len(results)} results"
            )

            return SearchResponse(
                query_text=query_text,
                results=results,
                total_results=len(results),
                search_time_ms=search_time,
            )

        except Exception as e:
            logger.error(f"Error during semantic search: {str(e)}")
            raise


async def search_queries_from_db(
    query_text: str, limit: int = 10, score_threshold: float = 0.0
) -> SearchResponse:
    """
    Convenience function to perform semantic search

    Args:
        query_text: The text to search for
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score threshold

    Returns:
        SearchResponse containing the search results
    """
    search_service = SemanticSearchService()
    return await search_service.search_similar_queries(
        query_text=query_text, limit=limit, score_threshold=score_threshold
    )


async def fetch_and_search_queries(
    db_limit: int = 1000,
    search_limit: int = 100,
    score_threshold: float = 0.6,
) -> list[SearchResponse]:
    """
    Fetches queries from DB where has_bge_m3_candidate is False and performs search for each one
    This function finds similar queries for each fetched query

    Args:
        db_limit: Number of queries to fetch from database
        search_limit: Maximum number of similar results to return for each query
        score_threshold: Minimum similarity score threshold

    Returns:
        List of SearchResponse objects, one for each fetched query
    """
    start_time = time.time()

    try:
        # Fetch queries from database where has_bge_m3_candidate is False
        async with get_async_session_context() as session:
            postgres_repo = PostgresRepo(session)
            queries = await postgres_repo.get_queries_without_bge_m3_candidate(
                offset=0, limit=db_limit
            )

        if not queries:
            logger.warning("No queries found in database without BGE M3 candidate flag")
            return []

        logger.info(
            f"Fetched {len(queries)} queries without BGE M3 candidate from database"
        )

        # Search for similar queries for each fetched query
        search_service = SemanticSearchService()
        results = []

        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query.query[:50]}...")

            try:
                search_result = await search_service.search_similar_queries(
                    query_text=query.query,
                    limit=search_limit,
                    score_threshold=score_threshold,
                )
                results.append(search_result)

                # Extract candidate queries from search results
                bge_m3_candidates = [result.query for result in search_result.results]

                # Update the query with candidates and set the flag
                async with get_async_session_context() as session:
                    postgres_repo = PostgresRepo(session)
                    await postgres_repo.update_query_candidates(
                        query_id=query.id,
                        bge_m3_candidates=bge_m3_candidates,
                        has_bge_m3_candidate=True,
                    )

                logger.info(
                    f"Updated query {query.id} with {len(bge_m3_candidates)} BGE M3 candidates"
                )

            except Exception as e:
                logger.error(f"Error searching for query ID {query.id}: {str(e)}")
                # Continue with the next query instead of failing completely
                continue

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"Completed searching for {len(results)} queries in {total_time:.2f}ms"
        )

        return results

    except Exception as e:
        logger.error(f"Error during fetch and search: {str(e)}")
        raise


if __name__ == "__main__":
    import asyncio

    async def test_search():
        """Test function for semantic search"""
        test_query = "سینی پنج تکه مهره دار"

        logger.info(f"Testing semantic search with query: '{test_query}'")

        try:
            # Test the main search function
            results = await search_queries_from_db(
                query_text=test_query, limit=200, score_threshold=0.5
            )

            logger.info(f"Found {results.total_results} results:")
            for i, result in enumerate(results.results, 1):
                logger.info(f"{i}. ID: {result.id}, Score: {result.score:.4f}")
                logger.info(f"   Query: {result.query[:100]}...")

        except Exception as e:
            logger.error(f"Test failed: {str(e)}")

    async def run_batch_processing():
        """Run fetch_and_search_queries in a loop until no more data to process"""
        batch_number = 0

        while True:
            batch_number += 1
            logger.info(f"Starting batch {batch_number}...")

            try:
                results = await fetch_and_search_queries(
                    db_limit=1000, search_limit=100, score_threshold=0.6
                )

                if not results:
                    logger.info(
                        "No more queries to process. Batch processing complete."
                    )
                    break

                logger.info(
                    f"Batch {batch_number} completed: processed {len(results)} queries"
                )

            except Exception as e:
                logger.error(f"Error in batch {batch_number}: {str(e)}")
                logger.info("Stopping batch processing due to error")
                break

    # Run the batch processing
    asyncio.run(run_batch_processing())
