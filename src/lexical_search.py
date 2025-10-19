import asyncio
import time

from elasticsearch import Elasticsearch
from loguru import logger

from src.config import config
from src.database import get_async_session_context
from src.elastic_repo import ElasticRepository
from src.models import SearchResponse, SearchResult
from src.postgres_repo import PostgresRepo


class LexicalSearchService:
    """Service for performing lexical search on queries using Elasticsearch"""

    def __init__(self):
        self.elastic_client = Elasticsearch(
            hosts=[config.ELASTIC_STAGE_HOST_URL],
            verify_certs=False,
            ssl_show_warn=False,
        )
        self.elastic_repo = ElasticRepository(self.elastic_client)
        self.index_name = config.ELASTIC_STAGE_INDEX_NAME

    def _build_match_query(self, query_text: str, field: str = "keyword") -> dict:
        """Build a simple match query for Elasticsearch"""
        return {"query": {"match": {field: {"query": query_text, "operator": "and"}}}}

    def _build_fuzzy_query(self, query_text: str, field: str = "keyword") -> dict:
        """Build a fuzzy query for Elasticsearch"""
        return {"query": {"fuzzy": {field: {"value": query_text, "fuzziness": "AUTO"}}}}

    def _build_multi_match_query(
        self, query_text: str, fields: list[str] = None
    ) -> dict:
        """Build a multi-match query for Elasticsearch"""
        if fields is None:
            fields = ["keyword", "query"]

        return {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            }
        }

    def _build_combined_query(self, query_text: str, field: str = "keyword") -> dict:
        """Build a combined query with match and fuzzy search"""
        return {
            "query": {
                "bool": {
                    "should": [
                        {"match": {field: {"query": query_text, "boost": 2.0}}},
                        {
                            "fuzzy": {
                                field: {
                                    "value": query_text,
                                    "fuzziness": "AUTO",
                                    "boost": 1.0,
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
        }

    async def search_similar_queries(
        self,
        query_text: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        search_type: str = "combined",
    ) -> SearchResponse:
        """
        Search for similar queries using lexical search

        Args:
            query_text: The text to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            search_type: Type of search ('match', 'fuzzy', 'multi_match', 'combined')

        Returns:
            SearchResponse containing the search results
        """
        start_time = time.time()

        try:
            # Build query based on search type
            if search_type == "match":
                query = self._build_match_query(query_text)
            elif search_type == "fuzzy":
                query = self._build_fuzzy_query(query_text)
            elif search_type == "multi_match":
                query = self._build_multi_match_query(query_text)
            else:  # combined
                query = self._build_combined_query(query_text)

            logger.info(
                f"Performing {search_type} lexical search for: {query_text[:100]}..."
            )

            # Search in Elasticsearch
            elastic_res = await asyncio.to_thread(
                self.elastic_repo.search, self.index_name, query, limit
            )

            # Process search results
            results = []
            if elastic_res and "hits" in elastic_res and "hits" in elastic_res["hits"]:
                async with get_async_session_context() as session:
                    postgres_repo = PostgresRepo(session)

                    for hit in elastic_res["hits"]["hits"]:
                        # Get score from Elasticsearch (0-1 scale, higher is better)
                        score = hit.get("_score", 0.0)

                        # Filter by score threshold
                        if score < score_threshold:
                            continue

                        # Get the full query from PostgreSQL using the keyword
                        keyword = hit["_source"].get("keyword", "")
                        if keyword:
                            query_obj = await postgres_repo.get_query_by_keyword(
                                keyword
                            )
                            if query_obj:
                                result = SearchResult(
                                    id=query_obj.id,
                                    query=query_obj.query,
                                    score=score,
                                    distance=1.0 - score,  # Convert score to distance
                                )
                                results.append(result)

            search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            logger.info(
                f"Lexical search completed in {search_time:.2f}ms, found {len(results)} results"
            )

            return SearchResponse(
                query_text=query_text,
                results=results,
                total_results=len(results),
                search_time_ms=search_time,
            )

        except Exception as e:
            logger.error(f"Error during lexical search: {str(e)}")
            raise

    async def search_queries_by_keyword(
        self, keyword: str, limit: int = 10, score_threshold: float = 0.0
    ) -> SearchResponse:
        """
        Search for queries by exact keyword match

        Args:
            keyword: The keyword to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            SearchResponse containing the search results
        """
        start_time = time.time()

        try:
            query = {"query": {"term": {"keyword": keyword}}}

            logger.info(f"Searching by keyword: {keyword}")

            # Search in Elasticsearch
            elastic_res = await asyncio.to_thread(
                self.elastic_repo.search, self.index_name, query, limit
            )

            # Process search results
            results = []
            if elastic_res and "hits" in elastic_res and "hits" in elastic_res["hits"]:
                async with get_async_session_context() as session:
                    postgres_repo = PostgresRepo(session)

                    for hit in elastic_res["hits"]["hits"]:
                        score = hit.get("_score", 1.0)  # Exact match gets high score

                        if score < score_threshold:
                            continue

                        keyword = hit["_source"].get("keyword", "")
                        if keyword:
                            query_obj = await postgres_repo.get_query_by_keyword(
                                keyword
                            )
                            if query_obj:
                                result = SearchResult(
                                    id=query_obj.id,
                                    query=query_obj.query,
                                    score=score,
                                    distance=1.0 - score,
                                )
                                results.append(result)

            search_time = (time.time() - start_time) * 1000

            logger.info(
                f"Keyword search completed in {search_time:.2f}ms, found {len(results)} results"
            )

            return SearchResponse(
                query_text=keyword,
                results=results,
                total_results=len(results),
                search_time_ms=search_time,
            )

        except Exception as e:
            logger.error(f"Error during keyword search: {str(e)}")
            raise


async def search_queries_lexically(
    query_text: str,
    limit: int = 10,
    score_threshold: float = 0.0,
    search_type: str = "combined",
) -> SearchResponse:
    """
    Convenience function to perform lexical search

    Args:
        query_text: The text to search for
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score threshold
        search_type: Type of search ('match', 'fuzzy', 'multi_match', 'combined')

    Returns:
        SearchResponse containing the search results
    """
    search_service = LexicalSearchService()
    return await search_service.search_similar_queries(
        query_text=query_text,
        limit=limit,
        score_threshold=score_threshold,
        search_type=search_type,
    )


async def fetch_and_search_queries_lexically(
    db_limit: int = 1000,
    search_limit: int = 100,
    score_threshold: float = 0.6,
    search_type: str = "combined",
) -> list[SearchResponse]:
    """
    Fetches queries from DB where has_elastic_candidate is False and performs lexical search for each one

    Args:
        db_limit: Number of queries to fetch from database
        search_limit: Maximum number of similar results to return for each query
        score_threshold: Minimum similarity score threshold
        search_type: Type of search ('match', 'fuzzy', 'multi_match', 'combined')

    Returns:
        List of SearchResponse objects, one for each fetched query
    """
    start_time = time.time()

    try:
        # Fetch queries from database where has_elastic_candidate is False
        async with get_async_session_context() as session:
            postgres_repo = PostgresRepo(session)
            queries = await postgres_repo.get_queries_without_elastic_candidate(
                offset=0, limit=db_limit
            )

        if not queries:
            logger.warning(
                "No queries found in database without elastic candidate flag"
            )
            return []

        logger.info(
            f"Fetched {len(queries)} queries without elastic candidate from database"
        )

        # Search for similar queries for each fetched query
        search_service = LexicalSearchService()
        results = []

        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query.query[:50]}...")

            try:
                search_result = await search_service.search_similar_queries(
                    query_text=query.query,
                    limit=search_limit,
                    score_threshold=score_threshold,
                    search_type=search_type,
                )
                results.append(search_result)

                # Extract candidate queries from search results
                elastic_candidates = [result.query for result in search_result.results]

                # Update the query with candidates and set the flag
                async with get_async_session_context() as session:
                    postgres_repo = PostgresRepo(session)
                    await postgres_repo.update_query_elastic_candidates(
                        query_id=query.id,
                        elastic_candidates=elastic_candidates,
                        has_elastic_candidate=True,
                    )

                logger.info(
                    f"Updated query {query.id} with {len(elastic_candidates)} elastic candidates"
                )

            except Exception as e:
                logger.error(f"Error searching for query ID {query.id}: {str(e)}")
                continue

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"Completed lexical searching for {len(results)} queries in {total_time:.2f}ms"
        )

        return results

    except Exception as e:
        logger.error(f"Error during fetch and search: {str(e)}")
        raise


# Legacy functions for backward compatibility
def _elastic_candidates(self, q: str, limit: int):
    """Legacy function - kept for backward compatibility"""
    query = {
        "query": {
            "match": {"keyword": q},
        },
    }
    elastic_res = self.elastic_repo.search(self.index_name, query, size=limit)
    return elastic_res


async def elastic_candidates(self, q: str, limit: int):
    """Legacy function - kept for backward compatibility"""
    elastic_res = await asyncio.to_thread(self._elastic_candidates, q, limit)
    elastic_candidates = [
        hit["_source"]["keyword"] for hit in elastic_res["hits"]["hits"]
    ]
    return elastic_candidates


if __name__ == "__main__":
    import asyncio

    async def test_lexical_search():
        """Test function for lexical search"""
        test_query = "سینی پنج تکه مهره دار"

        logger.info(f"Testing lexical search with query: '{test_query}'")

        try:
            # Test different search types
            search_types = ["match", "fuzzy", "multi_match", "combined"]

            for search_type in search_types:
                logger.info(f"\n--- Testing {search_type} search ---")
                results = await search_queries_lexically(
                    query_text=test_query,
                    limit=10,
                    score_threshold=0.5,
                    search_type=search_type,
                )

                logger.info(
                    f"Found {results.total_results} results with {search_type} search:"
                )
                for i, result in enumerate(results.results, 1):
                    logger.info(f"{i}. ID: {result.id}, Score: {result.score:.4f}")
                    logger.info(f"   Query: {result.query[:100]}...")

        except Exception as e:
            logger.error(f"Test failed: {str(e)}")

    async def run_batch_processing():
        """Run fetch_and_search_queries_lexically in a loop until no more data to process"""
        batch_number = 0

        while True:
            batch_number += 1
            logger.info(f"Starting lexical search batch {batch_number}...")

            try:
                results = await fetch_and_search_queries_lexically(
                    db_limit=1000,
                    search_limit=100,
                    score_threshold=0.6,
                    search_type="combined",
                )

                if not results:
                    logger.info(
                        "No more queries to process. Lexical search batch processing complete."
                    )
                    break

                logger.info(
                    f"Batch {batch_number} completed: processed {len(results)} queries"
                )

            except Exception as e:
                logger.error(f"Error in batch {batch_number}: {str(e)}")
                logger.info("Stopping batch processing due to error")
                break

    # Run the test
    asyncio.run(test_lexical_search())
