import asyncio
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

    async def batch_search_similar_queries(
        self,
        query_texts: list[str],
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[SearchResponse]:
        """
        Search for similar queries for multiple query texts in a single batch operation

        Args:
            query_texts: List of texts to search for
            limit: Maximum number of results to return for each query
            score_threshold: Minimum similarity score threshold

        Returns:
            List of SearchResponse objects, one for each query text
        """
        start_time = time.time()

        try:
            if not query_texts:
                return []

            # Generate embeddings for all queries in batch
            logger.info(
                f"Generating embeddings for {len(query_texts)} queries in batch..."
            )
            query_embeddings = self.embedder.embed(query_texts, show_progress_bar=True)

            # Search in Milvus with batch
            logger.info(
                f"Performing batch search in Milvus collection: {self.collection_name}"
            )
            batch_search_results = self.milvus_repo.batch_search(
                collection_name=self.collection_name,
                query_vectors=query_embeddings,
                limit=limit,
                output_fields=["id", "query"],
            )

            # Process all search results
            all_responses = []

            async with get_async_session_context() as session:
                postgres_repo = PostgresRepo(session)

                for idx, (query_text, search_results) in enumerate(
                    zip(query_texts, batch_search_results)
                ):
                    results = []

                    if search_results and len(search_results) > 0:
                        for hit in search_results:
                            # Filter by score threshold
                            if hit["distance"] < score_threshold:
                                continue

                            # Get the full query from PostgreSQL
                            query_obj = await postgres_repo.get_query_by_id(hit["id"])
                            if query_obj:
                                result = SearchResult(
                                    id=hit["id"],
                                    query=query_obj.query,
                                    score=hit["distance"],
                                    distance=hit["distance"],
                                )
                                results.append(result)

                    response = SearchResponse(
                        query_text=query_text,
                        results=results,
                        total_results=len(results),
                        search_time_ms=0,  # Will be updated with average time
                    )
                    all_responses.append(response)

            # Calculate average search time per query
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(query_texts) if query_texts else 0

            # Update search times
            for response in all_responses:
                response.search_time_ms = avg_time

            logger.info(
                f"Batch search completed in {total_time:.2f}ms for {len(query_texts)} queries "
                f"(avg {avg_time:.2f}ms per query)"
            )

            return all_responses

        except Exception as e:
            logger.error(f"Error during batch semantic search: {str(e)}")
            raise

    async def _search_chunk(
        self,
        query_texts_chunk: list[str],
        embeddings_chunk: list[list[float]],
        limit: int,
        score_threshold: float,
        semaphore: asyncio.Semaphore,
        chunk_idx: int,
    ) -> list[SearchResponse]:
        """
        Search a chunk of queries concurrently with semaphore control

        Args:
            query_texts_chunk: Chunk of query texts
            embeddings_chunk: Corresponding embeddings
            limit: Maximum number of results per query
            score_threshold: Minimum similarity score
            semaphore: Semaphore to control concurrency
            chunk_idx: Index of the chunk for logging

        Returns:
            List of SearchResponse objects for this chunk
        """
        async with semaphore:
            try:
                logger.info(
                    f"Processing chunk {chunk_idx}: {len(query_texts_chunk)} queries"
                )

                # Search in Milvus for this chunk
                batch_search_results = self.milvus_repo.batch_search(
                    collection_name=self.collection_name,
                    query_vectors=embeddings_chunk,
                    limit=limit,
                    output_fields=["id", "query"],
                )

                # Process search results for this chunk
                chunk_responses = []

                async with get_async_session_context() as session:
                    postgres_repo = PostgresRepo(session)

                    for query_text, search_results in zip(
                        query_texts_chunk, batch_search_results
                    ):
                        results = []

                        if search_results and len(search_results) > 0:
                            for hit in search_results:
                                if hit["distance"] < score_threshold:
                                    continue

                                query_obj = await postgres_repo.get_query_by_id(
                                    hit["id"]
                                )
                                if query_obj:
                                    result = SearchResult(
                                        id=hit["id"],
                                        query=query_obj.query,
                                        score=hit["distance"],
                                        distance=hit["distance"],
                                    )
                                    results.append(result)

                        response = SearchResponse(
                            query_text=query_text,
                            results=results,
                            total_results=len(results),
                            search_time_ms=0,
                        )
                        chunk_responses.append(response)

                logger.info(f"Completed chunk {chunk_idx}")
                return chunk_responses

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                raise

    async def concurrent_batch_search_similar_queries(
        self,
        query_texts: list[str],
        limit: int = 10,
        score_threshold: float = 0.0,
        chunk_size: int = 100,
        max_concurrent: int = 10,
    ) -> list[SearchResponse]:
        """
        Search for similar queries using concurrent batch operations with semaphore control

        Args:
            query_texts: List of texts to search for
            limit: Maximum number of results to return for each query
            score_threshold: Minimum similarity score threshold
            chunk_size: Number of queries to process in each chunk
            max_concurrent: Maximum number of concurrent chunk operations

        Returns:
            List of SearchResponse objects, one for each query text (in original order)
        """
        start_time = time.time()

        try:
            if not query_texts:
                return []

            # Generate embeddings for all queries in batch (this is still fast)
            logger.info(
                f"Generating embeddings for {len(query_texts)} queries in batch..."
            )
            query_embeddings = self.embedder.embed(query_texts, show_progress_bar=True)

            # Split queries and embeddings into chunks
            chunks = []
            for i in range(0, len(query_texts), chunk_size):
                text_chunk = query_texts[i : i + chunk_size]
                embedding_chunk = query_embeddings[i : i + chunk_size]
                chunks.append((text_chunk, embedding_chunk))

            logger.info(
                f"Split {len(query_texts)} queries into {len(chunks)} chunks "
                f"(chunk_size={chunk_size}, max_concurrent={max_concurrent})"
            )

            # Create semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(max_concurrent)

            # Process chunks concurrently
            logger.info(
                f"Starting concurrent batch search with {max_concurrent} concurrent workers..."
            )
            tasks = [
                self._search_chunk(
                    query_texts_chunk=text_chunk,
                    embeddings_chunk=embedding_chunk,
                    limit=limit,
                    score_threshold=score_threshold,
                    semaphore=semaphore,
                    chunk_idx=idx,
                )
                for idx, (text_chunk, embedding_chunk) in enumerate(chunks)
            ]

            # Gather all results
            chunk_results = await asyncio.gather(*tasks)

            # Flatten results to maintain original order
            all_responses = []
            for chunk_response in chunk_results:
                all_responses.extend(chunk_response)

            # Calculate timing
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(query_texts) if query_texts else 0

            # Update search times
            for response in all_responses:
                response.search_time_ms = avg_time

            logger.info(
                f"Concurrent batch search completed in {total_time:.2f}ms "
                f"for {len(query_texts)} queries "
                f"(avg {avg_time:.2f}ms per query)"
            )

            return all_responses

        except Exception as e:
            logger.error(f"Error during concurrent batch semantic search: {str(e)}")
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
    search_service,
    db_limit: int = 1000,
    search_limit: int = 100,
    score_threshold: float = 0.6,
    chunk_size: int = 100,
    max_concurrent: int = 10,
) -> list[SearchResponse]:
    """
    Fetches queries from DB where has_bge_m3_candidate is False and performs concurrent batch search
    This function finds similar queries for each fetched query using optimized concurrent batch operations

    Args:
        search_service: The search service instance to use
        db_limit: Number of queries to fetch from database
        search_limit: Maximum number of similar results to return for each query
        score_threshold: Minimum similarity score threshold
        chunk_size: Number of queries to process in each concurrent chunk
        max_concurrent: Maximum number of concurrent chunk operations

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

        # Perform concurrent batch search for all queries
        query_texts = [query.query for query in queries]

        logger.info(
            f"Starting concurrent batch search for {len(query_texts)} queries..."
        )
        results = await search_service.concurrent_batch_search_similar_queries(
            query_texts=query_texts,
            limit=search_limit,
            score_threshold=score_threshold,
            chunk_size=chunk_size,
            max_concurrent=max_concurrent,
        )

        # Prepare batch updates for database
        updates = []
        for query, search_result in zip(queries, results):
            # Extract candidate queries from search results
            bge_m3_candidates = [result.query for result in search_result.results]

            updates.append(
                {
                    "query_id": query.id,
                    "bge_m3_candidates": bge_m3_candidates,
                    "has_bge_m3_candidate": True,
                }
            )

        # Perform batch update of all queries in a single transaction
        logger.info(f"Updating {len(updates)} queries with their candidates...")
        async with get_async_session_context() as session:
            postgres_repo = PostgresRepo(session)
            await postgres_repo.batch_update_query_candidates(updates)

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"Completed concurrent batch processing for {len(results)} queries in {total_time:.2f}ms "
            f"(avg {total_time / len(results):.2f}ms per query)"
        )

        return results

    except Exception as e:
        logger.error(f"Error during fetch and search: {str(e)}")
        raise


if __name__ == "__main__":

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
        search_service = SemanticSearchService()
        while True:
            batch_number += 1
            logger.info(f"Starting batch {batch_number}...")

            try:
                results = await fetch_and_search_queries(
                    search_service=search_service,
                    db_limit=1000,
                    search_limit=100,
                    score_threshold=0.6,
                    chunk_size=50,
                    max_concurrent=20,
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
