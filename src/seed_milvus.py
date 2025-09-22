import asyncio

from loguru import logger
from sqlmodel import update

from src.config import config
from src.database import get_async_session_context
from src.embedding import Embedder
from src.milvus_repo import MilvusRepo
from src.models import Query
from src.postgres_repo import PostgresRepo


async def fetch_and_embed_queries(
    batch_size: int = 100,
    embedding_model: str = "BAAI/bge-m3",
    collection_name: str = "bge_m3_embeddings",
):
    """
    Fetch queries from PostgreSQL that need BGE embeddings, generate embeddings,
    insert to Milvus, and update PostgreSQL records.

    Args:
        batch_size: Number of queries to process in each batch
        embedding_model: Name of the embedding model to use
        collection_name: Name of the Milvus collection to insert embeddings
    """
    # Initialize components
    embedder = Embedder(embedding_model)
    milvus_repo = MilvusRepo(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)

    # Get embedding dimension (test with a sample text)
    sample_embedding = embedder.embed(["test"], show_progress_bar=False)
    embedding_dimension = len(sample_embedding[0])
    logger.info(f"Embedding dimension: {embedding_dimension}")

    # Create or ensure Milvus collection exists
    milvus_repo.create_collection(
        collection_name=collection_name,
        dimension=embedding_dimension,
        drop_if_exists=False,  # Don't drop existing collection
    )

    offset = 0
    total_processed = 0

    async with get_async_session_context() as session:
        postgres_repo = PostgresRepo(session)

        while True:
            # Fetch batch of queries that need embeddings
            queries = await postgres_repo.get_bge_queries_to_embed(offset, batch_size)

            if not queries:
                logger.info(
                    f"No more queries to process. Total processed: {total_processed}"
                )
                break

            logger.info(
                f"Processing batch of {len(queries)} queries (offset: {offset})"
            )

            try:
                # Extract query texts and IDs
                query_texts = [query.query for query in queries]
                query_ids = [query.id for query in queries]

                # Generate embeddings
                logger.info("Generating embeddings...")
                embeddings = embedder.embed(query_texts, show_progress_bar=True)

                # Prepare data for Milvus insertion
                milvus_data = []
                for i, (query_id, embedding) in enumerate(zip(query_ids, embeddings)):
                    milvus_data.append(
                        {
                            "id": query_id,  # Use PostgreSQL query ID as Milvus ID
                            "query_text": query_texts[i],
                            "vector": embedding,
                        }
                    )

                # Insert to Milvus
                logger.info("Inserting embeddings to Milvus...")
                milvus_result = milvus_repo.insert_data(collection_name, milvus_data)
                logger.info(f"Milvus insertion result: {milvus_result}")

                # Update PostgreSQL records to mark as embedded
                logger.info("Updating PostgreSQL records...")
                stmt = (
                    update(Query)
                    .where(Query.id.in_(query_ids))
                    .values(has_bge_embedding=True)
                )
                await session.exec(stmt)
                await session.commit()

                total_processed += len(queries)
                logger.info(
                    f"Successfully processed batch. Total processed so far: {total_processed}"
                )

            except Exception as e:
                logger.error(f"Error processing batch at offset {offset}: {str(e)}")
                # Rollback the session in case of error
                await session.rollback()
                # Continue with next batch or break depending on your error handling strategy
                break

            offset += batch_size


async def get_embedding_stats(collection_name: str = "bge_m3_embeddings"):
    """
    Get statistics about the current state of embeddings

    Args:
        collection_name: Name of the Milvus collection
    """
    milvus_repo = MilvusRepo(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)

    async with get_async_session_context() as session:
        # Get count of queries with BGE embeddings
        from sqlmodel import func, select

        result = await session.exec(
            select(func.count(Query.id)).where(Query.has_bge_embedding == True)
        )
        embedded_count = result.one()

        # Get count of queries without BGE embeddings
        result = await session.exec(
            select(func.count(Query.id)).where(Query.has_bge_embedding == False)
        )
        not_embedded_count = result.one()

        # Get total count
        result = await session.exec(select(func.count(Query.id)))
        total_count = result.one()

        logger.info("Embedding Statistics:")
        logger.info(f"  Total queries: {total_count}")
        logger.info(f"  With BGE embeddings: {embedded_count}")
        logger.info(f"  Without BGE embeddings: {not_embedded_count}")
        logger.info(f"  Progress: {(embedded_count / total_count) * 100:.2f}%")

        # Check Milvus collection info if it exists
        if milvus_repo.client.has_collection(collection_name):
            collection_info = milvus_repo.client.describe_collection(collection_name)
            logger.info(f"  Milvus collection '{collection_name}' exists")
            logger.info(f"  Collection info: {collection_info}")
        else:
            logger.info(f"  Milvus collection '{collection_name}' does not exist")


async def main():
    """Main function to run the embedding pipeline"""
    logger.info("Starting BGE embedding pipeline...")

    # Show initial stats
    await get_embedding_stats()

    # Run the embedding pipeline
    await fetch_and_embed_queries(
        batch_size=10_000,
        embedding_model="BAAI/bge-m3",
        collection_name="bge_m3_embeddings",
    )

    # Show final stats
    logger.info("BGE embedding pipeline completed!")
    await get_embedding_stats()


if __name__ == "__main__":
    asyncio.run(main())
