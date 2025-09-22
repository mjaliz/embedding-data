from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.models import Query


class PostgresRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert_queries(self, queries: list[Query]) -> list[Query]:
        self.session.add_all(queries)
        await self.session.commit()
        return queries

    async def get_bge_queries_to_embed(self, offset: int, limit: int) -> list[Query]:
        result = await self.session.exec(
            select(Query)
            .where(Query.has_bge_embedding == False)
            .offset(offset)
            .limit(limit)
        )
        return result.all()

    async def get_queries_for_search(
        self, offset: int = 0, limit: int = 100
    ) -> list[Query]:
        """
        Fetch queries that have BGE embeddings for search purposes

        Args:
            offset: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of Query objects that have BGE embeddings
        """
        result = await self.session.exec(
            select(Query).where(Query.has_bge_embedding).offset(offset).limit(limit)
        )
        return result.all()

    async def get_queries_without_bge_m3_candidate(
        self, offset: int = 0, limit: int = 100
    ) -> list[Query]:
        """
        Fetch queries that don't have BGE M3 candidate flag set

        Args:
            offset: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of Query objects that have has_bge_m3_candidate as False
        """
        result = await self.session.exec(
            select(Query)
            .where(Query.has_bge_m3_candidate == False)
            .offset(offset)
            .limit(limit)
        )
        return result.all()

    async def get_query_by_id(self, query_id: int) -> Query | None:
        """
        Fetch a specific query by its ID

        Args:
            query_id: The ID of the query to fetch

        Returns:
            Query object if found, None otherwise
        """
        result = await self.session.exec(select(Query).where(Query.id == query_id))
        return result.first()
