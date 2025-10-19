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

    async def update_query_candidates(
        self,
        query_id: int,
        bge_m3_candidates: list[str] | None = None,
        xml_candidates: list[str] | None = None,
        has_bge_m3_candidate: bool | None = None,
        has_xml_candidate: bool | None = None,
    ) -> Query | None:
        """
        Update query candidates and their flags

        Args:
            query_id: The ID of the query to update
            bge_m3_candidates: List of BGE M3 candidate strings
            xml_candidates: List of XML candidate strings
            has_bge_m3_candidate: Flag to indicate if BGE M3 candidates are set
            has_xml_candidate: Flag to indicate if XML candidates are set

        Returns:
            Updated Query object if found, None otherwise
        """
        query = await self.get_query_by_id(query_id)
        if not query:
            return None

        if bge_m3_candidates is not None:
            query.bge_m3_candidates = bge_m3_candidates
        if xml_candidates is not None:
            query.xml_candidates = xml_candidates
        if has_bge_m3_candidate is not None:
            query.has_bge_m3_candidate = has_bge_m3_candidate
        if has_xml_candidate is not None:
            query.has_xml_candidate = has_xml_candidate

        self.session.add(query)
        await self.session.commit()
        await self.session.refresh(query)
        return query

    async def batch_update_query_candidates(
        self,
        updates: list[dict],
    ) -> list[Query]:
        """
        Batch update multiple queries with their candidates and flags

        Args:
            updates: List of dicts containing:
                - query_id: The ID of the query to update
                - bge_m3_candidates: Optional list of BGE M3 candidate strings
                - xml_candidates: Optional list of XML candidate strings
                - has_bge_m3_candidate: Optional flag for BGE M3 candidates
                - has_xml_candidate: Optional flag for XML candidates

        Returns:
            List of updated Query objects
        """
        updated_queries = []

        for update in updates:
            query_id = update.get("query_id")
            query = await self.get_query_by_id(query_id)

            if not query:
                continue

            if "bge_m3_candidates" in update:
                query.bge_m3_candidates = update["bge_m3_candidates"]
            if "xml_candidates" in update:
                query.xml_candidates = update["xml_candidates"]
            if "has_bge_m3_candidate" in update:
                query.has_bge_m3_candidate = update["has_bge_m3_candidate"]
            if "has_xml_candidate" in update:
                query.has_xml_candidate = update["has_xml_candidate"]

            self.session.add(query)
            updated_queries.append(query)

        await self.session.commit()

        for query in updated_queries:
            await self.session.refresh(query)

        return updated_queries
