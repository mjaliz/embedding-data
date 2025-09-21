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
        return await self.session.exec(
            select(Query)
            .where(Query.has_bge_embedding == False)
            .offset(offset)
            .limit(limit)
        ).all()
