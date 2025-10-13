import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from src.config import config


def json_serializer(obj):
    return json.dumps(obj, ensure_ascii=False)


engine = AsyncEngine(
    create_engine(
        url=str(config.POSTGRES_URL),
        echo=False,
        future=True,
        json_serializer=json_serializer,
    )
)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for async database sessions"""
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@asynccontextmanager
async def get_async_session_context():
    """Async context manager for database sessions outside of FastAPI"""
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


async def get_milvus_client() -> MilvusClient:
    return MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)
