from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, String
from sqlmodel import Field, SQLModel


class Query(SQLModel, table=True):
    __tablename__ = "queries"
    id: int = Field(default=None, primary_key=True)
    query: str = Field(sa_column=Column(String, nullable=False))
    has_bge_embedding: bool = Field(default=False)
    has_bge_m3_candidate: bool = Field(default=False)
    has_xml_embedding: bool = Field(default=False)
    has_xml_candidate: bool = Field(default=False)
    created_at: datetime = Field(
        default=datetime.now(timezone.utc), sa_type=DateTime(timezone=True)
    )
    updated_at: datetime = Field(
        default=datetime.now(timezone.utc), sa_type=DateTime(timezone=True)
    )
    deleted_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True), nullable=True
    )


class SearchResult(BaseModel):
    """Model for search results from Milvus"""

    id: int
    query: str
    score: float
    distance: float


class SearchResponse(BaseModel):
    """Response model for semantic search"""

    query_text: str
    results: list[SearchResult]
    total_results: int
    search_time_ms: float
