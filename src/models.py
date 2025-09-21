from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String
from sqlmodel import Field, SQLModel


class Query(SQLModel, table=True):
    __tablename__ = "queries"
    id: int = Field(default=None, primary_key=True)
    query: str = Field(sa_column=Column(String, nullable=False))
    has_bge_embedding: bool = Field(default=False)
    has_xml_embedding: bool = Field(default=False)
    created_at: datetime = Field(
        default=datetime.now(timezone.utc), sa_type=DateTime(timezone=True)
    )
    updated_at: datetime = Field(
        default=datetime.now(timezone.utc), sa_type=DateTime(timezone=True)
    )
    deleted_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True), nullable=True
    )
