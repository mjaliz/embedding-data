from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )
    MILVUS_URI: str
    MILVUS_TOKEN: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int
    POSTGRES_HOST: str
    POSTGRES_URL: str

    ELASTIC_STAGE_HOST_URL: str = "https://stage-elastic.bs0.basalam.dev:443"
    ELASTIC_STAGE_INDEX_NAME: str = "search_queries"
    EMBEDDING_URL: str = "http://185.13.230.203:8002/serach/vector_search"


config = Config()
