import numpy as np
from loguru import logger
from pymilvus import MilvusClient

from src.config import config


class MilvusRepo:
    def __init__(self, uri: str, token: str):
        self.client: MilvusClient = MilvusClient(uri=uri, token=token)

    def create_collection(
        self, collection_name: str, dimension: int, drop_if_exists: bool = True
    ):
        if self.client.has_collection(collection_name=collection_name):
            if drop_if_exists:
                logger.info(f"Dropping collection {collection_name}")
                self.client.drop_collection(collection_name=collection_name)
            else:
                logger.info(f"Collection {collection_name} already exists")
                return
        self.client.create_collection(
            collection_name=collection_name, dimension=dimension
        )

    def insert_data(self, collection_name: str, data):
        res = self.client.insert(collection_name=collection_name, data=data)
        return res


if __name__ == "__main__":
    client = MilvusRepo(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)
    client.create_collection(collection_name="test", dimension=128)
    res = client.insert_data(collection_name="test", data=np.random.rand(100, 128))
    print(res)
