from typing import Any

from elasticsearch import Elasticsearch


class ElasticRepository:
    def __init__(self, client: Elasticsearch):
        self._client = client

    def insert_docs(self, index_name, docs):
        operations = []
        for doc in docs:
            operations.append({"index": {"_index": index_name}})
            operations.append(doc)
        return self._client.bulk(body=operations)

    def search(
        self,
        index_name: str,
        query,
        size: int,
    ):
        res = self._client.search(
            index=index_name,
            body=query,
            size=size,
        )
        return res
