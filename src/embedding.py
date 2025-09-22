from loguru import logger
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name
        logger.info(f"Initializing embedding model with {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, text: list[str], show_progress_bar: bool = True):
        # Implement embedding logic here
        embedding = self.model.encode(
            text,
            batch_size=64,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embedding.tolist()
