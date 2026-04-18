from sentence_transformers import SentenceTransformer
from app.config import get_settings
from loguru import logger


class EmbeddingService:
    def __init__(self):
        self.settings = get_settings()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text(self, text: str) -> list[float]:
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        try:
            return self.model.encode(texts, batch_size=batch_size).tolist()
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            raise
