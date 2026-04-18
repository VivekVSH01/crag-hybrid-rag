from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # API Keys
    gemini_api_key: str
    tavily_api_key: str
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "crag_documents"
    
    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gemini-1.5-flash"
    embedding_dimensions: int = 384
    
    # CRAG Settings
    crag_relevance_threshold: float = 0.7
    crag_ambiguous_threshold: float = 0.5
    
    # Retrieval
    top_k_results: int = 5

    # Hybrid Search Settings
    hybrid_search_enabled: bool = True
    sparse_vector_enabled: bool = True
    rrf_k: int = 60

    # Reranking Settings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_initial_top_k: int = 3
    reranking_enabled_by_default: bool = False

    # Reranking Backend Selection
    reranker_backend: Literal["local", "voyage"] = "local"
    voyage_api_key: str | None = None
    voyage_model: str = "rerank-2.5"

    # Upload
    upload_dir: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
