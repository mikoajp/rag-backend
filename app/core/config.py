from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Application
    app_name: str = "RAG Backend"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # LM Studio
    lm_studio_url: str = "http://localhost:1234"
    lm_studio_model: str = "llama-3.1-8b-instruct"

    # Vector Database
    chroma_db_path: str = "./storage/vector_db"
    embedding_model: str = "all-MiniLM-L6-v2"

    # File Storage
    upload_dir: str = "./storage/documents"
    max_file_size: int = 50  # MB
    allowed_extensions: List[str] = ["pdf", "docx", "txt", "md"]

    # Chunking configuration
    chunk_size: int = 800
    chunk_overlap: int = 200

    # Security
    secret_key: str = "your-super-secret-key-here"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"


settings = Settings()