from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class DocumentStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentUpload(BaseModel):
    filename: str
    collection: str = "default"
    metadata: Optional[Dict[str, Any]] = None

class DocumentInfo(BaseModel):
    id: str
    filename: str
    collection: str
    status: DocumentStatus
    file_size: int
    content_type: str
    chunks_count: int = 0
    created_at: datetime
    processed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentChunk(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None

class CollectionInfo(BaseModel):
    name: str
    documents_count: int
    chunks_count: int
    created_at: datetime
    size_mb: float