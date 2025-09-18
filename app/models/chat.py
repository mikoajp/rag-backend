from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatQuery(BaseModel):
    query: str
    collection: str = "default"
    max_sources: int = 5
    temperature: float = 0.1

class ChatSource(BaseModel):
    document_id: str
    filename: str
    page: Optional[str] = None
    content_preview: str
    similarity_score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]
    model_info: Dict[str, Any]
    processing_time: float
    tokens_used: Optional[int] = None

class StreamToken(BaseModel):
    token: str
    done: bool = False
    error: Optional[str] = None