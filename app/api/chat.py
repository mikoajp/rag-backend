from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import json
import logging

from ..models.chat import ChatQuery, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

# Global RAG service (will be initialized in main.py)
rag_service = None


def set_rag_service(service):
    global rag_service
    rag_service = service


logger = logging.getLogger(__name__)


@router.post("/query", response_model=ChatResponse)
async def query_documents(query_request: ChatQuery):
    """
    Ask a question grounded in documents (standard response)

    Args:
        query_request: Query parameters

    Returns:
        ChatResponse: AI answer with sources
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        logger.info(f"Processing query: '{query_request.query}' in collection: '{query_request.collection}'")

        response = await rag_service.query(
            query=query_request.query,
            collection=query_request.collection,
            max_sources=query_request.max_sources,
            temperature=query_request.temperature
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_query(query_request: ChatQuery):
    """
    Ask a question grounded in documents (streaming answer)

    Args:
        query_request: Query parameters

    Returns:
        StreamingResponse: Server-sent events with streaming answer
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        logger.info(f"Processing streaming query: '{query_request.query}' in collection: '{query_request.collection}'")

        async def generate_stream():
            try:
                async for chunk in rag_service.stream_query(
                        query=query_request.query,
                        collection=query_request.collection,
                        max_sources=query_request.max_sources,
                        temperature=query_request.temperature
                ):
                    # Format as Server-Sent Events
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"Error in streaming query: {e}")
                error_chunk = {"type": "error", "error": str(e)}
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

    except Exception as e:
        logger.error(f"Error setting up streaming query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/preview")
async def preview_collection(collection_name: str, limit: int = 10):
    """
    Podgląd pierwszych kilku chunks z kolekcji

    Args:
        collection_name: Nazwa kolekcji
        limit: Liczba chunks do zwrócenia (domyślnie 10)

    Returns:
        A list of sample chunks from the collection
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        # Simple query to get some chunks
        chunks = await rag_service.vector_service.similarity_search(
            query="przykład",  # Generic query
            collection_name=collection_name,
            k=limit
        )

        preview_data = []
        for chunk in chunks:
            preview_data.append({
                "id": chunk.id,
                "document_id": chunk.document_id,
                "filename": chunk.metadata.get("filename", "Unknown"),
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "chunk_index": chunk.chunk_index
            })

        return {
            "collection": collection_name,
            "chunks_count": len(preview_data),
            "preview": preview_data
        }

    except Exception as e:
        logger.error(f"Error previewing collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_llm():
    """
    Test endpoint to verify LM Studio is running

    Returns:
        Test response from LM Studio
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        # Simple test message
        test_messages = [
            {"role": "user", "content": "Napisz krótko: 'Test LM Studio działa!'"}
        ]

        response = await rag_service.llm_service.generate_response(
            messages=test_messages,
            temperature=0.1,
            max_tokens=50
        )

        return {
            "status": "success",
            "llm_response": response["content"],
            "model": response.get("model", "unknown"),
            "tokens_used": response.get("usage", {}).get("total_tokens", 0)
        }

    except Exception as e:
        logger.error(f"Error testing LLM: {e}")
        raise HTTPException(status_code=500, detail=f"LM Studio test failed: {str(e)}")