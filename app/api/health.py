from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/health", tags=["health"])

# Global RAG service instance (will be initialized in main.py)
rag_service = None


def set_rag_service(service):
    """Set RAG service instance"""
    global rag_service
    rag_service = service


@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """
    Check overall RAG system status

    Returns:
        - status: healthy/unhealthy
        - components: status of each component
        - model_info: information about the LLM
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        health_info = await rag_service.health_check()

        # Return appropriate HTTP status
        if health_info["status"] == "unhealthy":
            raise HTTPException(status_code=503, detail=health_info)

        return health_info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/lm-studio")
async def check_lm_studio():
    """Checks only LM Studio status"""
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        status = await rag_service.llm_service.check_server_status()

        return {
            "lm_studio_running": status,
            "endpoint": rag_service.llm_service.base_url,
            "model": rag_service.llm_service.model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector-db")
async def check_vector_db():
    """Checks vector database status"""
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        status = await rag_service.vector_service.health_check()
        collections = await rag_service.vector_service.get_collections()

        return {
            "vector_db_running": status,
            "collections_count": len(collections),
            "collections": [col["name"] for col in collections]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))