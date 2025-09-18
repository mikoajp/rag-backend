from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from ..models.document import DocumentInfo, CollectionInfo
from ..core.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])

# Global services (will be initialized in main.py)
rag_service = None


def set_rag_service(service):
    global rag_service
    rag_service = service


logger = logging.getLogger(__name__)


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
        file: UploadFile = File(...),
        collection: str = Form("default"),
        metadata: Optional[str] = Form(
            None,
            description="Optional JSON string with metadata, e.g. {\"source\":\"user\", \"tags\":[\"cv\"]}"
        )  # JSON string
):
    """
    Upload a document to the RAG system

    Args:
        file: File to upload (PDF, DOCX, TXT, MD)
        collection: Collection name (default 'default')
        metadata: Additional metadata as JSON string

    Returns:
        DocumentInfo: Information about the uploaded document
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Check file extension
        file_extension = Path(file.filename).suffix.lower().replace('.', '')
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type .{file_extension} not allowed. Allowed: {', '.join(settings.allowed_extensions)}"
            )

        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)

        if file_size_mb > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_file_size}MB, your file: {file_size_mb:.1f}MB"
            )

        # Parse metadata if provided
        metadata_dict = None
        if metadata:
            import json
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata field")

        # Upload document
        doc_info = await rag_service.document_service.upload_document(
            file_content=file_content,
            filename=file.filename,
            collection=collection,
            metadata=metadata_dict
        )

        # Process document for RAG (async)
        import asyncio
        asyncio.create_task(
            rag_service.process_document_for_rag(doc_info.id, collection)
        )

        logger.info(f"Document uploaded: {file.filename} (ID: {doc_info.id})")
        return doc_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[DocumentInfo])
async def list_documents(collection: Optional[str] = None):
    """
    List all documents

    Args:
        collection: Filtruj po kolekcji (opcjonalnie)

    Returns:
        List of documents
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        documents = await rag_service.document_service.list_documents(collection)
        return documents

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """
    Get information about a specific document

    Args:
        document_id: ID dokumentu

    Returns:
        DocumentInfo: Document information
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        document = await rag_service.document_service.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the system

    Args:
        document_id: ID dokumentu do usunięcia

    Returns:
        Potwierdzenie usunięcia
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        success = await rag_service.document_service.delete_document(document_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"message": f"Document {document_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/", response_model=List[Dict[str, Any]])
async def list_collections():
    """
    List all document collections

    Returns:
        A list of collections with chunk counts
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        collections = await rag_service.get_collections()
        return collections

    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{collection_name}")
async def create_collection(collection_name: str):
    """
    Create a new document collection

    Args:
        collection_name: Nazwa nowej kolekcji

    Returns:
        Creation confirmation
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        success = await rag_service.vector_service.create_collection(collection_name)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to create collection")

        return {"message": f"Collection '{collection_name}' created successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a document collection

    Args:
        collection_name: Nazwa kolekcji do usunięcia

    Returns:
        Potwierdzenie usunięcia
    """
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")

        success = await rag_service.vector_service.delete_collection(collection_name)

        if not success:
            raise HTTPException(status_code=404, detail="Collection not found")

        return {"message": f"Collection '{collection_name}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))