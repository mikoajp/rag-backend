from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

# Import API routers
from .api import health, documents, chat

# Import services
from .services.rag_service import RAGService
from .core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_system.log')
    ]
)

logger = logging.getLogger(__name__)

# Global RAG service instance
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("Starting RAG Backend...")

    try:
        # Initialize RAG service
        global rag_service
        rag_service = RAGService()

        # Set RAG service in API modules
        health.set_rag_service(rag_service)
        documents.set_rag_service(rag_service)
        chat.set_rag_service(rag_service)

        # Create storage directories
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.chroma_db_path).mkdir(parents=True, exist_ok=True)

        # Test connections
        await startup_checks()

        logger.info("RAG Backend started successfully!")

    except Exception as e:
        logger.error(f"Failed to start RAG Backend: {e}")
        raise e

    yield

    # Shutdown
    logger.info("Shutting down RAG Backend...")
    if rag_service:
        await rag_service.cleanup()
    logger.info("RAG Backend shutdown complete")


async def startup_checks():
    """Perform startup health checks"""
    logger.info("Performing startup checks...")

    try:
        # Check LM Studio connection
        llm_status = await rag_service.llm_service.check_server_status()
        if llm_status:
            logger.info("✅ LM Studio connection: OK")
        else:
            logger.warning("⚠️  LM Studio connection: FAILED - Make sure LM Studio is running on localhost:1234")

        # Check Vector DB
        vector_status = await rag_service.vector_service.health_check()
        if vector_status:
            logger.info("✅ Vector Database: OK")
        else:
            logger.error("❌ Vector Database: FAILED")

        # Check storage directories
        if Path(settings.upload_dir).exists():
            logger.info("✅ Upload directory: OK")
        else:
            logger.error("❌ Upload directory: FAILED")

    except Exception as e:
        logger.error(f"Startup checks failed: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    🧠 RAG Backend API

    Retrieval Augmented Generation system with a local LLM (LM Studio)

    ## Features
    - 📄 Upload and process documents (PDF, DOCX, TXT, MD)
    - 🔍 Semantic search across documents
    - 💬 Chat interface grounded in your documents
    - 📊 Streaming responses (ChatGPT-like)
    - 🏠 Full privacy (local LLM)

    ## How to use
    1. Run LM Studio with Llama 3.1 8B model on port 1234
    2. Upload documents via `/documents/upload`
    3. Ask questions via `/chat/query` or `/chat/stream`

    ## Status
    Check system status at `/health/`
    """,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Welcome endpoint with basic information
    """
    return {
        "message": "🧠 RAG Backend API",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health/",
        "endpoints": {
            "upload_document": "POST /documents/upload",
            "list_documents": "GET /documents/",
            "query_documents": "POST /chat/query",
            "stream_query": "POST /chat/stream",
            "health_check": "GET /health/"
        },
        "requirements": {
            "lm_studio": "Running on localhost:1234",
            "model": "Llama 3.1 8B Instruct or compatible"
        }
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    # Przepuszczaj HTTPException z właściwym kodem i komunikatem
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    # Dla pozostałych wyjątków zwróć standardowe 500
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"LM Studio URL: {settings.lm_studio_url}")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )