# RAG Backend API

Retrieval-Augmented Generation backend with a local LLM (LM Studio) and a vector database (ChromaDB). Upload documents, index them into a vector store, and query with grounded answers.

## Features
- Upload and process documents (PDF, DOCX, TXT, MD)
- Semantic search across documents (ChromaDB + SentenceTransformers)
- Chat endpoints (regular and streaming)
- Local LLM integration via LM Studio API
- Configurable chunking (size/overlap)
- Health checks for all components
<img width="1470" height="751" alt="Zrzut ekranu 2025-09-18 o 15 45 08" src="https://github.com/user-attachments/assets/2c66240a-bdf8-4c1f-b9e5-030b4d5f15a7" />

## Architecture
- FastAPI application (app/main.py)
- API routers in `app/api`: health, documents, chat
- Services in `app/services`:
  - `document_service.py`: file I/O, text extraction, chunking (per-page for PDF), in-memory metadata
  - `vector_service.py`: ChromaDB client, embeddings, similarity search
  - `llm_service.py`: LM Studio (OpenAI-compatible) client
  - `rag_service.py`: Orchestration (process docs → add to vectors → query)
- Models in `app/models`: chat and documents
- Config in `app/core/config.py` (via pydantic-settings)

## Requirements
- Python 3.11+
- LM Studio running locally with a chat model (e.g., Llama 3.1 8B Instruct) on port 1234
- CPU/GPU as needed for SentenceTransformers (default: all-MiniLM-L6-v2)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Configuration
Environment variables are loaded from `.env` (see defaults in `app/core/config.py`).

Common settings:
- APP host/port: `host`, `port`
- LM Studio: `lm_studio_url` (default: http://localhost:1234), `lm_studio_model` (default: llama-3.1-8b-instruct)
- Vector DB: `chroma_db_path` (default: ./storage/vector_db), `embedding_model` (default: all-MiniLM-L6-v2)
- File storage: `upload_dir` (default: ./storage/documents), `max_file_size` (MB), `allowed_extensions` (pdf, docx, txt, md)
- Chunking: `chunk_size` (default: 800), `chunk_overlap` (default: 200)

Example `.env`:
```
HOST=0.0.0.0
PORT=8000
DEBUG=true
LM_STUDIO_URL=http://localhost:1234
LM_STUDIO_MODEL=llama-3.1-8b-instruct
CHROMA_DB_PATH=./storage/vector_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
UPLOAD_DIR=./storage/documents
MAX_FILE_SIZE=50
ALLOWED_EXTENSIONS=["pdf","docx","txt","md"]
CHUNK_SIZE=800
CHUNK_OVERLAP=200
```

## Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open docs: http://localhost:8000/docs

## API Overview
- Health
  - GET `/health/` — overall status
  - GET `/health/lm-studio` — LM Studio status
  - GET `/health/vector-db` — ChromaDB status
- Documents
  - POST `/documents/upload` — upload a file (multipart/form-data)
  - GET `/documents/` — list documents (optional `collection` filter)
  - GET `/documents/{document_id}` — get document info
  - DELETE `/documents/{document_id}` — delete document and its file
  - GET `/documents/collections/` — list collections
  - POST `/documents/collections/{collection_name}` — create collection
  - DELETE `/documents/collections/{collection_name}` — delete collection
- Chat
  - POST `/chat/query` — RAG answer (non-streaming)
  - POST `/chat/stream` — RAG answer (streaming SSE)
  - GET `/chat/collections/{collection_name}/preview` — preview top-k chunks (semantic)

## Usage Examples
Upload (curl):
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/Resume.pdf;type=application/pdf" \
  -F "collection=default" \
  -F 'metadata={"source":"user","tags":["cv"]}'
```

Query (curl):
```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What projects are mentioned?", "collection":"default", "max_sources":5, "temperature":0.1}'
```

Streaming (curl):
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize the key experience.", "collection":"default"}'
```

## Chunking and Embeddings
- PDFs are processed page-by-page; each chunk includes `filename` and `page` in metadata.
- Text is cleaned (control chars removed, whitespace normalized) before chunking.
- Chunking parameters come from settings: `chunk_size` and `chunk_overlap`.
- Embeddings are computed locally using SentenceTransformers and stored in ChromaDB with each document chunk. Queries compute embeddings locally as well.

## Data Storage
- Uploaded files are stored on disk under `./storage/documents` as `{uuid}_{original_filename}`.
- Document info and chunks are kept in-memory (temporary). On restart, you will need to re-upload or reindex to repopulate the index.
- Vector data persists in `./storage/vector_db` (ChromaDB PersistentClient).

## Troubleshooting
- 400 Bad Request on upload:
  - Ensure multipart/form-data with field name `file`.
  - Allowed extensions: pdf, docx, txt, md.
  - File size must be <= `max_file_size` (MB).
  - `metadata` must be a valid JSON string or be left empty in Swagger UI.
- Filename shows as `unknown` in results:
  - Chunks indexed prior to the recent update may lack `filename` in metadata. Recreate the collection or re-upload documents to index with proper metadata.
- LM Studio connection errors:
  - Verify LM Studio is running at `lm_studio_url` and the model is loaded.
- PDF contains strange characters:
  - Some PDFs don’t extract cleanly; basic cleaning is applied. For scanned PDFs, consider adding OCR in the future.

## Security & CORS
- CORS allows all origins for development. In production, restrict `allow_origins` to trusted domains.
- No authentication is implemented by default.

## Roadmap / Next Steps
- Persistent storage for document metadata and chunks (e.g., SQLite/Postgres)
- OCR for scanned PDFs
- Reindexing/migration utilities for existing files
- Extended metadata (page references for non-PDF, source deduplication)
- Tests (end-to-end: upload → index → query → sources)

## License
This project is provided as-is for demonstration and internal use.
