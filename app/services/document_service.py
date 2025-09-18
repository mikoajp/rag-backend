import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

# Document processors
import PyPDF2
from docx import Document

from app.core.config import settings
from app.models.document import DocumentInfo, DocumentStatus
from app.utils.file_utils import save_uploaded_file, get_file_info

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage for document info (use a database in production)
        self.documents: Dict[str, DocumentInfo] = {}
        # In-memory storage for processed chunks (temporary; persist in DB in production)
        self.document_chunks: Dict[str, List[Dict[str, Any]]] = {}

    async def upload_document(
            self,
            file_content: bytes,
            filename: str,
            collection: str = "default",
            metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentInfo:
        """Upload and basic document processing"""

        document_id = str(uuid.uuid4())
        # Sanitize filename (basename only)
        safe_filename = Path(filename).name
        file_path = self.upload_dir / f"{document_id}_{safe_filename}"

        try:
            # Save file
            await save_uploaded_file(file_content, file_path)
            file_info = get_file_info(file_path)

            # Create document info
            doc_info = DocumentInfo(
                id=document_id,
                filename=safe_filename,
                collection=collection,
                status=DocumentStatus.UPLOADING,
                file_size=file_info["size"],
                content_type=file_info["mime_type"],
                created_at=datetime.now(),
                metadata=metadata or {}
            )

            # Store in memory (persist to database in production)
            self.documents[document_id] = doc_info

            logger.info(f"Document uploaded: {filename} (ID: {document_id})")

            # Start async processing
            asyncio.create_task(self._process_document(document_id, file_path))

            return doc_info

        except Exception as e:
            logger.error(f"Error uploading document {filename}: {e}")
            raise e

    async def _process_document(self, document_id: str, file_path: Path):
        """Process document in the background"""
        try:
            doc_info = self.documents[document_id]
            doc_info.status = DocumentStatus.PROCESSING

            # Extract text based on file type + chunk with page metadata for PDFs
            chunks: List[Dict[str, Any]] = []
            file_extension = file_path.suffix.lower()

            if file_extension == '.pdf':
                pages = self._extract_pdf_pages(file_path)
                for page_num, page_text in pages:
                    clean = self._clean_text(page_text)
                    page_chunks = await self._split_into_chunks(
                        clean, document_id, filename=doc_info.filename, page=page_num,
                        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
                    )
                    chunks.extend(page_chunks)
            else:
                text_content = await self._extract_text(file_path)
                clean = self._clean_text(text_content)
                chunks = await self._split_into_chunks(
                    clean, document_id, filename=doc_info.filename,
                    chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
                )

            # Save chunks in memory for retrieval by RAGService
            self.document_chunks[document_id] = chunks

            # Update document info
            doc_info.chunks_count = len(chunks)
            doc_info.status = DocumentStatus.COMPLETED
            doc_info.processed_at = datetime.now()

            logger.info(f"Document processed: {doc_info.filename} ({len(chunks)} chunks)")

            return chunks

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            if document_id in self.documents:
                self.documents[document_id].status = DocumentStatus.FAILED
                # Clear potentially partially saved chunks
                self.document_chunks.pop(document_id, None)

    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from different file formats (PDF is handled specially elsewhere)"""

        file_extension = file_path.suffix.lower()

        try:
            if file_extension == '.docx':
                return self._extract_docx_text(file_path)
            elif file_extension in ['.txt', '.md']:
                return self._extract_plain_text(file_path)
            elif file_extension == '.pdf':
                # Dla PDF używaj _extract_pdf_pages w _process_document, ta ścieżka nie jest wykorzystywana
                return self._extract_pdf_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise e

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF (kept for compatibility; prefer _extract_pdf_pages)"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ""
                text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
        return text

    def _extract_pdf_pages(self, file_path: Path) -> List[Tuple[int, str]]:
        """Extract PDF text page-by-page"""
        pages: List[Tuple[int, str]] = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ""
                pages.append((page_num, page_text))
        return pages

    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        doc = Document(file_path)
        paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        return "\n\n".join(paragraphs)

    def _extract_plain_text(self, file_path: Path) -> str:
        """Extract text from a text file (with encoding fallback)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as file:
                    return file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
                    return file.read()

    async def _split_into_chunks(
            self,
            text: str,
            document_id: str,
            filename: Optional[str] = None,
            page: Optional[int] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Split text into chunks"""

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + chunk_size

            # If not at the end of text, try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                last_exclamation = text.rfind('!', end - 100, end)
                last_question = text.rfind('?', end - 100, end)

                sentence_end = max(last_period, last_exclamation, last_question)
                if sentence_end > start:
                    end = sentence_end + 1

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "filename": filename or "unknown",
                    "page": page if page is not None else "unknown",
                    "metadata": {
                        "start_pos": start,
                        "end_pos": end,
                        "length": len(chunk_text)
                    }
                })
                chunk_index += 1

            # Move start position (with overlap)
            start = end - chunk_overlap if end < len(text) else end

        return chunks

    def _clean_text(self, text: str) -> str:
        """Czyści tekst: usuwa znaki kontrolne (poza \n i \t), normalizuje białe znaki."""
        if not text:
            return ""
        # Usuń null bytes i podobne
        cleaned = text.replace('\x00', ' ')
        # Usuń inne kontrolne poza nową linią i tabem
        cleaned = ''.join(ch if ch in ['\n', '\t'] or ord(ch) >= 32 else ' ' for ch in cleaned)
        # Normalizuj wielokrotne spacje i puste linie
        lines = [" ".join(line.split()) for line in cleaned.splitlines()]
        return "\n".join(line for line in lines if line)

    async def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """Pobiera informacje o dokumencie"""
        return self.documents.get(document_id)

    async def list_documents(self, collection: Optional[str] = None) -> List[DocumentInfo]:
        """Lista wszystkich dokumentów"""
        if collection:
            return [doc for doc in self.documents.values() if doc.collection == collection]
        return list(self.documents.values())

    async def delete_document(self, document_id: str) -> bool:
        """Usuwa dokument"""
        try:
            if document_id in self.documents:
                doc_info = self.documents[document_id]

                # Delete file
                file_path = self.upload_dir / f"{document_id}_{doc_info.filename}"
                if file_path.exists():
                    file_path.unlink()

                # Remove from memory
                del self.documents[document_id]
                self.document_chunks.pop(document_id, None)

                logger.info(f"Document deleted: {document_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False