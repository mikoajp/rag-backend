import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging

from .llm_service import LMStudioService
from .vector_service import VectorService
from .document_service import DocumentService
from ..models.document import DocumentChunk
from ..models.chat import ChatResponse, ChatSource

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.llm_service = LMStudioService()
        self.vector_service = VectorService()
        self.document_service = DocumentService()
        logger.info("RAG Service initialized")

    async def process_document_for_rag(
            self,
            document_id: str,
            collection: str = "default"
    ) -> bool:
        """Przetwarza dokument i dodaje do vector database"""
        try:
            # Get document info
            doc_info = await self.document_service.get_document(document_id)
            if not doc_info:
                logger.error(f"Document {document_id} not found")
                return False

            # Wait for document processing to complete
            max_retries = 30  # 30 seconds timeout
            retry_count = 0

            while doc_info.status.value in ["uploading", "processing"] and retry_count < max_retries:
                await asyncio.sleep(1)
                doc_info = await self.document_service.get_document(document_id)
                retry_count += 1

            if doc_info.status.value != "completed":
                logger.error(f"Document {document_id} processing failed or timed out")
                return False

            # Get processed chunks from document service
            chunks = await self._get_document_chunks(document_id)
            if not chunks:
                logger.error(f"No chunks found for document {document_id}")
                return False

            # Create collection if doesn't exist
            await self.vector_service.create_collection(collection)

            # Add chunks to vector database
            success = await self.vector_service.add_document_chunks(collection, chunks)

            if success:
                logger.info(f"Document {document_id} successfully added to RAG system")

            return success

        except Exception as e:
            logger.error(f"Error processing document {document_id} for RAG: {e}")
            return False

    async def query(
            self,
            query: str,
            collection: str = "default",
            max_sources: int = 5,
            temperature: float = 0.1
    ) -> ChatResponse:
        """Main RAG function - retrieval + generation"""
        start_time = time.time()

        try:
            # 1. Retrieve relevant documents
            relevant_chunks = await self.vector_service.similarity_search(
                query=query,
                collection_name=collection,
                k=max_sources
            )

            if not relevant_chunks:
                return ChatResponse(
                    answer="I couldn't find any relevant documents to answer your question. Try uploading relevant documents to the collection or ask a different question.",
                    sources=[],
                    model_info={
                        "model": "llama-3.1-8b-instruct",
                        "local": True,
                        "chunks_used": 0
                    },
                    processing_time=time.time() - start_time
                )

            # 2. Prepare context for LLM
            context = self._prepare_context(relevant_chunks)

            # 3. Create optimized prompt
            messages = self._create_rag_prompt(query, context)

            # 4. Generate answer with LM Studio
            llm_response = await self.llm_service.generate_response(
                messages=messages,
                temperature=temperature,
                max_tokens=300
            )

            # 5. Prepare sources
            sources = self._prepare_sources(relevant_chunks)

            # 6. Create response
            response = ChatResponse(
                answer=llm_response["content"],
                sources=sources,
                model_info={
                    "model": llm_response.get("model", "llama-3.1-8b-instruct"),
                    "local": True,
                    "chunks_used": len(relevant_chunks),
                    "temperature": temperature
                },
                processing_time=time.time() - start_time,
                tokens_used=llm_response.get("usage", {}).get("total_tokens")
            )

            logger.info(f"RAG query processed in {response.processing_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return ChatResponse(
                answer=f"Przepraszam, wystąpił błąd podczas przetwarzania pytania: {str(e)}",
                sources=[],
                model_info={"error": str(e)},
                processing_time=time.time() - start_time
            )

    async def stream_query(
            self,
            query: str,
            collection: str = "default",
            max_sources: int = 5,
            temperature: float = 0.1
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming version of RAG query"""

        try:
            # First yield sources info
            yield {"type": "sources_search", "message": "Searching for relevant documents..."}

            # Retrieve relevant documents
            relevant_chunks = await self.vector_service.similarity_search(
                query=query,
                collection_name=collection,
                k=max_sources
            )

            if not relevant_chunks:
                yield {
                    "type": "answer",
                    "content": "No relevant documents were found to answer this question.",
                    "done": True
                }
                return

            # Send sources info
            sources = self._prepare_sources(relevant_chunks)
            yield {
                "type": "sources",
                "sources": [source.dict() for source in sources]
            }

            # Prepare context and prompt
            context = self._prepare_context(relevant_chunks)
            messages = self._create_rag_prompt(query, context)

            yield {"type": "generation_start", "message": "Generating answer..."}

            # Stream response from LLM
            async for token in self.llm_service.generate_streaming_response(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=300
            ):
                if token.startswith("ERROR:"):
                    yield {"type": "error", "error": token}
                    break
                else:
                    yield {"type": "token", "content": token}

            yield {"type": "done", "done": True}

        except Exception as e:
            logger.error(f"Error in streaming RAG query: {e}")
            yield {"type": "error", "error": str(e)}

    async def _get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Fetch chunks from the processed document"""
        # W tej implementacji chunks są tworzone w document_service
        # Tu musimy je pobrać - w prawdziwej aplikacji byłyby w bazie danych

        # Tymczasowe rozwiązanie - chunks są tworzone w _process_document
        # W production zastąp to prawdziwym storage
        doc_info = await self.document_service.get_document(document_id)
        if not doc_info:
            return []

        # Symulujemy chunks - w production byłyby zapisane w DB
        # Tu zakładamy że document_service ma sposób na zwrócenie chunks
        return []  # TODO: Implement proper chunk retrieval

    def _prepare_context(self, chunks: List[DocumentChunk]) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            filename = chunk.metadata.get("filename", "Nieznany dokument")
            page = chunk.metadata.get("page", "nieznana")

            context_parts.append(f"""
[Source {i}]
Document: {filename}
Page: {page}
Content: {chunk.content}
            """.strip())

        return "\n\n".join(context_parts)

    def _create_rag_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """Tworzy zoptymalizowany prompt dla local LLM"""

        system_prompt = """You are a helpful AI assistant for document analysis. 

RULES:
- Answer ONLY based on the provided sources.
- If the information is not in the sources, write exactly: "Information not found in documents."
- Do not repeat the user's question or include prefixes like "User:".
- Avoid repetition and verbosity; be concise and clear.

OUTPUT FORMAT (Markdown):
Answer: <1–2 sentences>
Rationale:
- <up to 2–4 brief bullet points>
"""

        user_prompt = f"""
Answer the question based on the following document fragments.

AVAILABLE SOURCES:
{context}

QUESTION: {query}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _prepare_sources(self, chunks: List[DocumentChunk]) -> List[ChatSource]:
        """Prepare source info for the response"""
        sources = []

        for chunk in chunks:
            source = ChatSource(
                document_id=chunk.document_id,
                filename=chunk.metadata.get("filename", "Unknown document"),
                page=str(chunk.metadata.get("page", "unknown")),
                content_preview=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                similarity_score=round(chunk.similarity_score or 0.0, 3)
            )
            sources.append(source)

        return sources

    async def get_collections(self) -> List[Dict[str, Any]]:
        """Get information about all collections"""
        try:
            collections = await self.vector_service.get_collections()

            # Dodaj informacje o dokumentach dla każdej kolekcji
            for collection in collections:
                documents = await self.document_service.list_documents(collection["name"])
                collection["documents_count"] = len([d for d in documents if d.status.value == "completed"])
                collection["total_size_mb"] = sum(d.file_size for d in documents) / (1024 * 1024)

            return collections

        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Sprawdza status całego systemu RAG"""
        try:
            # Check LM Studio
            llm_status = await self.llm_service.check_server_status()

            # Check Vector DB
            vector_status = await self.vector_service.health_check()

            # System status
            overall_status = "healthy" if all([llm_status, vector_status]) else "unhealthy"

            return {
                "status": overall_status,
                "components": {
                    "llm_service": "up" if llm_status else "down",
                    "vector_database": "up" if vector_status else "down",
                    "document_service": "up"  # Always up for now
                },
                "model_info": {
                    "model": "llama-3.1-8b-instruct",
                    "endpoint": self.llm_service.base_url,
                    "local": True
                }
            }

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self):
        """Cleanup resources"""
        await self.llm_service.close()