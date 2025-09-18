import chromadb
from chromadb.config import Settings as ChromaSettings
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging

from app.core.config import settings
from app.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        logger.info(f"VectorService initialized with model: {settings.embedding_model}")

    async def create_collection(self, name: str) -> bool:
        """Create a new document collection"""
        try:
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created collection: {name}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection {name} already exists")
                return True
            logger.error(f"Error creating collection {name}: {e}")
            return False

    async def add_document_chunks(
            self,
            collection_name: str,
            chunks: List[Dict[str, Any]]
    ) -> bool:
        """Add document chunks to a collection"""
        try:
            collection = self.client.get_collection(collection_name)

            # Prepare data
            ids = []
            documents = []
            metadatas = []

            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                documents.append(chunk["content"])
                metadatas.append({
                    "document_id": chunk["document_id"],
                    "chunk_index": chunk["chunk_index"],
                    "filename": chunk.get("filename", "unknown"),
                    "page": chunk.get("page", "unknown"),
                    **chunk.get("metadata", {})
                })

            # Oblicz embeddingi dokumentów lokalnie i dodaj do kolekcji
            try:
                embeddings = self.embedding_model.encode(documents)
                embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return False

            # Dodaj do ChromaDB z embeddingami
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

            logger.info(f"Added {len(chunks)} chunks to collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding chunks to collection {collection_name}: {e}")
            return False

    async def similarity_search(
            self,
            query: str,
            collection_name: str,
            k: int = 5
    ) -> List[DocumentChunk]:
        """Search similar document chunks"""
        try:
            collection = self.client.get_collection(collection_name)

            # Compute query embedding locally to avoid relying on the collection's embedding_function
            try:
                query_emb = self.embedding_model.encode([query])
                if hasattr(query_emb, 'tolist'):
                    query_emb = query_emb.tolist()
            except Exception as e:
                logger.error(f"Query embedding generation failed: {e}")
                return []

            results = collection.query(
                query_embeddings=query_emb,
                n_results=k
            )

            chunks = []
            if results.get('documents') and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                )):
                    chunk = DocumentChunk(
                        id=results['ids'][0][i],
                        document_id=metadata.get('document_id', ''),
                        chunk_index=metadata.get('chunk_index', 0),
                        content=doc,
                        metadata=metadata,
                        similarity_score=1.0 - distance  # Convert distance to similarity
                    )
                    chunks.append(chunk)

            logger.info(f"Found {len(chunks)} similar chunks for query in {collection_name}")
            return chunks

        except Exception as e:
            logger.error(f"Error searching in collection {collection_name}: {e}")
            return []

    async def get_collections(self) -> List[Dict[str, Any]]:
        """Get all collections"""
        try:
            collections = self.client.list_collections()
            result = []

            for collection in collections:
                col_info = self.client.get_collection(collection.name)
                count = col_info.count()

                result.append({
                    "name": collection.name,
                    "chunks_count": count,
                    "metadata": collection.metadata or {}
                })

            return result

        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {name}: {e}")
            return False

    async def health_check(self) -> bool:
        """Check vector database status"""
        try:
            collections = self.client.list_collections()
            return True
        except Exception as e:
            logger.error(f"Vector DB health check failed: {e}")
            return False