"""
RAG (Retrieval-Augmented Generation) pipeline implementation.

This module provides a complete RAG pipeline for document retrieval and
semantic search using FAISS vector index and sentence transformers.
It handles document loading, embedding generation, index building,
and similarity search for knowledge base queries.
"""

import json
import os

import faiss
import numpy as np

from exceptions import (
    DocumentFormatError,
    DocumentNotFoundError,
    IndexNotReadyError,
    QueryFormatError,
)
from logging_config import get_logger
from models import EmbeddingModel
from settings import settings

logger = get_logger(__name__)


class RAGPipeline:
    """RAG pipeline for document retrieval and semantic search.

    Provides document loading, embedding generation, FAISS indexing,
    and semantic similarity search over a knowledge base.
    """

    def __init__(self, model_name: str = None):
        """Initialize RAG pipeline with embedding model.

        Args:
            model_name: Name of the embedding model. Defaults to settings.model_name.
        """
        self.embedding_model = EmbeddingModel(model_name)
        self.index = None  # FAISS index
        self.documents = []  # list of documents
        self.document_embeddings = None  # embeddings of documents

        self.top_k = settings.top_k
        self.rebuild_index = settings.rebuild_index

        # Convert relative paths to absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        # Handle index_path
        if not os.path.isabs(settings.index_path):
            self.index_path = os.path.join(project_root, settings.index_path)
        else:
            self.index_path = settings.index_path

    def load_documents(self, docs_path: str = None) -> list[dict]:
        """Load documents from JSON file.

        Args:
            docs_path: Path to documents JSON file. Defaults to settings.docs_path.

        Returns:
            list[dict]: List of loaded documents.

        Raises:
            DocumentNotFoundError: If the documents file is not found.
            DocumentFormatError: If the JSON format is invalid.
        """
        docs_path = docs_path or settings.docs_path

        # If it's a relative path, make it absolute relative to project root
        if not os.path.isabs(docs_path):
            # Get the project root directory (two levels up from src/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            docs_path = os.path.join(project_root, docs_path)

        try:
            with open(docs_path, encoding="utf-8") as f:
                self.documents = json.load(f)
            for i, doc in enumerate(self.documents):
                required = ("id", "title", "section", "content")
                missing = [
                    k for k in required if k not in doc or not str(doc[k]).strip()
                ]
                if missing:
                    raise DocumentFormatError(f"doc index {i} missing {missing}")
                doc.setdefault(
                    "ref", f"{doc['title']} - {doc['section']} (#{doc['id']})"
                )

            logger.info("Loaded %d documents from %s", len(self.documents), docs_path)
            return self.documents
        except FileNotFoundError:
            raise DocumentNotFoundError(docs_path)
        except json.JSONDecodeError:
            raise DocumentFormatError(docs_path)

    def prepare_documents(self) -> list[str]:
        """Prepare documents for embedding by extracting text content.

        Returns:
            list[str]: List of prepared text strings from documents.
        """
        prepared_texts = []
        for doc in self.documents:
            # Combine title, section, and content for better semantic search
            text = f"{doc['title']}: {doc['section']}. {doc['content']}"
            prepared_texts.append(text)
        return prepared_texts

    def create_embeddings(self, texts: list[str]) -> np.ndarray:
        """Create embeddings from text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            np.ndarray: Normalized embeddings array for FAISS indexing.
        """
        embeddings = self.embedding_model.encode(texts)
        embeddings = embeddings.astype("float32", copy=False)  # FAISS needs float32
        faiss.normalize_L2(embeddings)  # Normalize embedding for cosine similarity
        logger.debug(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings

    def build_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index from embeddings.

        Args:
            embeddings: Normalized embeddings array to index.
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

    def save_index(self, index_path: str = None) -> bool:
        """Save FAISS index to file.

        Args:
            index_path: Path to save the index. Defaults to self.index_path.

        Returns:
            bool: True if index was saved successfully.
        """
        index_path = index_path or self.index_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        try:
            faiss.write_index(self.index, index_path)
            logger.info(f"Index saved to {index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False

    def load_index(self, index_path: str = None) -> bool:
        """Load FAISS index from file.

        Args:
            index_path: Path to load the index from. Defaults to self.index_path.

        Returns:
            bool: True if index was loaded successfully.
        """
        index_path = index_path or self.index_path

        try:
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info(f"Index loaded from {index_path}")
                return True
            else:
                logger.info(f"Index file not found at {index_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def setup_pipeline(self, docs_path: str = None) -> bool:
        """Setup the complete RAG pipeline.

        Args:
            docs_path: Path to documents file. Defaults to settings.docs_path.

        Returns:
            bool: True if pipeline setup was successful.
        """
        try:
            self.load_documents(docs_path)
        except Exception:
            logger.exception("Error loading documents")
            raise
        texts = self.prepare_documents()
        if not texts:
            raise DocumentFormatError(docs_path)

        if self.rebuild_index or not os.path.exists(self.index_path):
            logger.info("Building new index...")
            self.document_embeddings = self.create_embeddings(texts)
            self.build_index(self.document_embeddings)
            self.save_index()
        else:
            logger.info("Loading existing index...")
            if not self.load_index():
                logger.info("Failed to load index, building new one...")
                self.document_embeddings = self.create_embeddings(texts)
                self.build_index(self.document_embeddings)
                self.save_index()

        logger.info("RAG pipeline setup completed")
        return True

    def search(self, query: str, k: int = None) -> list[tuple[int, float, dict]]:
        """Search for similar documents using semantic similarity.

        Args:
            query: Search query string.
            k: Number of top results to return. Defaults to self.top_k.

        Returns:
            list[tuple[int, float, dict]]: List of (doc_index, score, document) tuples.

        Raises:
            IndexNotReadyError: If the index is not built.
            QueryFormatError: If the query is invalid.
        """
        if self.index is None:
            logger.error("Error: Index not built. Run setup_pipeline() first.")
            raise IndexNotReadyError()

        if not query or not query.strip():
            logger.warning("Empty query provided to search()")
            raise QueryFormatError()

        k = k or self.top_k

        query_embedding = self.embedding_model.encode_query(query)
        query_embedding = query_embedding.astype("float32", copy=False)
        faiss.normalize_L2(
            query_embedding
        )  # Normalize query embedding for cosine similarity

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for _i, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
            if 0 <= doc_idx < len(self.documents):
                results.append((int(doc_idx), float(score), self.documents[doc_idx]))

        return results

    def get_relevant_context(self, query: str, k: int = None) -> str:
        """Get formatted context string from relevant documents.

        Args:
            query: Search query string.
            k: Number of top results to return. Defaults to self.top_k.

        Returns:
            str: Formatted context string with relevant documents.
        """
        k = k or self.top_k

        results = self.search(query, k)
        if not results:
            return "No relevant documents found"

        context_parts = []
        for i, (_doc_idx, score, doc) in enumerate(results, 1):
            context_parts.append(
                f"Document {i} (Relevance: {score:.3f}):\n"
                f"Title: {doc['title']}\n"
                f"Section: {doc['section']}\n"
                f"Content: {doc['content']}\n"
                f"Reference: {doc['ref']}\n"
            )

        return "\n".join(context_parts)
