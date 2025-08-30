from sentence_transformers import SentenceTransformer
from settings import settings
import numpy as np
import faiss
import json
import os
from typing import List, Dict, Tuple

class EmbeddingModel:
    # Enable embedding model to be switched out, set model_name in settings.py
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.model_name
        self.model = SentenceTransformer(self.model_name)
        # print(f"Initialized embedding model: {self.model_name}")
    
    # Encode text to embeddings
    def encode_text(self, texts: List[str], convert_to_numpy=True) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=convert_to_numpy)
        # print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    # Encode query for retrieval
    def encode_query(self, query: str) -> np.ndarray:
        query_embeddings = self.model.encode([query], convert_to_numpy=True)
        # print(f"Created query embedding with shape: {query_embeddings.shape}")
        return query_embeddings

class RAGPipeline:    
    def __init__(self, model_name: str = None):
        self.embedding_model = EmbeddingModel(model_name)
        self.index = None # FAISS index
        self.documents = [] # list of documents
        self.document_embeddings = None # embeddings of documents
        
        self.top_k = settings.top_k
        self.rebuild_index = settings.rebuild_index
        
        # Convert relative paths to absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
        project_root = os.path.dirname(current_dir)  # project root
        
        # Handle index_dir
        if not os.path.isabs(settings.index_dir):
            self.index_dir = os.path.join(project_root, settings.index_dir)
        else:
            self.index_dir = settings.index_dir
        
        # Handle index_path
        if not os.path.isabs(settings.index_path):
            self.index_path = os.path.join(project_root, settings.index_path)
        else:
            self.index_path = settings.index_path
    
    # ===== Document Loading and Preparation =====
    def load_documents(self, docs_path: str = None) -> List[Dict]:
        docs_path = docs_path or settings.docs_path
        
        # If it's a relative path, make it absolute relative to project root
        if not os.path.isabs(docs_path):
            # Get the project root directory (two levels up from src/)
            current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
            project_root = os.path.dirname(current_dir)  # project root
            docs_path = os.path.join(project_root, docs_path)
        
        try:
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"Loaded {len(self.documents)} documents from {docs_path}")
            return self.documents
        except FileNotFoundError:
            print(f"Error: Documents file not found at {docs_path}")
            return []
    
    def prepare_documents(self) -> List[str]:
        prepared_texts = []
        for doc in self.documents:
            # Combine title, section, and content for better semantic search
            text = f"{doc['title']}: {doc['section']}. {doc['content']}"
            prepared_texts.append(text)
        return prepared_texts
    
    # ===== Embedding and Index Management =====
    # Create embeddings from text
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode_text(texts)
        embeddings = embeddings.astype("float32", copy=False) # FAISS needs float32
        faiss.normalize_L2(embeddings) # Normalize embedding for cosine similarity
        # print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings

    # Build FAISS index from embeddings
    def build_index(self, embeddings: np.ndarray) -> None:
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # Inner product for cosine similarity
        self.index.add(embeddings)
        # print(f"FAISS index built with {self.index.ntotal} vectors")
    
    # Save FAISS index to file
    def save_index(self, index_path: str = None) -> bool:
        index_path = index_path or self.index_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        try:
            faiss.write_index(self.index, index_path)
            print(f"Index saved to {index_path}")
            return True
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    # Load FAISS index from file
    def load_index(self, index_path: str = None) -> bool:
        index_path = index_path or self.index_path
        
        try:
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                # print(f"Index loaded from {index_path}")
                return True
            else:
                # print(f"Index file not found at {index_path}")
                return False
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    # ===== Pipeline Setup =====
    def setup_pipeline(self, docs_path: str = None) -> bool:
        # Load documents
        if not self.load_documents(docs_path):
            return False
        
        texts = self.prepare_documents()
        if not texts:
            return False
        
        # Check if we should rebuild index
        if self.rebuild_index or not os.path.exists(self.index_path):
            print("Building new index...")
            self.document_embeddings = self.create_embeddings(texts)
            self.build_index(self.document_embeddings)
            self.save_index()
        else:
            print("Loading existing index...")
            if not self.load_index():
                print("Failed to load index, building new one...")
                self.document_embeddings = self.create_embeddings(texts)
                self.build_index(self.document_embeddings)
                self.save_index()
        
        print("RAG pipeline setup completed")
        return True
    
    # ===== Search and Retrieval =====
    def search(self, query: str, k: int = None) -> List[Tuple[int, float, Dict]]:
        if self.index is None:
            print("Error: Index not built. Run setup_pipeline() first.")
            return []
        
        k = k or self.top_k
        
        # Create query embedding
        query_embedding = self.embedding_model.encode_query(query)
        query_embedding = query_embedding.astype("float32", copy=False)
        faiss.normalize_L2(query_embedding) # Normalize query embedding for cosine similarity
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
            if doc_idx < len(self.documents):
                results.append((int(doc_idx), float(score), self.documents[doc_idx]))
        
        return results
    
    def get_relevant_context(self, query: str, k: int = None) -> str:
        k = k or self.top_k

        results = self.search(query, k)
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, (doc_idx, score, doc) in enumerate(results, 1):
            context_parts.append(
                f"Document {i} (Relevance: {score:.3f}):\n"
                f"Title: {doc['title']}\n"
                f"Section: {doc['section']}\n"
                f"Content: {doc['content']}\n"
                f"Reference: {doc['ref']}\n"
            )
        
        return "\n".join(context_parts)