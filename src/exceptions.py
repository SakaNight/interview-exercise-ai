class RAGError(Exception):
    # Base class for all RAG-related errors
    def __init__(self, message: str, code: str = "RAG_ERROR"):
        super().__init__(message)
        self.code = code

# ===== Document-related errors ===== 
class DocumentNotFoundError(RAGError):
    # Raised when the documents file (docs.json) is missing or invalid
    def __init__(self, path: str):
        super().__init__(f"Documents file not found: {path}", "DOCUMENT_NOT_FOUND")
        self.path = path

class DocumentFormatError(RAGError):
    # Raised when a document JSON has missing required fields
    def __init__(self, where: str):
        super().__init__(f"Invalid document format: {where}", "DOCUMENT_FORMAT_ERROR")
        self.where = where

#  ==== Query-related errors =====
class QueryFormatError(RAGError):
    # Raised when a query is empty
    def __init__(self, reason: str = "empty query"):
        super().__init__(f"Invalid query format: {reason}", "QUERY_FORMAT_ERROR")
        self.reason = reason

# ===== Index-related errors ===== 
class IndexNotReadyError(RAGError):
    # Raised when FAISS index is not built or loaded yet
    def __init__(self):
        super().__init__("FAISS index not built. Run setup_pipeline() first.", "INDEX_NOT_READY")

class IndexLoadError(RAGError):
    # Raised when FAISS index file cannot be loaded
    def __init__(self, path: str, reason: str = None):
        msg = f"Failed to load FAISS index from {path}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, "INDEX_LOAD_ERROR")
        self.path = path
        self.reason = reason
        
# ===== Embedding/Model errors ===== 
class EmbeddingModelError(RAGError):
    # Raised when the embedding model fails to encode text
    def __init__(self, message: str):
        super().__init__(message, "EMBEDDING_MODEL_ERROR")

# ===== LLM errors ===== 
class LLMProviderError(RAGError):
    # Raised when the LLM provider returns an error
    def __init__(self, message: str):
        super().__init__(message, "LLM_PROVIDER_ERROR")

class MCPOutputError(RAGError):
    # Raised when the LLM output is not valid MCP JSON
    def __init__(self, message: str):
        super().__init__(message, "MCP_OUTPUT_ERROR")