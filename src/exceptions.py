"""
Custom exception classes for RAG pipeline and LLM service.

This module defines a hierarchy of custom exceptions for handling various
error conditions in the RAG pipeline, LLM interactions, and data validation.
"""


class RAGError(Exception):
    """Base class for all RAG-related errors."""

    def __init__(self, message: str, code: str = "RAG_ERROR"):
        super().__init__(message)
        self.code = code


class DocumentNotFoundError(RAGError):
    """Raised when the documents file (docs.json) is missing or invalid."""

    def __init__(self, path: str):
        super().__init__(f"Documents file not found: {path}", "DOCUMENT_NOT_FOUND")
        self.path = path


class DocumentFormatError(RAGError):
    """Raised when a document JSON has missing required fields."""

    def __init__(self, where: str):
        super().__init__(f"Invalid document format: {where}", "DOCUMENT_FORMAT_ERROR")
        self.where = where


class QueryFormatError(RAGError):
    """Raised when a query is empty or invalid."""

    def __init__(self, reason: str = "empty query"):
        super().__init__(f"Invalid query format: {reason}", "QUERY_FORMAT_ERROR")
        self.reason = reason


class IndexNotReadyError(RAGError):
    """Raised when FAISS index is not built or loaded yet."""

    def __init__(self):
        super().__init__(
            "FAISS index not built. Run setup_pipeline() first.", "INDEX_NOT_READY"
        )


class IndexLoadError(RAGError):
    """Raised when FAISS index file cannot be loaded."""

    def __init__(self, path: str, reason: str = None):
        msg = f"Failed to load FAISS index from {path}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, "INDEX_LOAD_ERROR")
        self.path = path
        self.reason = reason


class EmbeddingModelError(RAGError):
    """Raised when the embedding model fails to encode text."""

    def __init__(self, message: str):
        super().__init__(message, "EMBEDDING_MODEL_ERROR")


class LLMProviderError(RAGError):
    """Raised when the LLM provider returns an error."""

    def __init__(self, message: str):
        super().__init__(message, "LLM_PROVIDER_ERROR")


class MCPOutputError(RAGError):
    """Raised when the LLM output is not valid MCP JSON."""

    def __init__(self, message: str):
        super().__init__(message, "MCP_OUTPUT_ERROR")


class LLMConfigError(RAGError):
    """Raised when LLM configuration is invalid."""

    def __init__(self, message: str, config_key: str = None):
        msg = f"LLM configuration error: {message}"
        if config_key:
            msg += f" (key: {config_key})"
        super().__init__(msg, "LLM_CONFIG_ERROR")
        self.config_key = config_key


class LLMRetryError(RAGError):
    """Raised when LLM retry mechanism fails."""

    def __init__(self, message: str, attempts: int = 0, last_error: str = None):
        msg = f"LLM retry failed after {attempts} attempts: {message}"
        if last_error:
            msg += f" (last error: {last_error})"
        super().__init__(msg, "LLM_RETRY_ERROR")
        self.attempts = attempts
        self.last_error = last_error


class LLMTimeoutError(RAGError):
    """Raised when LLM request times out."""

    def __init__(self, message: str, timeout_seconds: float = 0):
        msg = f"LLM request timed out: {message}"
        if timeout_seconds > 0:
            msg += f" (timeout: {timeout_seconds}s)"
        super().__init__(msg, "LLM_TIMEOUT_ERROR")
        self.timeout_seconds = timeout_seconds


class LLMRateLimitError(RAGError):
    """Raised when LLM API rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int = None):
        msg = f"LLM rate limit exceeded: {message}"
        if retry_after:
            msg += f" (retry after: {retry_after}s)"
        super().__init__(message, "LLM_RATE_LIMIT_ERROR")
        self.retry_after = retry_after


class OpenAIAPIAuthenticationError(RAGError):
    """Raised when OpenAI API authentication fails."""

    def __init__(self, message: str = "Authentication failed", status_code: int = 401):
        super().__init__(message, "OPENAI_API_AUTH_ERROR")
        self.status_code = status_code


class OpenAIAPIServerError(RAGError):
    """Raised when OpenAI API server returns an error."""

    def __init__(self, message: str = "Server error", status_code: int = 500):
        super().__init__(message, "OPENAI_API_SERVER_ERROR")
        self.status_code = status_code


class ValidationError(RAGError):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: str = None, value: str = None):
        msg = f"Validation error: {message}"
        if field:
            msg += f" (field: {field})"
        if value:
            msg += f" (value: {value})"
        super().__init__(msg, "VALIDATION_ERROR")
        self.field = field
        self.value = value
