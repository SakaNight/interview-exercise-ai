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

class LLMConfigError(RAGError):
    # Raised when LLM configuration is invalid
    def __init__(self, message: str, config_key: str = None):
        msg = f"LLM configuration error: {message}"
        if config_key:
            msg += f" (key: {config_key})"
        super().__init__(msg, "LLM_CONFIG_ERROR")
        self.config_key = config_key

class LLMRetryError(RAGError):
    # Raised when LLM retry mechanism fails
    def __init__(self, message: str, attempts: int = 0, last_error: str = None):
        msg = f"LLM retry failed after {attempts} attempts: {message}"
        if last_error:
            msg += f" (last error: {last_error})"
        super().__init__(msg, "LLM_RETRY_ERROR")
        self.attempts = attempts
        self.last_error = last_error

class LLMTimeoutError(RAGError):
    # Raised when LLM request times out
    def __init__(self, message: str, timeout_seconds: float = 0):
        msg = f"LLM request timed out: {message}"
        if timeout_seconds > 0:
            msg += f" (timeout: {timeout_seconds}s)"
        super().__init__(msg, "LLM_TIMEOUT_ERROR")
        self.timeout_seconds = timeout_seconds

class LLMRateLimitError(RAGError):
    # Raised when LLM API rate limit is exceeded
    def __init__(self, message: str, retry_after: int = None):
        msg = f"LLM rate limit exceeded: {message}"
        if retry_after:
            msg += f" (retry after: {retry_after}s)"
        super().__init__(msg, "LLM_RATE_LIMIT_ERROR")
        self.retry_after = retry_after

# ===== OpenAI API errors =====
class OpenAIAPIError(RAGError):
    # Base class for OpenAI API-related errors
    def __init__(self, message: str, status_code: int = None, endpoint: str = None):
        msg = f"OpenAI API error: {message}"
        if status_code:
            msg += f" (status: {status_code})"
        if endpoint:
            msg += f" (endpoint: {endpoint})"
        super().__init__(msg, "OPENAI_API_ERROR")
        self.status_code = status_code
        self.endpoint = endpoint

class OpenAIAPIAuthenticationError(OpenAIAPIError):
    # Raised when OpenAI API authentication fails
    def __init__(self, message: str = "Authentication failed", status_code: int = 401):
        super().__init__(message, status_code, "OPENAI_API_AUTH_ERROR")

class OpenAIAPINotFoundError(OpenAIAPIError):
    # Raised when OpenAI API endpoint is not found
    def __init__(self, message: str = "Endpoint not found", status_code: int = 404):
        super().__init__(message, status_code, "OPENAI_API_NOT_FOUND")

class OpenAIAPIServerError(OpenAIAPIError):
    # Raised when OpenAI API server returns an error
    def __init__(self, message: str = "Server error", status_code: int = 500):
        super().__init__(message, status_code, "OPENAI_API_SERVER_ERROR")

# ===== API (resolve-ticket) errors =====
class CustomAPIError(RAGError):
    # Base class for resolve-ticket API-related errors
    def __init__(self, message: str, status_code: int = None, endpoint: str = None):
        msg = f"Custom API error: {message}"
        if status_code:
            msg += f" (status: {status_code})"
        if endpoint:
            msg += f" (endpoint: {endpoint})"
        super().__init__(msg, "CUSTOM_API_ERROR")
        self.status_code = status_code
        self.endpoint = endpoint

class CustomAPIAuthenticationError(CustomAPIError):
    # Raised when resolve-ticket API authentication fails
    def __init__(self, message: str = "Authentication failed", status_code: int = 401):
        super().__init__(message, status_code, "CUSTOM_API_AUTH_ERROR")

class CustomAPINotFoundError(CustomAPIError):
    # Raised when resolve-ticket API endpoint is not found
    def __init__(self, message: str = "Endpoint not found", status_code: int = 404):
        super().__init__(message, status_code, "CUSTOM_API_NOT_FOUND")

class CustomAPIServerError(CustomAPIError):
    # Raised when resolve-ticket API server returns an error
    def __init__(self, message: str = "Server error", status_code: int = 500):
        super().__init__(message, status_code, "CUSTOM_API_SERVER_ERROR")

class CustomAPIValidationError(CustomAPIError):
    # Raised when resolve-ticket API input validation fails
    def __init__(self, message: str = "Validation failed", field: str = None, value: str = None):
        msg = f"Custom API validation error: {message}"
        if field:
            msg += f" (field: {field})"
        if value:
            msg += f" (value: {value})"
        super().__init__(msg, 400, "CUSTOM_API_VALIDATION_ERROR")
        self.field = field
        self.value = value

# ===== Validation errors =====
class ValidationError(RAGError):
    # Base class for validation errors
    def __init__(self, message: str, field: str = None, value: str = None):
        msg = f"Validation error: {message}"
        if field:
            msg += f" (field: {field})"
        if value:
            msg += f" (value: {value})"
        super().__init__(msg, "VALIDATION_ERROR")
        self.field = field
        self.value = value

class SchemaValidationError(ValidationError):
    # Raised when JSON schema validation fails
    def __init__(self, message: str, schema_path: str = None):
        msg = f"Schema validation failed: {message}"
        if schema_path:
            msg += f" (schema: {schema_path})"
        super().__init__(msg, "SCHEMA_VALIDATION_ERROR")
        self.schema_path = schema_path

# ===== System/Configuration errors =====
class ConfigurationError(RAGError):
    # Raised when system configuration is invalid
    def __init__(self, message: str, config_file: str = None):
        msg = f"Configuration error: {message}"
        if config_file:
            msg += f" (file: {config_file})"
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_file = config_file

class EnvironmentError(RAGError):
    # Raised when required environment variables are missing
    def __init__(self, message: str, missing_vars: list = None):
        msg = f"Environment error: {message}"
        if missing_vars:
            msg += f" (missing: {', '.join(missing_vars)})"
        super().__init__(msg, "ENVIRONMENT_ERROR")
        self.missing_vars = missing_vars

class ServiceUnavailableError(RAGError):
    # Raised when a required service is unavailable
    def __init__(self, message: str, service_name: str = None, retry_after: int = None):
        msg = f"Service unavailable: {message}"
        if service_name:
            msg += f" (service: {service_name})"
        if retry_after:
            msg += f" (retry after: {retry_after}s)"
        super().__init__(msg, "SERVICE_UNAVAILABLE")
        self.service_name = service_name
        self.retry_after = retry_after