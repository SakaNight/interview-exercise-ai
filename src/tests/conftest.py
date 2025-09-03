"""
Pytest Configuration and Shared Fixtures.

This module provides shared pytest fixtures and configuration for all test modules,
including mock services, test data, and environment setup.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from . import (
    MOCK_LLM_RESPONSE_DATA,
    MOCK_RAG_RESULTS,
    cleanup_test_environment,
    setup_test_environment,
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Setup test environment for the entire test session."""
    setup_test_environment()
    yield
    cleanup_test_environment()


@pytest.fixture(scope="function")
def temp_test_dir():
    """Provide a temporary directory for each test."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def mock_rag_pipeline():
    """Provide a mocked RAG pipeline for testing."""
    mock_pipeline = Mock()
    mock_pipeline.search.return_value = MOCK_RAG_RESULTS
    mock_pipeline.setup_pipeline.return_value = True
    mock_pipeline.documents = [
        {"title": "Test Doc 1", "section": "Test", "content": "Test content"},
        {"title": "Test Doc 2", "section": "Test", "content": "Test content"},
    ]
    return mock_pipeline


@pytest.fixture(scope="function")
def mock_llm_service():
    """Provide a mocked LLM service for testing."""
    mock_service = Mock()

    from llm import MCPResponse

    success_response = MCPResponse(
        answer=MOCK_LLM_RESPONSE_DATA["success"]["answer"],
        references=MOCK_LLM_RESPONSE_DATA["success"]["references"],
        action_required=MOCK_LLM_RESPONSE_DATA["success"]["action_required"],
    )
    mock_service.generate_response.return_value = success_response

    return mock_service


@pytest.fixture(scope="function")
def mock_llm_service_no_docs():
    """Provide a mocked LLM service that returns no-docs response."""
    mock_service = Mock()

    from llm import MCPResponse

    no_docs_response = MCPResponse(
        answer=MOCK_LLM_RESPONSE_DATA["no_docs"]["answer"],
        references=MOCK_LLM_RESPONSE_DATA["no_docs"]["references"],
        action_required=MOCK_LLM_RESPONSE_DATA["no_docs"]["action_required"],
    )
    mock_service.generate_response.return_value = no_docs_response

    return mock_service


@pytest.fixture(scope="function")
def sample_ticket_requests():
    return {
        "valid": {"ticket_text": "I cannot access my domain, it says invalid password"},
        "empty": {"ticket_text": ""},
        "long": {"ticket_text": "A" * 5001},  # Exceeds max length
        "missing_field": {"invalid_field": "test"},
    }


@pytest.fixture(scope="function")
def expected_response_structure():
    return {
        "success": True,
        "data": {"answer": str, "references": list, "action_required": str},
        "processing_time": float,
        "documents_retrieved": int,
    }


@pytest.fixture(scope="function")
def valid_action_types():
    return ["none", "escalate_to_support", "escalate_to_abuse_team", "contact_customer"]


@pytest.fixture(scope="function")
def performance_thresholds():
    return {
        "max_response_time": 30.0,  # seconds
        "max_processing_time": 25.0,  # seconds
        "min_documents_retrieved": 0,
        "max_documents_retrieved": 10,
    }


@pytest.fixture(scope="function")
def expected_error_codes():
    return {
        "validation_error": 422,
        "service_unavailable": 503,
        "internal_error": 500,
        "not_found": 404,
    }


# Mock environment variables
@pytest.fixture(scope="function")
def mock_env_vars():
    with patch.dict(
        "os.environ", {"TESTING": "true", "LOG_LEVEL": "INFO", "API_PORT": "8000"}
    ):
        yield


@pytest.fixture(scope="function")
def mock_file_system():
    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.mkdir") as mock_mkdir,
        patch("pathlib.Path.unlink") as mock_unlink,
    ):
        mock_exists.return_value = True
        mock_mkdir.return_value = None
        mock_unlink.return_value = None

        yield {"exists": mock_exists, "mkdir": mock_mkdir, "unlink": mock_unlink}


@pytest.fixture(scope="function")
def capture_logs(caplog):
    caplog.set_level("INFO")
    return caplog


@pytest.fixture(scope="function")
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def generate_test_documents():
    def _generate(count=3, title_prefix="Test Doc", content_length=100):
        docs = []
        for i in range(count):
            docs.append(
                {
                    "title": f"{title_prefix} {i+1}",
                    "section": f"Section {i+1}",
                    "content": f"Test content for document {i+1}. "
                    * (content_length // 30),
                    "ref": f"test-ref-{i+1}",
                    "tags": [f"tag{i+1}", f"category{i+1}"],
                }
            )
        return docs

    return _generate
