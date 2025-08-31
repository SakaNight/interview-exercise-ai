import os
import sys
import pytest
from pathlib import Path

# Add src directory to path for consistent imports across all test files
SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

# Test configuration constants
TEST_DATA_DIR = SRC_DIR / "data"
TEST_INDEX_PATH = SRC_DIR / "test_faiss_index.bin"
TEST_DOCS_PATH = TEST_DATA_DIR / "docs.json"

# Mock data for testing
# Note: RAG search returns (doc_idx, score, doc) format
MOCK_RAG_RESULTS = [
    (0, 0.95, {"title": "Domain Access Guide", "content": "How to access your domain"}),
    (1, 0.87, {"title": "Password Reset", "content": "Steps to reset your password"}),
    (2, 0.82, {"title": "Account Security", "content": "Security best practices for your account"})
]

MOCK_LLM_RESPONSE_DATA = {
    "success": {
        "answer": "Based on the documentation, you need to reset your password. Please check your email for reset instructions.",
        "references": ["Domain Access Guide", "Password Reset"],
        "action_required": "contact_customer"
    },
    "no_docs": {
        "answer": "I couldn't find specific documentation for your issue. Please contact support for assistance.",
        "references": [],
        "action_required": "escalate_to_support"
    }
}

# Test utilities
def get_test_data_path(filename: str) -> Path:
    return TEST_DATA_DIR / filename

def create_mock_ticket_data(text: str = None) -> dict:
    if text is None:
        text = "I cannot access my domain, it says invalid password"
    
    return {
        "ticket_text": text
    }

def create_mock_context_docs(count: int = 2) -> list:
    docs = []
    for i in range(count):
        docs.append({
            "title": f"Test Document {i+1}",
            "section": f"Test Section {i+1}",
            "content": f"This is test content for document {i+1}",
            "ref": f"test-doc-{i+1}"
        })
    return docs

# Pytest configuration with custom markers and settings
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

# Automatically mark tests based on their names
def pytest_collection_modifyitems(config, items):
    for item in items:
        # Mark API tests as unit tests
        if "api_tests" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        # Mark RAG tests as integration tests (they use real models)
        elif "rag_tests" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        # Mark LLM tests as integration tests
        elif "llm_tests" in item.nodeid:
            item.add_marker(pytest.mark.integration)

# Test fixtures that can be shared across test files
@pytest.fixture(scope="session")
def test_data_dir():
    return TEST_DATA_DIR

# Provide consistent mock RAG results for testing
@pytest.fixture(scope="session")
def mock_rag_results():
    return MOCK_RAG_RESULTS

# Provide consistent mock LLM response data for testing
@pytest.fixture(scope="session")
def mock_llm_response_data():
    return MOCK_LLM_RESPONSE_DATA

@pytest.fixture
def sample_ticket_data():
    return create_mock_ticket_data()

@pytest.fixture
def sample_context_docs():
    return create_mock_context_docs()

# Environment setup helpers
def setup_test_environment():

    # Ensure test data directory exists
    TEST_DATA_DIR.mkdir(exist_ok=True)
    
    # Set test-specific environment variables
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "INFO")

def cleanup_test_environment():
    # Remove test-specific files if they exist
    if TEST_INDEX_PATH.exists():
        TEST_INDEX_PATH.unlink()

# Import commonly used test utilities
from .conftest import *  # This will import any conftest.py configurations

# Version and metadata
__version__ = "1.0.0"
__author__ = "Test Suite"
__description__ = "Comprehensive test suite for RAG system"

# Export useful items for easy access
__all__ = [
    "TEST_DATA_DIR",
    "TEST_INDEX_PATH", 
    "MOCK_RAG_RESULTS",
    "MOCK_LLM_RESPONSE_DATA",
    "create_mock_ticket_data",
    "create_mock_context_docs",
    "setup_test_environment",
    "cleanup_test_environment"
]