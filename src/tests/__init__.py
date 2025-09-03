"""
Test Suite Utilities and Configuration.

This module provides shared utilities, mock data, and configuration
for the comprehensive test suite covering RAG pipeline, LLM service,
and API endpoints.
"""

import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

TEST_DATA_DIR = SRC_DIR / "data"
TEST_INDEX_PATH = SRC_DIR / "test_faiss_index.bin"
TEST_DOCS_PATH = TEST_DATA_DIR / "docs.json"

# Mock data for testing
# Note: RAG search returns (doc_idx, score, doc) format
MOCK_RAG_RESULTS = [
    (0, 0.95, {"title": "Domain Access Guide", "content": "How to access your domain"}),
    (1, 0.87, {"title": "Password Reset", "content": "Steps to reset your password"}),
    (
        2,
        0.82,
        {
            "title": "Account Security",
            "content": "Security best practices for your account",
        },
    ),
]

MOCK_LLM_RESPONSE_DATA = {
    "success": {
        "answer": "Based on the documentation, you need to reset your password. Please check your email for reset instructions.",
        "references": [
            {
                "doc_id": "doc_001",
                "title": "Domain Access Guide",
                "section": "Access Methods",
                "url": None,
            },
            {
                "doc_id": "doc_002",
                "title": "Password Reset",
                "section": "Reset Process",
                "url": None,
            },
        ],
        "action_required": "contact_customer",
    },
    "no_docs": {
        "answer": "I couldn't find specific documentation for your issue. Please contact support for assistance.",
        "references": [],
        "action_required": "escalate_to_support",
    },
}


def get_test_data_path(filename: str) -> Path:
    return TEST_DATA_DIR / filename


def create_mock_ticket_data(text: str = None) -> dict:
    if text is None:
        text = "I cannot access my domain, it says invalid password"

    return {"ticket_text": text}


def create_mock_context_docs(count: int = 2) -> list:
    docs = []
    for i in range(count):
        docs.append(
            {
                "title": f"Test Document {i+1}",
                "section": f"Test Section {i+1}",
                "content": f"This is test content for document {i+1}",
                "ref": f"test-doc-{i+1}",
            }
        )
    return docs


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "api_tests" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "rag_tests" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "llm_tests" in item.nodeid:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def test_data_dir():
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def mock_rag_results():
    return MOCK_RAG_RESULTS


@pytest.fixture(scope="session")
def mock_llm_response_data():
    return MOCK_LLM_RESPONSE_DATA


@pytest.fixture
def sample_ticket_data():
    return create_mock_ticket_data()


@pytest.fixture
def sample_context_docs():
    return create_mock_context_docs()


def setup_test_environment():
    TEST_DATA_DIR.mkdir(exist_ok=True)

    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "INFO")


def cleanup_test_environment():
    if TEST_INDEX_PATH.exists():
        TEST_INDEX_PATH.unlink()


from .conftest import *  # This will import any conftest.py configurations

__version__ = "1.0.0"
__author__ = "Test Suite"
__description__ = "Comprehensive test suite for RAG system"

__all__ = [
    "TEST_DATA_DIR",
    "TEST_INDEX_PATH",
    "MOCK_RAG_RESULTS",
    "MOCK_LLM_RESPONSE_DATA",
    "create_mock_ticket_data",
    "create_mock_context_docs",
    "setup_test_environment",
    "cleanup_test_environment",
]
