"""
API Endpoint Unit Tests.

This module contains unit tests for the FastAPI endpoints, including
health checks, ticket resolution, and error handling scenarios.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from llm import MCPResponse
from main import app

from . import MOCK_LLM_RESPONSE_DATA, create_mock_ticket_data


class TestHealthEndpoint:
    """Test cases for the health check endpoint."""

    def test_health_endpoint_success(self):
        """Test successful health endpoint response."""
        # Create TestClient after mocking to avoid real service initialization
        with (
            patch("main.RAGPipeline") as MockRAGPipeline,
            patch("main.LLMService") as MockLLMService,
        ):
            # Mock the classes to return lightweight instances
            mock_rag = Mock()
            mock_rag.setup_pipeline.return_value = True
            mock_rag.documents = []
            MockRAGPipeline.return_value = mock_rag

            mock_llm = Mock()
            MockLLMService.return_value = mock_llm

            # Now create TestClient - lifespan will use our mocked classes
            client = TestClient(app)

            response = client.get("/")

            assert response.status_code == 200

            data = response.json()
            assert data["message"] == "Ticket Resolution API"
            assert data["version"] == "1.0.0"
            assert data["status"] == "healthy"
            assert "performance" in data

            performance = data["performance"]
            assert "uptime_seconds" in performance
            assert "total_requests" in performance
            assert "avg_response_time" in performance

            # Check that uptime is a positive number
            assert performance["uptime_seconds"] >= 0
            assert performance["total_requests"] >= 0
            assert performance["avg_response_time"] >= 0

    def test_health_endpoint_structure(self):
        """Test health endpoint response structure validation."""
        with (
            patch("main.RAGPipeline") as MockRAGPipeline,
            patch("main.LLMService") as MockLLMService,
        ):
            mock_rag = Mock()
            mock_rag.setup_pipeline.return_value = True
            mock_rag.documents = []
            MockRAGPipeline.return_value = mock_rag

            mock_llm = Mock()
            MockLLMService.return_value = mock_llm

            client = TestClient(app)

            response = client.get("/")

            assert response.status_code == 200
            data = response.json()

            required_fields = ["message", "version", "status", "performance"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            performance_fields = [
                "uptime_seconds",
                "total_requests",
                "avg_response_time",
            ]
            for field in performance_fields:
                assert (
                    field in data["performance"]
                ), f"Missing performance field: {field}"

    def test_health_endpoint_performance_headers(self):
        """Test health endpoint performance headers."""
        with (
            patch("main.RAGPipeline") as MockRAGPipeline,
            patch("main.LLMService") as MockLLMService,
        ):
            mock_rag = Mock()
            mock_rag.setup_pipeline.return_value = True
            mock_rag.documents = []
            MockRAGPipeline.return_value = mock_rag

            mock_llm = Mock()
            MockLLMService.return_value = mock_llm

            client = TestClient(app)

            response = client.get("/")

            assert response.status_code == 200

            assert "X-Response-Time" in response.headers
            assert "X-Request-ID" in response.headers

            response_time_header = response.headers["X-Response-Time"]
            assert response_time_header.endswith("s")

            request_id = response.headers["X-Request-ID"]
            assert request_id.isdigit()


class TestResolveTicketEndpoint:
    """Test cases for the ticket resolution endpoint."""

    def test_resolve_ticket_success(self):
        """Test successful ticket resolution with mocked services."""
        mock_rag = Mock()
        mock_rag.search.return_value = [
            (
                0,
                0.95,
                {
                    "title": "Domain Access Guide",
                    "content": "How to access your domain",
                },
            ),
            (
                1,
                0.87,
                {"title": "Password Reset", "content": "Steps to reset your password"},
            ),
        ]
        mock_rag.setup_pipeline.return_value = True
        mock_rag.documents = []

        mock_llm = Mock()
        mock_response = MCPResponse(
            answer=MOCK_LLM_RESPONSE_DATA["success"]["answer"],
            references=MOCK_LLM_RESPONSE_DATA["success"]["references"],
            action_required=MOCK_LLM_RESPONSE_DATA["success"]["action_required"],
        )
        mock_llm.generate_response.return_value = mock_response

        with patch("main.rag_pipeline", mock_rag), patch("main.llm_service", mock_llm):
            client = TestClient(app)

            ticket_data = create_mock_ticket_data()

            response = client.post("/resolve-ticket", json=ticket_data)

            assert response.status_code == 200

            response_data = response.json()
            assert "answer" in response_data
            assert "references" in response_data
            assert "action_required" in response_data

            assert "success" not in response_data
            assert "data" not in response_data
            assert "processing_time" not in response_data
            assert "documents_retrieved" not in response_data

            assert isinstance(response_data["answer"], str)
            assert isinstance(response_data["references"], list)
            assert isinstance(response_data["action_required"], str)

            valid_actions = [
                "none",
                "escalate_to_support",
                "escalate_to_abuse_team",
                "contact_customer",
            ]
            assert response_data["action_required"] in valid_actions

            for ref in response_data["references"]:
                assert "doc_id" in ref
                assert "title" in ref
                assert "section" in ref
                assert "url" in ref  # Can be None

            mock_rag.search.assert_called_once_with(ticket_data["ticket_text"], k=3)
            mock_llm.generate_response.assert_called_once()

    def test_resolve_ticket_no_documents_found(self):
        """Test ticket resolution when no relevant documents are found."""
        mock_rag = Mock()
        mock_rag.search.return_value = []
        mock_rag.setup_pipeline.return_value = True
        mock_rag.documents = []

        mock_llm = Mock()
        mock_response = MCPResponse(
            answer=MOCK_LLM_RESPONSE_DATA["no_docs"]["answer"],
            references=MOCK_LLM_RESPONSE_DATA["no_docs"]["references"],
            action_required=MOCK_LLM_RESPONSE_DATA["no_docs"]["action_required"],
        )
        mock_llm.generate_response.return_value = mock_response

        with patch("main.rag_pipeline", mock_rag), patch("main.llm_service", mock_llm):
            client = TestClient(app)

            ticket_data = create_mock_ticket_data(
                "Very specific technical issue not covered in docs"
            )

            response = client.post("/resolve-ticket", json=ticket_data)

            assert response.status_code == 200

            response_data = response.json()
            assert "answer" in response_data
            assert "references" in response_data
            assert "action_required" in response_data

            assert "success" not in response_data
            assert "data" not in response_data
            assert "processing_time" not in response_data
            assert "documents_retrieved" not in response_data

            assert response_data["action_required"] == "escalate_to_support"
            assert response_data["references"] == []

    def test_resolve_ticket_validation_error_missing_field(self):
        """Test validation error when required field is missing."""
        with (
            patch("main.RAGPipeline") as MockRAGPipeline,
            patch("main.LLMService") as MockLLMService,
        ):
            mock_rag = Mock()
            mock_rag.setup_pipeline.return_value = True
            mock_rag.documents = []
            MockRAGPipeline.return_value = mock_rag

            mock_llm = Mock()
            MockLLMService.return_value = mock_llm

            client = TestClient(app)

            invalid_data = {"invalid_field": "test"}

            response = client.post("/resolve-ticket", json=invalid_data)

            assert response.status_code == 422

            error_data = response.json()
            assert "detail" in error_data

    def test_resolve_ticket_validation_error_empty_text(self):
        """Test validation error when ticket text is empty."""
        with (
            patch("main.RAGPipeline") as MockRAGPipeline,
            patch("main.LLMService") as MockLLMService,
        ):
            mock_rag = Mock()
            mock_rag.setup_pipeline.return_value = True
            mock_rag.documents = []
            MockRAGPipeline.return_value = mock_rag

            mock_llm = Mock()
            MockLLMService.return_value = mock_llm

            client = TestClient(app)

            empty_data = {"ticket_text": ""}

            response = client.post("/resolve-ticket", json=empty_data)

            assert response.status_code == 422

            error_data = response.json()
            assert "detail" in error_data

    def test_resolve_ticket_validation_error_long_text(self):
        """Test validation error when ticket text exceeds maximum length."""
        with (
            patch("main.RAGPipeline") as MockRAGPipeline,
            patch("main.LLMService") as MockLLMService,
        ):
            mock_rag = Mock()
            mock_rag.setup_pipeline.return_value = True
            mock_rag.documents = []
            MockRAGPipeline.return_value = mock_rag

            mock_llm = Mock()
            MockLLMService.return_value = mock_llm

            client = TestClient(app)

            long_text = "A" * 5001  # Exceeds max length of 5000
            long_data = {"ticket_text": long_text}

            response = client.post("/resolve-ticket", json=long_data)

            assert response.status_code == 422

            error_data = response.json()
            assert "detail" in error_data

    def test_resolve_ticket_performance_headers(self):
        """Test that the resolve-ticket endpoint includes performance headers."""
        mock_rag = Mock()
        mock_rag.search.return_value = []
        mock_rag.setup_pipeline.return_value = True
        mock_rag.documents = []

        mock_llm = Mock()
        mock_response = MCPResponse(
            answer="Test response", references=[], action_required="none"
        )
        mock_llm.generate_response.return_value = mock_response

        with patch("main.rag_pipeline", mock_rag), patch("main.llm_service", mock_llm):
            client = TestClient(app)

            ticket_data = {"ticket_text": "Test performance headers"}

            response = client.post("/resolve-ticket", json=ticket_data)

            assert response.status_code == 200

            assert "X-Response-Time" in response.headers
            assert "X-Request-ID" in response.headers

            response_time_header = response.headers["X-Response-Time"]
            assert response_time_header.endswith("s")

            request_id = response.headers["X-Request-ID"]
            assert request_id.isdigit()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
