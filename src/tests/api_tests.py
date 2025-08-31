import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from main import app
from llm import MCPResponse

# Import test utilities and fixtures
from . import (
    MOCK_RAG_RESULTS,
    MOCK_LLM_RESPONSE_DATA,
    create_mock_ticket_data,
    create_mock_context_docs
)

class TestHealthEndpoint:    
    def test_health_endpoint_success(self):
        # Create TestClient after mocking to avoid real service initialization
        with patch('main.RAGPipeline') as MockRAGPipeline, \
             patch('main.LLMService') as MockLLMService:
            
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
        with patch('main.RAGPipeline') as MockRAGPipeline, \
             patch('main.LLMService') as MockLLMService:
            
            # Mock the classes
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
            
            # Verify all required fields are present
            required_fields = ["message", "version", "status", "performance"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Verify performance sub-structure
            performance_fields = ["uptime_seconds", "total_requests", "avg_response_time"]
            for field in performance_fields:
                assert field in data["performance"], f"Missing performance field: {field}"
    
    def test_health_endpoint_performance_headers(self):
        with patch('main.RAGPipeline') as MockRAGPipeline, \
             patch('main.LLMService') as MockLLMService:
            
            # Mock the classes
            mock_rag = Mock()
            mock_rag.setup_pipeline.return_value = True
            mock_rag.documents = []
            MockRAGPipeline.return_value = mock_rag
            
            mock_llm = Mock()
            MockLLMService.return_value = mock_llm
            
            client = TestClient(app)
            
            response = client.get("/")
            
            assert response.status_code == 200
            
            # Check for performance headers
            assert "X-Response-Time" in response.headers
            assert "X-Request-ID" in response.headers
            
            # Verify response time header format
            response_time_header = response.headers["X-Response-Time"]
            assert response_time_header.endswith("s")
            
            # Verify request ID header format
            request_id = response.headers["X-Request-ID"]
            assert request_id.isdigit()

# Test cases for the /resolve-ticket endpoint with mocked RAG/LLM services
class TestResolveTicketEndpoint:   
    # test successful ticket resolution with mocked services
    def test_resolve_ticket_success(self):
        mock_rag = Mock()
        mock_rag.search.return_value = [
            (0, 0.95, {"title": "Domain Access Guide", "content": "How to access your domain"}),
            (1, 0.87, {"title": "Password Reset", "content": "Steps to reset your password"})
        ]
        mock_rag.setup_pipeline.return_value = True
        mock_rag.documents = []
        
        mock_llm = Mock()
        mock_response = MCPResponse(
            answer=MOCK_LLM_RESPONSE_DATA["success"]["answer"],
            references=MOCK_LLM_RESPONSE_DATA["success"]["references"],
            action_required=MOCK_LLM_RESPONSE_DATA["success"]["action_required"]
        )
        mock_llm.generate_response.return_value = mock_response
        
        # Patch the global variables directly
        with patch('main.rag_pipeline', mock_rag), \
             patch('main.llm_service', mock_llm):
        
            client = TestClient(app)
            
            # Test data using utility function
            ticket_data = create_mock_ticket_data()
            
            response = client.post("/resolve-ticket", json=ticket_data)
            
            # Verify response
            assert response.status_code == 200
            
            response_data = response.json()
            assert response_data["success"] is True
            assert "data" in response_data
            assert "processing_time" in response_data
            assert "documents_retrieved" in response_data
            
            # Verify MCPResponse structure
            mcp_data = response_data["data"]
            assert "answer" in mcp_data
            assert "references" in mcp_data
            assert "action_required" in mcp_data
            
            # Verify action_required is valid
            valid_actions = ['none', 'escalate_to_support', 'escalate_to_abuse_team', 'contact_customer']
            assert mcp_data["action_required"] in valid_actions
            
            # Verify processing metrics
            assert response_data["processing_time"] > 0
            assert response_data["documents_retrieved"] == 2
            
            # Verify mocked services were called
            mock_rag.search.assert_called_once_with(ticket_data["ticket_text"], k=3)
            mock_llm.generate_response.assert_called_once()
    
    # test ticket resolution when no relevant documents are found
    def test_resolve_ticket_no_documents_found(self):
        mock_rag = Mock()
        mock_rag.search.return_value = []
        mock_rag.setup_pipeline.return_value = True
        mock_rag.documents = []
        
        mock_llm = Mock()
        mock_response = MCPResponse(
            answer=MOCK_LLM_RESPONSE_DATA["no_docs"]["answer"],
            references=MOCK_LLM_RESPONSE_DATA["no_docs"]["references"],
            action_required=MOCK_LLM_RESPONSE_DATA["no_docs"]["action_required"]
        )
        mock_llm.generate_response.return_value = mock_response
        
        # Patch the global variables directly
        with patch('main.rag_pipeline', mock_rag), \
             patch('main.llm_service', mock_llm):
            
            client = TestClient(app)
            
            ticket_data = create_mock_ticket_data("Very specific technical issue not covered in docs")
            
            response = client.post("/resolve-ticket", json=ticket_data)
            
            assert response.status_code == 200
            
            response_data = response.json()
            assert response_data["success"] is True
            assert response_data["documents_retrieved"] == 0
            assert response_data["data"]["action_required"] == "escalate_to_support"
    
    def test_resolve_ticket_validation_error_missing_field(self):
        with patch('main.RAGPipeline') as MockRAGPipeline, \
             patch('main.LLMService') as MockLLMService:
            
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
        with patch('main.RAGPipeline') as MockRAGPipeline, \
             patch('main.LLMService') as MockLLMService:
            
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
        with patch('main.RAGPipeline') as MockRAGPipeline, \
             patch('main.LLMService') as MockLLMService:
            
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
    
    # test that the resolve-ticket endpoint includes performance headers
    def test_resolve_ticket_performance_headers(self):
        mock_rag = Mock()
        mock_rag.search.return_value = []
        mock_rag.setup_pipeline.return_value = True
        mock_rag.documents = []
        
        mock_llm = Mock()
        mock_response = MCPResponse(
            answer="Test response",
            references=[],
            action_required="none"
        )
        mock_llm.generate_response.return_value = mock_response
        
        # Patch the global variables directly
        with patch('main.rag_pipeline', mock_rag), \
             patch('main.llm_service', mock_llm):
            
            client = TestClient(app)
            
            ticket_data = {"ticket_text": "Test performance headers"}
            
            response = client.post("/resolve-ticket", json=ticket_data)
            
            assert response.status_code == 200
            
            # Check for performance headers
            assert "X-Response-Time" in response.headers
            assert "X-Request-ID" in response.headers
            
            # Verify response time header format
            response_time_header = response.headers["X-Response-Time"]
            assert response_time_header.endswith("s")
            
            # Verify request ID header format
            request_id = response.headers["X-Request-ID"]
            assert request_id.isdigit()

# Test configuration for pytest
@pytest.fixture(autouse=True)
def setup_test_environment():

    pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
