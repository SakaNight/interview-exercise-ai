# AI Knowledge Assistant for Support Team

## Introduction
This project builds an AI assistant for supporting customer tickets responding based on the relevant supporting documents. It uses a **Retrieval-Augmented Generation (RAG)** pipeline powered by an **LLM** and follow the **Model Context Protocol (MCP)** to produce structured output.

> **One-Click Setup**: Clone, configure `.env`, and run `docker-compose --profile dev up --build` to get a fully working system.

## Tech Stack and Design Explanation
- **Languages & API**: Python + FastAPI
The combination is light-weighted, high-performance, integrated with OpenAPI, suitable for quick development and testing
- **Embedding Models**: sentence-transformers/all-MiniLM-L6-v2 (baseline, fast, widely used), BAAI/bge-small-en-v1.5 (better quality, same dimension = 384)
- **Vector Store**: FAISS (lightweight, file persistence). Future switch/extent to Qdrant (metadata filtering & clustering)
- **LLMs**: OpenAI gpt-4o-mini (temperature 0 for deterministic JSON)
- **Configuration Management**: Pydantic Settings to manage all the models, index, API keys etc. for easy switch

## Configuration
- **Allowed Actions**: The system supports four action types for customer support tickets:
  - `none`: Sufficient information provided, no further action needed
  - `escalate_to_support`: Complex technical issue or insufficient context
  - `escalate_to_abuse_team`: Security, abuse, or policy violation concerns
  - `contact_customer`: Need additional information from customer
- **Max References**: Default limit of 3 document references per response to maintain concise answers
- **Top K Results**: Default of 3 most relevant documents retrieved from the knowledge base

## Prompt & MCP Design
The system implements a **Model Context Protocol (MCP)** compliant design with strict JSON Schema validation and robust retry mechanisms.

### **Role**
You are a specialized knowledge assistant for customer support ticket resolution. Your expertise is in analyzing customer queries and providing structured, accurate responses based on provided documentation.

### **Context**
You will receive:
- Customer support ticket text
- Top-K retrieved documents with structured metadata (ID, title, section, content)
- Clear task instructions and output schema requirements

### **Task**
Your task is to:
1. Analyze the customer support query carefully
2. Use ONLY the provided context documents to formulate your response
3. Provide accurate, helpful answers based on the documentation
4. Determine the appropriate action required based on the query complexity
5. Return your response in the exact JSON schema format specified

### **Output Schema (Strict)**
```json
{
  "answer": "string (required) - Your detailed answer based on the provided context",
  "references": [
    {
      "doc_id": "string (required) - Document ID from the provided context",
      "title": "string (required) - Document title from the provided context", 
      "section": "string (required) - Document section from the provided context",
      "url": "string (optional) - URL if available, null otherwise"
    }
  ],
  "action_required": "string (required) - One of: 'none', 'escalate_to_support', 'escalate_to_abuse_team', 'contact_customer'"
}
```

### **Validation & Retry**
- **Strict JSON Schema**: Validates required fields, types, lengths, and enum values
- **3-Retry Mechanism**: Progressive enhancement with increasingly strict instructions
- **Error Handling**: Comprehensive validation with detailed error reporting
- **Service-Side Fallback**: Robust retry logic for rate limits, timeouts, and validation failures

## Project Structure

```
interview-exercise-ai/
├── src/                         # Source code directory
│   ├── data/                    # Data and configuration files
│   │   └── docs.json            # Knowledge base documents (JSON format)
│   ├── scripts/                 # Utility and setup scripts
│   │   └── prepare_index.py     # Index preparation and management script
│   ├── tests/                   # Test suite
│   │   ├── __init__.py          # Test package initialization
│   │   ├── api_tests.py         # API endpoint tests with mocked services
│   │   ├── conftest.py          # Pytest configuration and fixtures
│   │   ├── golden_cases_test.py # Golden cases regression tests for core functionality
│   │   ├── llm_tests.py         # LLM service functionality tests
│   │   └── rag_tests.py         # RAG pipeline integration tests
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── llm.py                   # LLM service with MCP-compliant prompt design
│   ├── main.py                  # FastAPI application and endpoints
│   ├── models.py                # ML model abstractions (embedding, LLM)
│   ├── rag.py                   # RAG pipeline implementation
│   └── settings.py              # Configuration management with Pydantic
├── .github/                     # GitHub workflows and templates
│   └── workflows/               # CI/CD pipeline definitions
│       └── test.yml             # Automated testing workflow
├── docs/                        # Documentation
│   └── SELF_CHECK.md            # Developer self-check scripts and tests
├── index/                       # Vector index storage
│   └── faiss_index.bin          # FAISS vector index file
├── .cache/                      # Cache directories (gitignored)
│   └── hf/                      # Hugging Face model cache
├── .dockerignore                # Docker build context exclusions
├── .env                         # Environment variables (gitignored)
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules for Python/AI projects
├── docker-compose.yml           # Multi-environment Docker orchestration
├── Dockerfile                   # Container image definition
├── nginx.conf                   # Nginx reverse proxy configuration
├── README.md                    # Project documentation
├── READMETC.md                  # Technical challenge requirements
└── requirements.txt             # Python dependencies
```

### **Core Components**

| Component | File | Purpose |
|-----------|------|---------|
| **API Layer** | `main.py` | FastAPI application, endpoints, middleware, error handling |
| **RAG Pipeline** | `rag.py` | Document retrieval, embedding, FAISS index management |
| **LLM Service** | `llm.py` | MCP-compliant prompt design, JSON validation, retry logic |
| **Model Abstractions** | `models.py` | Embedding and LLM model wrappers for provider flexibility |
| **Configuration** | `settings.py` | Pydantic-based settings management with environment variables |
| **Exception Handling** | `exceptions.py` | Custom exception hierarchy for different error types |
| **Test Suite** | `tests/` | Comprehensive testing with mocked services and fixtures |

### **Data Flow**

```
docs.json      → RAG Pipeline  → LLM Service    → MCP Response
     ↓                 ↓                ↓              ↓
Knowledge Base → Vector Search → Structured Gen → JSON Output
```

### **Deployment Architecture**

```
Docker Container
├── Python 3.11 Runtime
├── Dependencies (requirements.txt)
├── FastAPI Application
├── FAISS Index
└── Performance Monitoring
```

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Local Development
```bash
git clone <repo>
pip install -r requirements.txt
cp .env.example .env
python src/main.py
```

### API Server
```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```


### API Endpoints

#### Health Check
**GET** `/`
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "Ticket Resolution API",
  "version": "1.0.0",
  "status": "healthy",
  "performance": {
    "uptime_seconds": 1234.56,
    "total_requests": 42,
    "avg_response_time": 0.0456
  }
}
```

#### Resolve Support Ticket
**POST** `/resolve-ticket`
```bash
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I cannot access my domain, it says invalid password"
  }'
```

**Response (MCP):**
```json
{
  "answer": "Based on the documentation, you need to reset your password. Please check your email for reset instructions.",
  "references": [
    {
      "doc_id": "doc_001",
      "title": "Domain Access Guide",
      "section": "Password Management",
      "url": null
    },
    {
      "doc_id": "doc_002", 
      "title": "Password Reset",
      "section": "Reset Process",
      "url": null
    }
  ],
  "action_required": "contact_customer"
}
```

#### Debug Endpoint (Internal Use)
**POST** `/resolve-ticket/debug`
```bash
curl -X POST http://localhost:8000/resolve-ticket/debug \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I cannot access my domain, it says invalid password"
  }'
```

**Response (Envelope Format):**
```json
{
  "success": true,
  "data": {
    "answer": "Based on the documentation, you need to reset your password. Please check your email for reset instructions.",
    "references": [
      {
        "doc_id": "doc_001",
        "title": "Domain Access Guide",
        "section": "Password Management",
        "url": null
      },
      {
        "doc_id": "doc_002", 
        "title": "Password Reset",
        "section": "Reset Process",
        "url": null
      }
    ],
    "action_required": "contact_customer"
  },
  "processing_time": 1.234,
  "documents_retrieved": 2
}
```

## Docker Deployment

### One-Click Setup & Verification

**1. Clone and Start:**
```bash
git clone <repo>
cd interview-exercise-ai
cp .env.example .env  # Edit with your OpenAI API key
docker-compose --profile dev up --build
```

**2. Wait for startup (30-60 seconds), then verify:**

```bash
# Health check
curl http://localhost:8000/ | jq

# Test main endpoint (should return flat MCP JSON)
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_text":"My domain was suspended and I did not get any notice."}' | jq

# Test debug endpoint (should return envelope format)
curl -X POST http://localhost:8000/resolve-ticket/debug \
  -H "Content-Type: application/json" \
  -d '{"ticket_text":"How to transfer my domain?"}' | jq

# Test performance metrics
curl http://localhost:8000/metrics | jq
```

**Expected Results:**
- Health check returns `{"status": "healthy"}`
- Main endpoint returns flat MCP JSON with `answer`, `references`, `action_required`
- Debug endpoint returns envelope with `processing_time` and `documents_retrieved`
- Metrics endpoint returns performance statistics

### Production Deployment

```bash
docker-compose --profile prod up --build
```

## Testing

### Test Structure

The project includes comprehensive tests organized into different modules:

- **API Tests** (`tests/api_tests.py`): Endpoint testing with mocked services
- **Golden Cases Tests** (`tests/golden_cases_test.py`): Regression tests for core functionality and critical user scenarios
- **LLM Tests** (`tests/llm_tests.py`): LLM service functionality testing
- **RAG Tests** (`tests/rag_tests.py`): RAG pipeline testing
- **Integration Tests**: End-to-end functionality testing

**Run All Tests:**
```bash
cd src
python -m pytest tests/ -v
```

**Run Golden Cases Tests:**
```bash
cd src
python -m pytest tests/golden_cases_test.py -v
```

**Example Output:**
```bash
=========================================== test session starts ============================================
platform darwin -- Python 3.12.4, pytest-7.4.4, pluggy-1.0.0
collected 11 items                                                                                         

src/tests/golden_cases_test.py::test_golden_case_regression[My domain was suspended and I didn't get any notice. This is unfair!-escalate_to_abuse_team-suspension|abuse|policy] PASSED [  9%]
src/tests/golden_cases_test.py::test_golden_case_regression[I need to update my WHOIS information for my domain-none-whois|update] PASSED [ 18%]
src/tests/golden_cases_test.py::test_golden_case_regression[I want a refund for my domain renewal. I was charged twice.-escalate_to_support-renewal|billing|refund] PASSED [ 27%]
src/tests/golden_cases_test.py::test_golden_case_regression[My domain is not resolving. DNS not working after nameserver change.-none-dns|nameserver|propagation] PASSED [ 36%]
src/tests/golden_cases_test.py::test_golden_case_regression[How do I transfer my domain to another registrar?-none-transfer|epp|auth] PASSED [ 45%]
src/tests/golden_cases_test.py::test_golden_case_regression[Hola, mi dominio no funciona. ¿Pueden ayudarme?-contact_customer-dns|nameserver|info] PASSED [ 54%]
src/tests/golden_cases_test.py::test_golden_case_regression[Complex technical issue with advanced DNS configuration not covered in docs-escalate_to_support-dns|technical|configuration] PASSED [ 63%]
src/tests/golden_cases_test.py::test_golden_case_regression[Someone is using my domain for spam and phishing attacks-escalate_to_abuse_team-abuse|spam|security] PASSED [ 72%]
src/tests/golden_cases_test.py::test_golden_case_response_times PASSED                               [ 81%]
src/tests/golden_cases_test.py::test_golden_case_consistency PASSED                                  [ 90%]
src/tests/golden_cases_test.py::test_golden_case_load PASSED                                         [100%]

====================================== 11 passed, 7 warnings in 5.03s ======================================
```

### Golden Cases Regression Tests

The `golden_cases_test.py` file contains critical regression tests that validate core functionality remains intact after code changes. These tests serve as a safety net for key user scenarios:

**Test Categories:**
- **Domain Suspension Cases**: Verify escalation to abuse team for policy violations
- **WHOIS Update Cases**: Ensure direct handling of domain management requests
- **Billing/Refund Cases**: Validate escalation to support team for payment issues
- **DNS Resolution Cases**: Test direct handling of technical DNS problems
- **Domain Transfer Cases**: Verify proper guidance for domain transfers
- **Non-English Queries**: Ensure appropriate handling of international customers
- **Technical Complexity**: Validate escalation for advanced technical issues
- **Abuse/Spam Cases**: Verify escalation to abuse team for security concerns

**Test Features:**
- **Response Time Validation**: Ensures API responses complete within reasonable time limits
- **Consistency Testing**: Verifies identical queries return consistent results
- **Load Testing**: Validates system stability under multiple concurrent requests
- **MCP Structure Validation**: Ensures responses follow the correct Model Context Protocol format
- **Reference Quality**: Validates that returned references contain relevant keywords

**Mocked Services**: These tests use mocked RAG pipeline and LLM services to ensure fast, reliable execution without external dependencies.

### Developer Self-Check

For additional developer self-check scripts (load testing, JSON compliance checks, debug endpoints), see [docs/SELF_CHECK.md](docs/SELF_CHECK.md).

## License & Credits

### Embedding Models
- **sentence-transformers/all-MiniLM-L6-v2**: [Apache 2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5**: [MIT](https://huggingface.co/BAAI/bge-small-en-v1.5)

### Vector Store
- **FAISS**: [MIT](https://github.com/facebookresearch/faiss/blob/main/LICENSE)
- **Qdrant** (future): [Apache 2.0](https://github.com/qdrant/qdrant/blob/master/LICENSE)

### LLM
- **OpenAI GPT-4o-mini**: [OpenAI Terms of Service](https://openai.com/policies/terms-of-use)

### Project License
MIT License © 2025 [AriesChen](https://github.com/sakanight)