# AI Knowledge Assistant for Support Team

> **MCP-compliant RAG assistant for customer tickets → structured JSON with citations & actions (FastAPI + FAISS + GPT-4o-mini).**

## What This Does

**Input**: Customer support ticket text
**Output**: Structured JSON response with answer, document references, and action required

```bash
# One-Click Setup
git clone <repo> && cd interview-exercise-ai
echo "OPENAI_API_KEY=your_key_here" > .env
docker-compose --profile dev up --build

# Test it
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "I forgot my password"}'
```

**Expected Response:**
```json
{
  "answer": "Please check your email for password reset instructions.",
  "references": [{"doc_id": "doc_001", "title": "Password Reset Guide", "section": "Email Instructions", "url": null}],
  "action_required": "contact_customer"
}
```

## Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key

### Environment Variables
Create `.env` file with required variables:

```bash
# Required
OPENAI_API_KEY=your_actual_api_key_here

# Optional (defaults provided)
DEBUG_TOKEN=dev-debug-2025
REINDEX_TOKEN=dev-reindex-2025
```

### Local Development
```bash
git clone <repo>
pip install -r requirements.txt
# If you don't have .env yet, create it from the template above
python src/main.py
```

### API Server
```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Quick Test
Test the API with a simple support ticket:

```bash
# Flat MCP JSON
curl -sX POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "I forgot my password"}' | jq

# Debug envelope (requires DEBUG token)
curl -sX POST http://localhost:8000/resolve-ticket/debug \
  -H "Content-Type: application/json" \
  -H "X-Debug-Token: ${DEBUG_TOKEN:-dev-debug-2025}" \
  -d '{"ticket_text":"How to transfer my domain?"}' | jq
```

**Expected Response:**
```json
{
  "answer": "Please check your email for password reset instructions.",
  "references": [
    {
      "doc_id": "doc_001",
      "title": "Password Reset Guide",
      "section": "Email Instructions",
      "url": null
    }
  ],
  "action_required": "contact_customer"
}
```

## Project at a Glance

- `src/main.py` – FastAPI entry point and routing
- `src/rag.py` – Document loading, vectorization, FAISS retrieval
- `src/llm.py` – MCP prompt, JSON validation and retry logic
- `src/settings.py` – Configuration and validation (Pydantic)
- `src/tests/` – Offline unit tests and golden cases
- `docs/SELF_CHECK.md` – Self-check scripts and validation commands

## Tech Stack and Design Explanation
- **Languages & API**: Python + FastAPI (lightweight, high-performance, OpenAPI integrated)
- **Embedding Models**: sentence-transformers/all-MiniLM-L6-v2 (fast, widely used, 384 dimensions)
- **Vector Store**: FAISS (lightweight, file persistence)
- **LLMs**: OpenAI gpt-4o-mini (temperature 0 for deterministic JSON)
- **Configuration Management**: Pydantic Settings for models, index, API keys management

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

<details>
<summary>Detailed Project Structure</summary>

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
│   │   ├── endpoint_test.py     # Endpoint comparison and demonstration script
│   │   ├── golden_cases_test.py # Golden cases regression tests for core functionality
│   │   ├── llm_tests.py         # LLM service functionality tests
│   │   └── rag_tests.py         # RAG pipeline integration tests
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── llm.py                   # LLM service with MCP-compliant prompt design
│   ├── logging_config.py        # Logging configuration with desensitization
│   ├── main.py                  # FastAPI application and endpoints
│   ├── models.py                # ML model abstractions (embedding, LLM)
│   ├── rag.py                   # RAG pipeline implementation
│   └── settings.py              # Configuration management with Pydantic
├── .github/                     # GitHub workflows and templates
│   └── workflows/               # CI/CD pipeline definitions
│       └── test.yml             # Automated testing workflow
├── docs/                        # Documentation
│   ├── ENDPOINTS.md             # Detailed API endpoint documentation
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
└── requirements.txt             # Python dependencies
```

</details>

### **Core Components**

| Component | File | Purpose |
|-----------|------|---------|
| **API Layer** | `main.py` | FastAPI application, endpoints, middleware, error handling |
| **RAG Pipeline** | `rag.py` | Document retrieval, embedding, FAISS index management |
| **LLM Service** | `llm.py` | MCP-compliant prompt design, JSON validation, retry logic |
| **Model Abstractions** | `models.py` | Embedding and LLM model wrappers for provider flexibility |
| **Configuration** | `settings.py` | Pydantic-based settings management with environment variables |
| **Logging** | `logging_config.py` | Logging configuration with desensitization for security |
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

## API Endpoints

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/` | GET | Health check and system status | None |
| `/resolve-ticket` | POST | Resolve customer support tickets | None |
| `/resolve-ticket/debug` | POST | Debug endpoint with detailed logs | None |
| `/documents` | GET | List all available documents | X-Debug-Token |
| `/reindex` | POST | Rebuild FAISS vector index | X-Reindex-Token |
| `/metrics` | GET | Performance and system metrics | None |
| `/metrics/endpoints` | GET | List all monitored endpoints | None |

<details>
<summary>Complete API Examples</summary>

### Health Check
```bash
curl http://localhost:8000/
```

### Resolve Ticket
```bash
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "I cannot access my domain"}'
```

### Debug Endpoint
```bash
curl -X POST http://localhost:8000/resolve-ticket/debug \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "Domain access issue"}'
```

### Internal Endpoints (with tokens)
```bash
# List documents
curl -H "X-Debug-Token: dev-debug-2025" http://localhost:8000/documents

# Rebuild index
curl -X POST -H "X-Reindex-Token: dev-reindex-2025" http://localhost:8000/reindex
```

</details>

## Testing

[![CI](https://github.com/SakaNight/interview-exercise-ai/actions/workflows/test.yml/badge.svg)](https://github.com/SakaNight/interview-exercise-ai/actions/workflows/test.yml)

**Testing Note:** All 24 unit/regression tests run offline with mocked services, ensuring stable CI/CD without external dependencies.

```bash
cd src
python -m pytest tests/ -v
```

<details>
<summary>Test Details</summary>

### Test Structure
- **API Tests**: 8 tests - Endpoint testing with mocked services
- **Golden Cases**: 6 tests - Regression tests for core functionality
- **LLM Tests**: 4 tests - LLM service functionality testing
- **RAG Tests**: 6 tests - RAG pipeline testing

### Run Specific Tests
```bash
# Golden cases regression tests
python -m pytest tests/golden_cases_test.py -v

# API endpoint tests
python -m pytest tests/api_tests.py -v

# All tests with coverage
python -m pytest tests/ -v --cov=.
```

</details>

## Docker Deployment

### Development
```bash
docker-compose --profile dev up --build
```

### Production
```bash
docker-compose --profile prod up --build
```

## Troubleshooting

- If model download/cache fails locally (permission errors), set `HF_HOME` to a writable directory or run via Docker.
- If `/reindex`/`/documents` return 401, ensure you pass `X-Reindex-Token` / `X-Debug-Token`.

## Configuration

### Allowed Actions
- `none`: Sufficient information provided
- `escalate_to_support`: Complex technical issue
- `escalate_to_abuse_team`: Security/abuse concerns
- `contact_customer`: Need additional information

### Key Settings
- **Max References**: 3 document references per response
- **Top K Results**: 3 most relevant documents retrieved
- **Temperature**: 0 for deterministic JSON output

## Documentation

- **[API Endpoints](docs/ENDPOINTS.md)** - Complete endpoint documentation with examples
- **[Self-Check Guide](docs/SELF_CHECK.md)** - Developer testing and validation scripts

## Future Work

- Add authentication and rate limiting
- Implement caching for improved performance
- Multilingual support (BAAI/bge-m3)
- Hybrid retrieval (FAISS + BM25)
- Monitoring with Prometheus/Grafana

## License & Credits

### Embedding Models
- **sentence-transformers/all-MiniLM-L6-v2**: Fast, efficient sentence embeddings

### Vector Store
- **FAISS**: Facebook AI Similarity Search for efficient vector operations

### LLM
- **OpenAI GPT-4o-mini**: Large language model for text generation

### Code Style
- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Fast Python linter
- **pytest**: Testing framework

### Project License
MIT License © 2025 [AriesChen](https://github.com/SakaNight)
