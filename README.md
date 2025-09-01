# AI Knowledge Assistant for Support Team

## Introduction
This project builds an AI assistant for supporting customer tickets responding based on the relevant supporting documents. It uses a **Retrieval-Augmented Generation (RAG)** pipeline powered by an **LLM** and follow the **Model Context Protocol (MCP)** to produce structured output.

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

## Structure
```bash
├── src
│   ├── data
│   │   └── docs.json
│   ├── scripts
│   │   └── prepare_index.py
│   ├── tests
│   │   ├── __init__.py
│   │   ├── api_tests.py
│   │   ├── conftest.py
│   │   ├── llm_tests.py
│   │   └── rag_tests.py
│   ├── exceptions.py
│   ├── llm.py
│   ├── main.py
│   ├── models.py
│   ├── rag.py
│   └── settings.py
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
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

**Response:**
```json
{
  "success": true,
  "data": {
    "answer": "Based on the documentation, you need to reset your password. Please check your email for reset instructions.",
    "references": ["Domain Access Guide", "Password Reset"],
    "action_required": "contact_customer"
  },
  "processing_time": 1.234,
  "documents_retrieved": 2
}
```

## 🐳 Docker Deployment
```bash
# Start development environment with hot reload
docker-compose --profile dev up --build

## 🧪 Testing

### Test Structure

The project includes comprehensive tests organized into different modules:

- **API Tests** (`tests/api_tests.py`): Endpoint testing with mocked services
- **LLM Tests** (`tests/llm_tests.py`): LLM service functionality testing
- **RAG Tests** (`tests/rag_tests.py`): RAG pipeline testing
- **Integration Tests**: End-to-end functionality testing

**Run All Tests:**
```bash
cd src
python -m pytest tests/ -v
```

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
MIT License © 2025 [AriesChen](https://github.com/arieschan)