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

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Local Development

1. **Clone and setup environment:**
```bash
git clone <your-repo-url>
cd interview-exercise-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment configuration:**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_actual_api_key_here
```

4. **Prepare knowledge base:**
```bash
cd src/scripts
python prepare_index.py
cd ../..
```

5. **Run the application:**
```bash
cd src
python main.py
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

#### Performance Metrics
**GET** `/metrics`
```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
  "success": true,
  "data": {
    "uptime_seconds": 1234.56,
    "total_requests": 42,
    "total_errors": 2,
    "error_rate": 0.0476,
    "avg_response_time": 0.0456,
    "p95_response_time": 0.123
  },
  "timestamp": 1703123456.789
}
```

### Docker
```bash
docker-compose up --build
```

## Test

### Run Tests
```bash
cd src
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/llm_tests.py -v
python -m pytest tests/rag_tests.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
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