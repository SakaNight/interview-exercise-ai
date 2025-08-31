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
│   │   └── rag_tests.py
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
### Local
```bash
pip install sentence-transformers faiss-cpu numpy pydantic-settings openai
```
### API 
### Docker

## Test
Run unit test and integration test

## License & Credits
### Embedding:
### Vector:
### LLM:
MIT@2025[AriesChen]