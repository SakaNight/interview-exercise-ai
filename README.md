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

## Structure
```bash
├── src
│   ├── data
│   │   └── docs.json
│   ├── scripts
│   │   └── prepare_index.py
│   ├── tests
│   │   └── __init__.py
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
### API 
### Docker

## Test
Run unit test and integration test

## License & Credits
### Embedding:
### Vector:
### LLM:
MIT@2025[AriesChen]