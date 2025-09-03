# API Endpoints Documentation

## Overview

This document provides detailed information about all API endpoints, including complete request/response examples and authentication requirements.

## Structure

The API follows a RESTful design with the following structure:

```
Base URL: http://localhost:8000

Public Endpoints:
├── GET  /                    # Health check
├── POST /resolve-ticket      # Main ticket resolution
└── GET  /metrics            # Performance metrics

Internal Endpoints:
├── POST /resolve-ticket/debug    # Debug ticket resolution (no auth)
├── GET  /documents              # List documents (requires X-Debug-Token)
├── POST /reindex               # Rebuild index (requires X-Reindex-Token)
└── GET  /metrics/endpoints     # List all monitored endpoints (no auth)
```

### Authentication
- **Public endpoints**: No authentication required
- **Internal endpoints**: Require specific token headers:
  - `X-Debug-Token`: For document listing and debug operations
  - `X-Reindex-Token`: For index rebuilding operations

## API Endpoints Table

| Endpoint | Method | Purpose | Authentication | Status |
|----------|--------|---------|----------------|--------|
| `/` | GET | Health check and system status | None | Public |
| `/resolve-ticket` | POST | Resolve customer support tickets | None | Public |
| `/resolve-ticket/debug` | POST | Debug endpoint with detailed logs | None | Internal |
| `/documents` | GET | List all available documents | X-Debug-Token | Internal |
| `/reindex` | POST | Rebuild FAISS vector index | X-Reindex-Token | Internal |
| `/metrics` | GET | Performance and system metrics | None | Internal |
| `/metrics/endpoints` | GET | List all monitored endpoints | None | Internal |

## Detailed Endpoint Documentation

### Health Check

**GET** `/`

Returns system health status and performance metrics.

**Request:**
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

### Resolve Support Ticket

**POST** `/resolve-ticket`

Main endpoint for resolving customer support tickets using RAG + LLM.

**Request:**
```bash
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I cannot access my domain, it says invalid password"
  }'
```

**Response (MCP Format):**
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

### Debug Endpoint (Internal Use)

**POST** `/resolve-ticket/debug`

Debug version with detailed logging and intermediate results.

**Request:**
```bash
curl -X POST http://localhost:8000/resolve-ticket/debug \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I cannot access my domain, it says invalid password"
  }'
```

**Response:**
```json
{
  "answer": "Based on the documentation, you need to reset your password. Please check your email for reset instructions.",
  "references": [
    {
      "doc_id": "doc_001",
      "title": "Domain Access Guide",
      "section": "Password Management",
      "url": null
    }
  ],
  "action_required": "contact_customer",
  "debug_info": {
    "rag_results": {
      "query": "I cannot access my domain, it says invalid password",
      "retrieved_docs": 3,
      "similarity_scores": [0.85, 0.72, 0.68]
    },
    "llm_processing": {
      "model_used": "gpt-3.5-turbo",
      "tokens_used": 156,
      "processing_time": 0.45
    },
    "validation": {
      "mcp_compliant": true,
      "schema_valid": true
    }
  }
}
```

### List Documents (Internal Use)

**GET** `/documents`

List all available documents in the knowledge base.

**Request:**
```bash
curl -H "X-Debug-Token: dev-debug-2025" http://localhost:8000/documents
```

**Response:**
```json
{
  "total_documents": 5,
  "documents": [
    {
      "doc_id": "doc_001",
      "title": "Domain Access Guide",
      "sections": ["Password Management", "Account Setup", "Troubleshooting"],
      "last_updated": "2024-01-15T10:30:00Z"
    },
    {
      "doc_id": "doc_002",
      "title": "Password Reset",
      "sections": ["Reset Process", "Security Questions", "Email Verification"],
      "last_updated": "2024-01-10T14:20:00Z"
    }
  ]
}
```

### Rebuild Index (Internal Use)

**POST** `/reindex`

Rebuild the FAISS vector index from documents.

**Request:**
```bash
curl -X POST -H "X-Reindex-Token: dev-reindex-2025" http://localhost:8000/reindex
```

**Response:**
```json
{
  "status": "success",
  "message": "FAISS index rebuilt successfully",
  "details": {
    "documents_processed": 5,
    "vectors_created": 15,
    "index_size_mb": 2.3,
    "processing_time": 1.2
  }
}
```

### Performance Metrics

**GET** `/metrics`

Get system performance and usage metrics.

**Request:**
```bash
curl http://localhost:8000/metrics
```

### All Endpoints List

**GET** `/metrics/endpoints`

Get list of all monitored endpoints.

**Request:**
```bash
curl http://localhost:8000/metrics/endpoints
```

**Response:**
```json
{
  "system_metrics": {
    "uptime_seconds": 3600,
    "total_requests": 150,
    "successful_requests": 147,
    "failed_requests": 3,
    "avg_response_time": 0.0456
  },
  "rag_metrics": {
    "total_queries": 150,
    "avg_retrieval_time": 0.012,
    "avg_similarity_score": 0.78
  },
  "llm_metrics": {
    "total_tokens_used": 15600,
    "avg_tokens_per_request": 104,
    "avg_processing_time": 0.45
  }
}
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Error type",
  "message": "Human-readable error message",
  "details": {
    "field": "Additional error details"
  }
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `422`: Validation Error (schema validation failed)
- `500`: Internal Server Error
- `503`: Service Unavailable (LLM service down)

## Rate Limiting

Currently no rate limiting is implemented. For production deployment, consider adding rate limiting based on your requirements.

## Authentication

All endpoints are currently public (no authentication required). For production deployment, consider implementing:
- API key authentication
- JWT tokens
- OAuth 2.0
- Basic authentication
