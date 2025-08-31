from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from typing import Optional, Dict, List
import logging
import time
import statistics
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from rag import RAGPipeline
from llm import LLMService, MCPResponse
from exceptions import (ValidationError, LLMRetryError, MCPOutputError)
from settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request/Response Models
class TicketRequest(BaseModel):
    ticket_text: str = Field(..., description="support ticket", min_length=1, max_length=5000)
    
class TicketResponse(BaseModel):
    success: bool = True
    data: MCPResponse
    processing_time: float
    documents_retrieved: int
    
class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_type: str
    processing_time: Optional[float] = None

# Document management models
class DocumentInfo(BaseModel):
    id: int
    title: str
    section: str
    tags: Optional[List[str]] = None
    ref: str

class ReindexRequest(BaseModel):
    token: str = Field(..., description="Security token for reindexing")

class ReindexResponse(BaseModel):
    success: bool = True
    documents_processed: int
    processing_time: float
    message: str

# Global variables for services
rag_pipeline: Optional[RAGPipeline] = None
llm_service: Optional[RAGPipeline] = None

# Performance monitoring data structures
class PerformanceMetrics:
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.response_sizes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.total_requests: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()
    
    # Add request duration for an endpoint
    def add_request_time(self, endpoint: str, duration: float):
        self.request_times[endpoint].append(duration)
        self.total_requests[endpoint] += 1
    
    # Add response size for an endpoint
    def add_response_size(self, endpoint: str, size: int):
        self.response_sizes[endpoint].append(size)
    
    # Increment error count for an endpoint
    def add_error(self, endpoint: str):
        self.error_counts[endpoint] += 1
    
    # Get performance statistics for an endpoint
    def get_stats(self, endpoint: str) -> Dict:
        times = list(self.request_times[endpoint])
        sizes = list(self.response_sizes[endpoint])
        
        if not times:
            return {
                "total_requests": self.total_requests[endpoint],
                "error_count": self.error_counts[endpoint],
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
                "p95_response_time": 0,
                "avg_response_size": 0
            }
        
        return {
            "total_requests": self.total_requests[endpoint],
            "error_count": self.error_counts[endpoint],
            "avg_response_time": statistics.mean(times),
            "min_response_time": min(times),
            "max_response_time": max(times),
            "p95_response_time": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            "avg_response_size": statistics.mean(sizes) if sizes else 0
        }
    
    # Get overall performance statistics
    def get_overall_stats(self) -> Dict:
        all_times = []
        all_sizes = []
        total_requests = sum(self.total_requests.values())
        total_errors = sum(self.error_counts.values())
        
        for times in self.request_times.values():
            all_times.extend(times)
        for sizes in self.response_sizes.values():
            all_sizes.extend(sizes)
        
        if not all_times:
            return {
                "uptime_seconds": time.time() - self.start_time,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": 0,
                "avg_response_time": 0,
                "p95_response_time": 0
            }
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "avg_response_time": statistics.mean(all_times),
            "p95_response_time": statistics.quantiles(all_times, n=20)[18] if len(all_times) >= 20 else max(all_times)
        }

# Initialize performance metrics
performance_metrics = PerformanceMetrics()

# Simple token validation for development mode
async def verify_reindex_token(token: str = Header(..., alias="X-Reindex-Token")):
    """Verify reindex token (simple development mode protection)"""
    # In production, this should be a proper JWT or API key validation
    expected_token = "dev-reindex-2025"
    if token != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid reindex token"
        )
    return token

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Ticket Resolution API...")
    
    # Initialize RAG pipeline
    global rag_pipeline
    try:
        rag_pipeline = RAGPipeline()
        if not rag_pipeline.setup_pipeline():
            raise Exception("Failed to setup RAG pipeline")
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise
    
    # Initialize LLM service
    global llm_service
    try:
        llm_service = LLMService()
        logger.info("LLM service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM service: {e}")
        raise
    
    logger.info("API startup completed successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down Ticket Resolution API...")
    # Cleanup if needed
    logger.info("API shutdown completed")

# Create FastAPI app
app = FastAPI(
    title="Ticket Resolution API",
    description="API for resolving customer support tickets",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance monitoring middleware
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Get endpoint name
    endpoint = f"{request.method} {request.url.path}"
    
    # Process request
    response = await call_next(request)
    
    # Calculate metrics
    duration = time.time() - start_time
    response_size = len(str(response.body)) if hasattr(response, 'body') else 0
    
    # Record metrics
    performance_metrics.add_request_time(endpoint, duration)
    performance_metrics.add_response_size(endpoint, response_size)
    
    # Record errors
    if response.status_code >= 400:
        performance_metrics.add_error(endpoint)
    
    # Add performance headers
    response.headers["X-Response-Time"] = f"{duration:.4f}s"
    response.headers["X-Request-ID"] = str(int(start_time * 1000000))
    
    return response

# Global exception handlers
@app.exception_handler(PydanticValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "error_type": "VALIDATION_ERROR",
            "detail": str(exc)
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_type": "HTTP_ERROR",
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_type": "INTERNAL_ERROR",
            "detail": "An unexpected error occurred"
        }
    )

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Ticket Resolution API",
        "version": "1.0.0",
        "status": "healthy",
        "performance": {
            "uptime_seconds": round(time.time() - performance_metrics.start_time, 2),
            "total_requests": sum(performance_metrics.total_requests.values()),
            "avg_response_time": round(
                statistics.mean([t for times in performance_metrics.request_times.values() for t in times]) 
                if any(performance_metrics.request_times.values()) else 0, 4
            )
        }
    }

# Test error handling endpoint
@app.get("/test-error")
async def test_error():
    raise HTTPException(
        status_code=400,
        detail="This is a test error to verify error handling"
    )

# Get overall performance metrics
@app.get("/metrics")
async def get_metrics():
    return {
        "success": True,
        "data": performance_metrics.get_overall_stats(),
        "timestamp": time.time()
    }

# Get performance metrics for a specific endpoint
@app.get("/metrics/{endpoint}")
async def get_endpoint_metrics(endpoint: str):
    # Convert URL path to endpoint format
    endpoint_key = f"GET {endpoint}" if not endpoint.startswith(("GET ", "POST ", "PUT ", "DELETE ")) else endpoint
    
    # Get all available endpoints
    available_endpoints = list(performance_metrics.total_requests.keys())
    
    if endpoint_key not in available_endpoints:
        # Try to find a matching endpoint
        matching_endpoints = [ep for ep in available_endpoints if endpoint in ep]
        if matching_endpoints:
            endpoint_key = matching_endpoints[0]
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Endpoint not found. Available endpoints: {available_endpoints}"
            )
    
    return {
        "success": True,
        "data": {
            "endpoint": endpoint_key,
            "metrics": performance_metrics.get_stats(endpoint_key)
        },
        "timestamp": time.time()
    }

# Get list of all monitored endpoints
@app.get("/metrics/endpoints")
async def get_all_endpoints():
    endpoints = list(performance_metrics.total_requests.keys())
    return {
        "success": True,
        "data": {
            "endpoints": endpoints,
            "total_endpoints": len(endpoints)
        },
        "timestamp": time.time()
    }

# List all documents in the index with metadata
@app.get("/documents")
async def get_documents():
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not available"
            )
        
        # Get documents from RAG pipeline
        documents = rag_pipeline.documents
        
        # Convert to DocumentInfo format
        doc_list = []
        for i, doc in enumerate(documents):
            doc_info = DocumentInfo(
                id=i + 1,
                title=doc.get('title', f'Document {i + 1}'),
                section=doc.get('section', 'Unknown Section'),
                tags=doc.get('tags', []),
                ref=doc.get('ref', f"{doc.get('title', '')}-{doc.get('section', '')}")
            )
            doc_list.append(doc_info)
        
        return {
            "success": True,
            "data": {
                "documents": doc_list,
                "total_count": len(doc_list)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents"
        )

# Rebuild the search index (development mode only)
@app.post("/reindex", response_model=ReindexResponse)
async def rebuild_index(token: str = Depends(verify_reindex_token)):
    start_time = time.time()
    
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not available"
            )
        
        logger.info("Starting index rebuild...")
        
        # Rebuild the index
        success = rag_pipeline.setup_pipeline()
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to rebuild index"
            )
        
        # Get document count
        documents = rag_pipeline.documents
        doc_count = len(documents)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Index rebuild completed successfully in {processing_time:.2f}s with {doc_count} documents")
        
        return ReindexResponse(
            success=True,
            documents_processed=doc_count,
            processing_time=processing_time,
            message=f"Index rebuilt successfully with {doc_count} documents"
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild index: {str(e)}"
        )

# Resolve ticket endpoint
@app.post("/resolve-ticket", response_model=TicketResponse)
async def resolve_ticket_endpoint(ticket: TicketRequest):
    start_time = time.time()
    
    try:
        # Validate services are available
        if not rag_pipeline or not llm_service:
            raise HTTPException(
                status_code=503, 
                detail="Service not available. Please try again later."
            )
        
        logger.info(f"Processing ticket: {ticket.ticket_text[:100]}...")
        
        # Search for relevant documents
        logger.info("Searching for relevant documents...")
        relevant_docs = rag_pipeline.search(ticket.ticket_text, k=settings.top_k)
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
            context_docs = []
        else:
            # Extract document data from search results
            context_docs = [doc for _, _, doc in relevant_docs]
            logger.info(f"Found {len(context_docs)} relevant documents")
        
        # Generate structured response
        logger.info("Generating LLM response...")
        response = llm_service.generate_response(ticket.ticket_text, context_docs)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Ticket resolution completed successfully in {processing_time:.2f}s")
        
        return TicketResponse(
            success=True,
            data=response,
            processing_time=processing_time,
            documents_retrieved=len(context_docs)
        )
        
    except (ValidationError, MCPOutputError) as e:
        # JSON validation or parsing errors
        processing_time = time.time() - start_time
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Response validation failed: {str(e)}"
        )
        
    except LLMRetryError as e:
        # LLM retry errors
        processing_time = time.time() - start_time
        logger.error(f"LLM retry error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service temporarily unavailable: {str(e)}"
        )
        
    except Exception as e:
        # Unexpected errors
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=True,
        log_level="info"
    )