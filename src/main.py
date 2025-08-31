from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
import time
from contextlib import asynccontextmanager
from rag import RAGPipeline
from llm import LLMService, MCPResponse
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

# Global variables for services
rag_pipeline: Optional[RAGPipeline] = None
llm_service: Optional[LLMService] = None

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

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Ticket Resolution API",
        "version": "1.0.0",
        "status": "healthy"
    }

# Resolve ticket endpoint
@app.post("/resolve-ticket", response_model=TicketResponse)
async def resolve_ticket_endpoint(ticket: TicketRequest):
    start_time = time.time()
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=True,
        log_level="info"
    )