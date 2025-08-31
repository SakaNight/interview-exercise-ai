import logging
from rag import RAGPipeline
from llm import LLMService
from models import MCPResponse
import logging
from exceptions import MCPOutputError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def resolve_ticket(ticket_text: str) -> MCPResponse:
    try:
        # Step 1: Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag = RAGPipeline()
        
        # Step 2: Setup pipeline (load docs, create embeddings, build index)
        if not rag.setup_pipeline():
            raise Exception("Failed to setup RAG pipeline")
        
        # Step 3: Search for relevant documents
        logger.info(f"Searching for relevant documents for: {ticket_text}")
        relevant_docs = rag.search(ticket_text, k=3)
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
            # Create empty context for LLM
            context_docs = []
        else:
            # Extract document data from search results
            context_docs = [doc for _, _, doc in relevant_docs]
            logger.info(f"Found {len(context_docs)} relevant documents")
        
        # Step 4: Initialize LLM service
        logger.info("Initializing LLM service...")
        llm_service = LLMService()
        
        # Step 5: Generate structured response
        logger.info("Generating LLM response...")
        response = llm_service.generate_response(ticket_text, context_docs)
        
        logger.info("Ticket resolution completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Ticket resolution failed: {e}")
        # Return a fallback response
        return MCPResponse(
            answer=f"Sorry, I encountered an error while processing your request: {e}",
            references=[],
            action_required="escalate_to_support"
        )

def main():
    """Main function demonstrating the complete workflow"""
    
    # Example customer ticket
    ticket_text = "My domain was suspended and I didn't get any notice. How can I reactivate it?"
    
    print("=== Customer Support Ticket Resolution ===")
    print(f"Ticket: {ticket_text}")
    print("\nProcessing...")
    
    try:
        # Resolve the ticket
        response = resolve_ticket(ticket_text)
        
        # Display results
        print("\n=== Resolution Results ===")
        print(f"Answer: {response.answer}")
        print(f"References: {response.references}")
        print(f"Action Required: {response.action_required}")
        
        if response.references:
            print("\n=== Supporting Documents ===")
            for i, ref in enumerate(response.references, 1):
                print(f"{i}. {ref}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()