import json
import logging
from typing import List, Dict, Literal
from pydantic import BaseModel
from models import LLMModel
from exceptions import LLMProviderError, MCPOutputError

logger = logging.getLogger(__name__)

# Model Context Protocol (MCP) response structure
class MCPResponse(BaseModel):
    answer: str
    references: List[str]
    action_required: Literal['none', 'escalate_to_support', 'escalate_to_abuse_team', 'contact_customer']

class LLMService:
    def __init__(self, model_name: str = None, provider: str = None):
        # Initialize LLM model from models layer
        self.llm_model = LLMModel(model_name, provider)
        
        # MCP prompt template
        self.system_prompt = """You are a knowledge assistant for customer support tickets. 
Your task is to analyze customer support queries and return structured, relevant, and helpful responses based on provided documentation.

You must respond in the following JSON format:
{
    "answer": "Your answer based on the provided context",
    "references": ["List of document references that support your answer"],
    "action_required": "One of: none, escalate_to_support, escalate_to_abuse_team, contact_customer"
}

Use the provided context documents to answer accurately. If you cannot find relevant information, 
indicate that in your answer and set action_required to "escalate_to_support"."""
    
    # Build system and user messages separately for better JSON compliance and provider flexibility
    def build_messages(self, ticket_text: str, context_docs: List[Dict]) -> tuple[str, str]:
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(
                f"Document {i}:\n"
                f"Title: {doc['title']}\n"
                f"Section: {doc['section']}\n"
                f"Content: {doc['content']}\n"
                f"Reference: {doc['ref']}\n"
            )
        
        context_text = "\n".join(context_parts) if context_parts else "No relevant documents found."
        
        # System message: rules, JSON schema, and restrictions
        system_message = self.system_prompt
        
        # User message: context + ticket
        user_message = f"""Context Documents:
{context_text}

Customer Ticket:
{ticket_text}

Please provide your response in the required JSON format:"""
        
        return system_message, user_message

    # Generate structured response using LLM with RAG context
    def generate_response(self, ticket_text: str, context_docs: List[Dict], 
                         max_retries: int = 3) -> MCPResponse:       
        system_message, user_message = self.build_messages(ticket_text, context_docs)
        logger.info(f"Generated messages with {len(context_docs)} context documents")
        
        # Generate response with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM generation attempt {attempt + 1}/{max_retries}")
                
                # Call LLM model from models layer with separate system and user messages
                raw_response = self.llm_model.generate_with_messages(system_message, user_message)
                logger.info("LLM response generated successfully")
                
                # Parse and validate response
                return self.parse_response(raw_response)
                
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise LLMProviderError(f"LLM generation failed after {max_retries} attempts: {e}")
                # Continue to next attempt