import json
import logging
from typing import List, Dict, Literal
from pydantic import BaseModel
from models import LLMModel
from exceptions import (MCPOutputError, LLMRetryError, LLMTimeoutError, LLMRateLimitError, ValidationError)
from settings import settings

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
                    raise LLMRetryError(f"LLM generation failed", max_retries, str(e))
                # Continue to next attempt

        # Continue to next attempt

    # parse raw response to MCPResponse
    def parse_response(self, raw_response: str) -> MCPResponse:
        """
        Parse raw LLM output into MCPResponse.
        Strategy:
          1) extract JSON and validate
          2) if failed, trigger JSON-only fix retry (don't return original response)
        """
        try:
            data = self._extract_json(raw_response)
            return self._validate_and_build(data)
        except MCPOutputError as e:
            logger.warning("Primary JSON parse failed: %s. Trying JSON-only fix...", e)
            fixed = self._retry_json_only()
            data = self._extract_json(fixed)
            return self._validate_and_build(data)

    # extract JSON and validate
    def _extract_json(self, text: str) -> dict:
        # remove common code fences
        cleaned = text.replace('```json', '').replace('```JSON', '').replace('```', '').strip()
        # get first and last curly braces
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        if start == -1 or end <= 0 or end <= start:
            raise MCPOutputError("No JSON object found in LLM response")
        json_str = cleaned[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as je:
            raise MCPOutputError(f"Invalid JSON format: {je}") from je

    # validate and build MCPResponse
    def _validate_and_build(self, data: dict) -> MCPResponse:
        # required fields
        for field in ('answer', 'references', 'action_required'):
            if field not in data:
                raise ValidationError(f"Missing required field: {field}", field)

        # action_required validity
        action = data['action_required']
        if action not in settings.allowed_actions:
            raise ValidationError(f"Invalid action_required: {action}", "action_required", action)

        # references must be list[str]; deduplicate and limit
        refs = data['references']
        if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
            raise ValidationError("Field 'references' must be a list of strings", "references", str(type(refs)))
        
        # deduplicate, remove whitespace, and limit
        deduped = []
        seen = set()
        for r in refs:
            r_norm = r.strip()
            if r_norm and r_norm not in seen:
                deduped.append(r_norm)
                seen.add(r_norm)
            if len(deduped) >= settings.max_references:
                break

        # final build
        return MCPResponse(
            answer=str(data['answer']).strip(),
            action_required=action,
            references=deduped
        )

    # JSON-only fix retry, don't return original response
    def _retry_json_only(self) -> str:
        fix_prompt = (
            "Your previous output was not valid JSON. "
            "Now respond with ONLY a valid JSON object, no extra text/markdown/code fences. "
            "The JSON schema is:\n"
            "{\n"
            '  "answer": "string",\n'
            '  "references": ["string"],\n'
            '  "action_required": "one of: ' + ", ".join(settings.allowed_actions) + '"\n'
            "}\n"
        )
        # use the simplest text interface in models layer; if needed, can also use generate_with_messages
        try:
            return self.llm_model.generate(fix_prompt)
        except Exception as e:
            raise LLMRetryError(f"JSON fix retry failed", 1, str(e)) from e