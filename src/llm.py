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

IMPORTANT RULES:
1. Use only the provided context; do not fabricate information. If insufficient, explain and choose the appropriate action.
2. Be concise but comprehensive in your responses.
3. Always prioritize accuracy over completeness.

You must respond in the following JSON format:
{
    "answer": "Your answer based on the provided context",
    "references": ["List of document references that support your answer"],
    "action_required": "One of: none, escalate_to_support, escalate_to_abuse_team, contact_customer"
}

Use the provided context documents to answer accurately. If you cannot find relevant information, 
indicate that in your answer and set action_required to "escalate_to_support".

Action Guidelines:
- none: Sufficient information provided, no further action needed
- escalate_to_support: Complex technical issue or insufficient context
- escalate_to_abuse_team: Security, abuse, or policy violation concerns
- contact_customer: Need additional information from customer"""
    
    # Build system and user messages separately for better JSON compliance and provider flexibility
    def build_messages(self, ticket_text: str, context_docs: List[Dict]) -> tuple[str, str]:
        # Build context from documents with robust field handling and content truncation
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            # Robust field extraction with fallbacks
            title = doc.get('title', f'Document {i}')
            section = doc.get('section', 'Unknown Section')
            content = doc.get('content', '')
            ref = doc.get('ref', f"{title}-{section}")
            
            # Truncate content to avoid overly long prompts (600-800 chars per doc)
            max_content_length = 700
            if len(content) > max_content_length:
                content = content[:max_content_length] + "... [truncated]"
            
            context_parts.append(
                f"Document {i}:\n"
                f"Title: {title}\n"
                f"Section: {section}\n"
                f"Content: {content}\n"
                f"Reference: {ref}\n"
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
        import time
        
        system_message, user_message = self.build_messages(ticket_text, context_docs)
        logger.info(f"Generated messages with {len(context_docs)} context documents")
        
        # Generate response with retries
        for attempt in range(max_retries):
            start_time = time.time()
            try:
                logger.info(f"LLM generation attempt {attempt + 1}/{max_retries}")
                
                # Call LLM model from models layer with separate system and user messages
                raw_response = self.llm_model.generate_with_messages(system_message, user_message)
                request_time = time.time() - start_time
                logger.info(f"LLM response generated successfully in {request_time:.2f}s")
                
                # Parse and validate response
                parsed_response = self.parse_response(raw_response)
                total_time = time.time() - start_time
                logger.info(f"Response parsed successfully in {total_time:.2f}s: action_required={parsed_response.action_required}, references_count={len(parsed_response.references)}")
                
                return parsed_response
                
            except (ValidationError, MCPOutputError) as e:
                # JSON parsing/validation failed, try with enhanced instruction
                if attempt < max_retries - 1:
                    request_time = time.time() - start_time
                    logger.warning(f"JSON validation failed on attempt {attempt + 1} after {request_time:.2f}s: {e}. Retrying with enhanced instruction...")
                    system_message, user_message = self._enhance_messages_for_retry(system_message, user_message)
                else:
                    # Last attempt failed, raise the error
                    total_time = time.time() - start_time
                    raise LLMRetryError(f"JSON validation failed after {max_retries} attempts in {total_time:.2f}s", max_retries, str(e))
                    
            except LLMRateLimitError as e:
                # Rate limit: exponential backoff and retry
                if attempt < max_retries - 1:
                    request_time = time.time() - start_time
                    backoff_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                    logger.warning(f"Rate limit hit on attempt {attempt + 1} after {request_time:.2f}s. Backing off for {backoff_time}s...")
                    time.sleep(backoff_time)
                else:
                    total_time = time.time() - start_time
                    raise LLMRetryError(f"Rate limit exceeded after {max_retries} attempts in {total_time:.2f}s", max_retries, str(e))
                    
            except LLMTimeoutError as e:
                # Timeout: small backoff and retry once
                if attempt < max_retries - 1:
                    request_time = time.time() - start_time
                    logger.warning(f"Timeout on attempt {attempt + 1} after {request_time:.2f}s. Retrying once...")
                    time.sleep(1)  # Small backoff
                else:
                    total_time = time.time() - start_time
                    raise LLMRetryError(f"Timeout exceeded after {max_retries} attempts in {total_time:.2f}s", max_retries, str(e))
                    
            except Exception as e:
                # Other exceptions: log and retry if possible
                request_time = time.time() - start_time
                logger.warning(f"LLM generation attempt {attempt + 1} failed after {request_time:.2f}s: {e}")
                if attempt == max_retries - 1:
                    total_time = time.time() - start_time
                    raise LLMRetryError(f"LLM generation failed after {max_retries} attempts in {total_time:.2f}s", max_retries, str(e))
    
    # Enhance system message with stronger JSON formatting requirements
    def _enhance_messages_for_retry(self, system_message: str, user_message: str) -> tuple[str, str]:
        enhanced_system = system_message + """

IMPORTANT: You MUST respond with ONLY a valid JSON object. 
- No markdown formatting
- No code fences (```json)
- No additional text before or after the JSON
- No explanations outside the JSON structure

The response must be parseable by json.loads() directly."""
        
        # Enhance user message with explicit JSON requirement
        enhanced_user = user_message + """

CRITICAL: Respond with ONLY the JSON object, nothing else."""
        
        logger.info("Enhanced messages for JSON retry (context preserved)")
        return enhanced_system, enhanced_user

    # parse raw response to MCPResponse
    def parse_response(self, raw_response: str) -> MCPResponse:
        data = self._extract_json(raw_response)
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