import json
import logging
from typing import List, Dict, Literal
from pydantic import BaseModel
from models import LLMModel
from exceptions import MCPOutputError, LLMRetryError, LLMTimeoutError, LLMRateLimitError, ValidationError
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
        
        # MCP-compliant prompt template with clear four-section structure
        self.system_prompt = self._build_mcp_system_prompt()
    
    def _build_mcp_system_prompt(self) -> str:
        """Build MCP-compliant system prompt with clear four-section structure"""
        return """=== ROLE ===
You are a specialized knowledge assistant for customer support ticket resolution. Your expertise is in analyzing customer queries and providing structured, accurate responses based on provided documentation.

=== CONTEXT ===
You will receive:
1. Customer support ticket text
2. Relevant documentation context retrieved from the knowledge base
3. Clear task instructions and output schema requirements

=== TASK ===
Your task is to:
1. Analyze the customer support query carefully
2. Use ONLY the provided context documents to formulate your response
3. Provide accurate, helpful answers based on the documentation
4. Determine the appropriate action required based on the query complexity and available information
5. Return your response in the exact JSON schema format specified below

IMPORTANT CONSTRAINTS:
- Use ONLY information from the provided context documents
- Do not fabricate or assume information not present in the context
- If insufficient information is available, clearly state this and choose appropriate escalation
- Prioritize accuracy over completeness
- Be concise but comprehensive in your responses

=== SCHEMA ===
You MUST respond with a valid JSON object matching this exact schema:

{
  "answer": "string (required) - Your detailed answer based on the provided context",
  "references": ["string"] (required) - Array of document references that support your answer (max 3 items),
  "action_required": "string (required) - One of: 'none', 'escalate_to_support', 'escalate_to_abuse_team', 'contact_customer'"
}

ACTION GUIDELINES:
- "none": Sufficient information provided, customer can resolve issue independently
- "escalate_to_support": Complex technical issue or insufficient context in documentation
- "escalate_to_abuse_team": Security concerns, policy violations, or abuse-related issues
- "contact_customer": Need additional information from customer to proceed

RESPONSE REQUIREMENTS:
- Return ONLY the JSON object, no additional text
- No markdown formatting or code fences
- All fields are required and must be valid according to the schema
- References must be actual document titles/sections from the provided context"""

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
                    system_message, user_message = self._enhance_messages_for_retry(system_message, user_message, attempt)
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
    
    # Enhance system message with stronger JSON formatting requirements for retry
    def _enhance_messages_for_retry(self, system_message: str, user_message: str, attempt: int) -> tuple[str, str]:
        """Enhance messages for retry with progressively stronger JSON requirements"""
        
        # Progressive enhancement based on attempt number
        if attempt == 1:
            # First retry: Basic JSON formatting emphasis
            enhanced_system = system_message + """

=== CRITICAL JSON FORMATTING REQUIREMENTS ===
You MUST respond with ONLY a valid JSON object. 
- No markdown formatting (```json)
- No code fences or backticks
- No additional text before or after the JSON
- No explanations outside the JSON structure
- The response must be parseable by json.loads() directly

FAILURE TO FOLLOW THESE REQUIREMENTS WILL RESULT IN SYSTEM ERROR."""
            
            enhanced_user = user_message + """

CRITICAL: Respond with ONLY the JSON object, nothing else. No explanations, no formatting."""
            
        else:
            # Second+ retry: Maximum strictness
            enhanced_system = system_message + """

=== MAXIMUM STRICTNESS MODE ===
You have failed to provide valid JSON multiple times. This is your final attempt.

REQUIREMENTS:
1. Return ONLY the JSON object
2. No text before or after the JSON
3. No markdown, no code fences, no backticks
4. No explanations, no comments
5. Must be valid JSON that can be parsed by json.loads()

EXAMPLE OF CORRECT FORMAT:
{"answer": "Your answer here", "references": ["ref1", "ref2"], "action_required": "none"}

ANY DEVIATION FROM THIS FORMAT WILL CAUSE SYSTEM FAILURE."""
            
            enhanced_user = user_message + """

FINAL ATTEMPT: Return ONLY the JSON object. Nothing else. No explanations."""
        
        logger.info(f"Enhanced messages for JSON retry attempt {attempt + 1} (context preserved)")
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

    # validate and build MCPResponse with strict JSON Schema validation
    def _validate_and_build(self, data: dict) -> MCPResponse:
        # Strict JSON Schema validation
        self._validate_json_schema(data)
        
        # Extract and validate fields
        answer = self._validate_answer_field(data['answer'])
        action_required = self._validate_action_field(data['action_required'])
        references = self._validate_references_field(data['references'])

        # Build final response
        return MCPResponse(
            answer=answer,
            action_required=action_required,
            references=references
        )
    
    def _validate_json_schema(self, data: dict) -> None:
        """Strict JSON Schema validation for MCP response"""
        if not isinstance(data, dict):
            raise ValidationError("Response must be a JSON object", "root", str(type(data)))
        
        # Check required fields
        required_fields = ['answer', 'references', 'action_required']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}", "schema", str(missing_fields))
        
        # Check for extra fields (strict schema)
        allowed_fields = set(required_fields)
        extra_fields = [field for field in data.keys() if field not in allowed_fields]
        if extra_fields:
            raise ValidationError(f"Unexpected fields found: {extra_fields}", "schema", str(extra_fields))
    
    def _validate_answer_field(self, answer) -> str:
        """Validate and clean answer field"""
        if not isinstance(answer, str):
            raise ValidationError("Field 'answer' must be a string", "answer", str(type(answer)))
        
        answer_clean = answer.strip()
        if not answer_clean:
            raise ValidationError("Field 'answer' cannot be empty", "answer", "empty")
        
        if len(answer_clean) > 2000:  # Reasonable limit
            raise ValidationError("Field 'answer' exceeds maximum length (2000 chars)", "answer", f"length: {len(answer_clean)}")
        
        return answer_clean
    
    def _validate_action_field(self, action) -> str:
        """Validate action_required field"""
        if not isinstance(action, str):
            raise ValidationError("Field 'action_required' must be a string", "action_required", str(type(action)))
        
        action_clean = action.strip()
        if action_clean not in settings.allowed_actions:
            raise ValidationError(f"Invalid action_required: '{action_clean}'. Must be one of: {settings.allowed_actions}", 
                                "action_required", action_clean)
        
        return action_clean
    
    def _validate_references_field(self, refs) -> List[str]:
        """Validate and clean references field"""
        if not isinstance(refs, list):
            raise ValidationError("Field 'references' must be a list", "references", str(type(refs)))
        
        if len(refs) == 0:
            raise ValidationError("Field 'references' cannot be empty", "references", "empty list")
        
        # Validate each reference
        validated_refs = []
        for i, ref in enumerate(refs):
            if not isinstance(ref, str):
                raise ValidationError(f"Reference at index {i} must be a string", "references", f"index {i}: {type(ref)}")
            
            ref_clean = ref.strip()
            if not ref_clean:
                raise ValidationError(f"Reference at index {i} cannot be empty", "references", f"index {i}: empty")
            
            if len(ref_clean) > 200:  # Reasonable limit per reference
                raise ValidationError(f"Reference at index {i} exceeds maximum length (200 chars)", 
                                    "references", f"index {i}: length {len(ref_clean)}")
            
            validated_refs.append(ref_clean)
        
        # Deduplicate and limit
        deduped = []
        seen = set()
        for ref in validated_refs:
            if ref not in seen:
                deduped.append(ref)
                seen.add(ref)
            if len(deduped) >= settings.max_references:
                break
        
        return deduped