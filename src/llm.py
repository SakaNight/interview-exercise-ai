"""
LLM Service Module for Customer Support Ticket Resolution.

This module provides LLM service for processing customer support tickets
using Retrieval-Augmented Generation (RAG) with structured output following
the Model Context Protocol (MCP) specification.
"""

import json
from typing import Literal, Optional

from pydantic import BaseModel

from exceptions import (
    LLMRateLimitError,
    LLMRetryError,
    LLMTimeoutError,
    MCPOutputError,
    ValidationError,
)
from logging_config import get_logger
from models import LLMModel
from settings import settings

logger = get_logger(__name__)


class DocumentReference(BaseModel):
    """Structured reference model for document traceability in RAG responses."""

    doc_id: str
    title: str
    section: str
    url: Optional[str] = None

    def __str__(self) -> str:
        """String representation for backward compatibility."""
        return f"{self.title} - {self.section} (#{self.doc_id})"


class MCPResponse(BaseModel):
    """Model Context Protocol (MCP) compliant response structure."""

    answer: str
    references: list[DocumentReference]
    action_required: Literal[
        "none", "escalate_to_support", "escalate_to_abuse_team", "contact_customer"
    ]


class LLMService:
    """Main service class for LLM-powered customer support ticket resolution."""

    def __init__(
        self, model_name: Optional[str] = None, provider: Optional[str] = None
    ):
        """Initialize LLM service with specified model configuration.

        Args:
            model_name: Name of the LLM model to use. Defaults to settings.
            provider: LLM provider (e.g., 'openai', 'anthropic'). Defaults to settings.
        """
        self.llm_model = LLMModel(model_name, provider)
        self.system_prompt = self._build_mcp_system_prompt()

    def _build_mcp_system_prompt(self) -> str:
        """Build MCP-compliant system prompt template."""
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
  "references": [
    {
      "doc_id": "string (required) - Document ID from the provided context",
      "title": "string (required) - Document title from the provided context",
      "section": "string (required) - Document section from the provided context",
      "url": "string (optional) - URL if available, null otherwise"
    }
  ] (required) - Array of structured document references that support your answer (max 3 items),
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

    def build_messages(
        self, ticket_text: str, context_docs: list[dict]
    ) -> tuple[str, str]:
        """Build system and user messages for LLM interaction.

        Args:
            ticket_text: The customer support ticket content to process.
            context_docs: List of context documents from knowledge base.

        Returns:
            tuple[str, str]: (system_message, user_message) formatted for LLM.
        """
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            title = doc.get("title", f"Document {i}")
            section = doc.get("section", "Unknown Section")
            content = doc.get("content", "")
            ref = doc.get("ref", f"{title}-{section}")

            # Truncate content to avoid overly long prompts (600-800 chars per doc)
            max_content_length = 700
            if len(content) > max_content_length:
                content = content[:max_content_length] + "... [truncated]"

            doc_id = doc.get("id", f"doc_{i:03d}")

            context_parts.append(
                f"Document {i}:\n"
                f"ID: {doc_id}\n"
                f"Title: {title}\n"
                f"Section: {section}\n"
                f"Content: {content}\n"
                f"Reference: {ref}\n"
            )

        context_text = (
            "\n".join(context_parts)
            if context_parts
            else "No relevant documents found."
        )

        system_message = self.system_prompt

        user_message = f"""Context Documents:
{context_text}

Customer Ticket:
{ticket_text}

Please provide your response in the required JSON format:"""

        return system_message, user_message

    def generate_response(
        self, ticket_text: str, context_docs: list[dict], max_retries: int = 3
    ) -> MCPResponse:
        """Generate structured response using LLM with RAG context.

        Args:
            ticket_text: The customer support ticket content to process.
            context_docs: List of context documents from knowledge base.
            max_retries: Maximum number of retry attempts. Defaults to 3.

        Returns:
            MCPResponse: Structured response containing answer, references, and action.

        Raises:
            LLMRetryError: When all retry attempts are exhausted.
            LLMRateLimitError: When rate limit is exceeded after all retries.
            LLMTimeoutError: When request times out after all retries.
            ValidationError: When response validation fails.
            MCPOutputError: When JSON parsing fails.
        """
        import time

        system_message, user_message = self.build_messages(ticket_text, context_docs)
        logger.info(f"Generated messages with {len(context_docs)} context documents")

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                logger.info(f"LLM generation attempt {attempt + 1}/{max_retries}")

                raw_response = self.llm_model.generate_with_messages(
                    system_message, user_message
                )
                request_time = time.time() - start_time
                logger.info(
                    f"LLM response generated successfully in {request_time:.2f}s"
                )

                parsed_response = self.parse_response(raw_response)
                total_time = time.time() - start_time
                logger.info(
                    f"Response parsed successfully in {total_time:.2f}s: action_required={parsed_response.action_required}, references_count={len(parsed_response.references)}"
                )

                return parsed_response

            except (ValidationError, MCPOutputError) as e:
                if attempt < max_retries - 1:
                    request_time = time.time() - start_time
                    logger.warning(
                        f"JSON validation failed on attempt {attempt + 1} after {request_time:.2f}s: {e}. Retrying with enhanced instruction..."
                    )
                    system_message, user_message = self._enhance_messages_for_retry(
                        system_message, user_message, attempt
                    )
                else:
                    total_time = time.time() - start_time
                    raise LLMRetryError(
                        f"JSON validation failed after {max_retries} attempts in {total_time:.2f}s",
                        max_retries,
                        str(e),
                    )

            except LLMRateLimitError as e:
                if attempt < max_retries - 1:
                    request_time = time.time() - start_time
                    backoff_time = min(2**attempt, 30)
                    logger.warning(
                        f"Rate limit hit on attempt {attempt + 1} after {request_time:.2f}s. Backing off for {backoff_time}s..."
                    )
                    time.sleep(backoff_time)
                else:
                    total_time = time.time() - start_time
                    raise LLMRetryError(
                        f"Rate limit exceeded after {max_retries} attempts in {total_time:.2f}s",
                        max_retries,
                        str(e),
                    )

            except LLMTimeoutError as e:
                if attempt < max_retries - 1:
                    request_time = time.time() - start_time
                    logger.warning(
                        f"Timeout on attempt {attempt + 1} after {request_time:.2f}s. Retrying once..."
                    )
                    time.sleep(1)
                else:
                    total_time = time.time() - start_time
                    raise LLMRetryError(
                        f"Timeout exceeded after {max_retries} attempts in {total_time:.2f}s",
                        max_retries,
                        str(e),
                    )

            except Exception as e:
                request_time = time.time() - start_time
                logger.warning(
                    f"LLM generation attempt {attempt + 1} failed after {request_time:.2f}s: {e}"
                )
                if attempt == max_retries - 1:
                    total_time = time.time() - start_time
                    raise LLMRetryError(
                        f"LLM generation failed after {max_retries} attempts in {total_time:.2f}s",
                        max_retries,
                        str(e),
                    )

    def _enhance_messages_for_retry(
        self, system_message: str, user_message: str, attempt: int
    ) -> tuple[str, str]:
        """Enhance messages with stronger JSON formatting requirements for retry.

        Args:
            system_message: Original system message to enhance.
            user_message: Original user message to enhance.
            attempt: Current retry attempt number (0-based).

        Returns:
            tuple[str, str]: Enhanced messages with stricter JSON formatting.
        """
        if attempt == 1:
            enhanced_system = (
                system_message
                + """

=== CRITICAL JSON FORMATTING REQUIREMENTS ===
You MUST respond with ONLY a valid JSON object.
- No markdown formatting (```json)
- No code fences or backticks
- No additional text before or after the JSON
- No explanations outside the JSON structure
- The response must be parseable by json.loads() directly

FAILURE TO FOLLOW THESE REQUIREMENTS WILL RESULT IN SYSTEM ERROR."""
            )

            enhanced_user = (
                user_message
                + """

CRITICAL: Respond with ONLY the JSON object, nothing else. No explanations, no formatting."""
            )

        else:
            enhanced_system = (
                system_message
                + """

=== MAXIMUM STRICTNESS MODE ===
You have failed to provide valid JSON multiple times. This is your final attempt.

REQUIREMENTS:
1. Return ONLY the JSON object
2. No text before or after the JSON
3. No markdown, no code fences, no backticks
4. No explanations, no comments
5. Must be valid JSON that can be parsed by json.loads()

EXAMPLE OF CORRECT FORMAT:
{
  "answer": "Your answer here",
  "references": [
    {
      "doc_id": "doc_001",
      "title": "Domain Suspension Guidelines",
      "section": "Reasons for Suspension",
      "url": null
    }
  ],
  "action_required": "none"
}

ANY DEVIATION FROM THIS FORMAT WILL CAUSE SYSTEM FAILURE."""
            )

            enhanced_user = (
                user_message
                + """

FINAL ATTEMPT: Return ONLY the JSON object. Nothing else. No explanations."""
            )

        logger.info(
            f"Enhanced messages for JSON retry attempt {attempt + 1} (context preserved)"
        )
        return enhanced_system, enhanced_user

    def parse_response(self, raw_response: str) -> MCPResponse:
        """Parse raw LLM response into structured MCPResponse.

        Args:
            raw_response: Raw text response from the LLM.

        Returns:
            MCPResponse: Parsed and validated structured response.

        Raises:
            MCPOutputError: When JSON extraction or parsing fails.
            ValidationError: When response validation fails.
        """
        data = self._extract_json(raw_response)
        return self._validate_and_build(data)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON object from raw LLM response text.

        Args:
            text: Raw response text that may contain JSON.

        Returns:
            dict: Parsed JSON object.

        Raises:
            MCPOutputError: When no valid JSON object is found or parsing fails.
        """
        cleaned = (
            text.replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end <= 0 or end <= start:
            raise MCPOutputError("No JSON object found in LLM response")
        json_str = cleaned[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as je:
            raise MCPOutputError(f"Invalid JSON format: {je}") from je

    def _validate_and_build(self, data: dict) -> MCPResponse:
        """Validate JSON data and build MCPResponse object.

        Args:
            data: Parsed JSON data to validate and build from.

        Returns:
            MCPResponse: Validated and constructed response object.

        Raises:
            ValidationError: When validation fails at any stage.
        """
        self._validate_json_schema(data)

        answer = self._validate_answer_field(data["answer"])
        action_required = self._validate_action_field(data["action_required"])
        references = self._validate_references_field(data["references"])

        return MCPResponse(
            answer=answer, action_required=action_required, references=references
        )

    def _validate_json_schema(self, data: dict) -> None:
        """Validate JSON schema structure for MCP response.

        Args:
            data: JSON data to validate.

        Raises:
            ValidationError: When schema validation fails.
        """
        if not isinstance(data, dict):
            raise ValidationError(
                "Response must be a JSON object", "root", str(type(data))
            )

        required_fields = ["answer", "references", "action_required"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {missing_fields}",
                "schema",
                str(missing_fields),
            )

        allowed_fields = set(required_fields)
        extra_fields = [field for field in data.keys() if field not in allowed_fields]
        if extra_fields:
            raise ValidationError(
                f"Unexpected fields found: {extra_fields}", "schema", str(extra_fields)
            )

    def _validate_answer_field(self, answer: any) -> str:
        """Validate and clean the answer field.

        Args:
            answer: The answer field value to validate.

        Returns:
            str: Cleaned and validated answer text.

        Raises:
            ValidationError: When validation fails.
        """
        if not isinstance(answer, str):
            raise ValidationError(
                "Field 'answer' must be a string", "answer", str(type(answer))
            )

        answer_clean = answer.strip()
        if not answer_clean:
            raise ValidationError("Field 'answer' cannot be empty", "answer", "empty")

        if len(answer_clean) > 2000:  # Reasonable limit
            raise ValidationError(
                "Field 'answer' exceeds maximum length (2000 chars)",
                "answer",
                f"length: {len(answer_clean)}",
            )

        return answer_clean

    def _validate_action_field(self, action: any) -> str:
        """Validate the action_required field.

        Args:
            action: The action_required field value to validate.

        Returns:
            str: Validated action value.

        Raises:
            ValidationError: When validation fails.
        """
        if not isinstance(action, str):
            raise ValidationError(
                "Field 'action_required' must be a string",
                "action_required",
                str(type(action)),
            )

        action_clean = action.strip()
        if action_clean not in settings.allowed_actions:
            raise ValidationError(
                f"Invalid action_required: '{action_clean}'. Must be one of: {settings.allowed_actions}",
                "action_required",
                action_clean,
            )

        return action_clean

    def _validate_references_field(self, refs: any) -> list[DocumentReference]:
        """Validate and clean the references field.

        Args:
            refs: The references field value to validate.

        Returns:
            list[DocumentReference]: List of validated and cleaned references.

        Raises:
            ValidationError: When validation fails.
        """
        if not isinstance(refs, list):
            raise ValidationError(
                "Field 'references' must be a list", "references", str(type(refs))
            )

        if len(refs) == 0:
            raise ValidationError(
                "Field 'references' cannot be empty", "references", "empty list"
            )

        validated_refs = []
        for i, ref in enumerate(refs):
            if not isinstance(ref, dict):
                raise ValidationError(
                    f"Reference at index {i} must be an object",
                    "references",
                    f"index {i}: {type(ref)}",
                )

            required_fields = ["doc_id", "title", "section"]
            missing_fields = [field for field in required_fields if field not in ref]
            if missing_fields:
                raise ValidationError(
                    f"Reference at index {i} missing required fields: {missing_fields}",
                    "references",
                    f"index {i}: missing {missing_fields}",
                )

            doc_id = self._validate_reference_field(
                ref["doc_id"], i, "doc_id", max_length=50
            )
            title = self._validate_reference_field(
                ref["title"], i, "title", max_length=200
            )
            section = self._validate_reference_field(
                ref["section"], i, "section", max_length=200
            )

            url = None
            if "url" in ref:
                if ref["url"] is not None:
                    url = self._validate_reference_field(
                        ref["url"], i, "url", max_length=500, allow_empty=False
                    )

            validated_refs.append(
                DocumentReference(doc_id=doc_id, title=title, section=section, url=url)
            )

        deduped = []
        seen_doc_ids = set()
        for ref in validated_refs:
            if ref.doc_id not in seen_doc_ids:
                deduped.append(ref)
                seen_doc_ids.add(ref.doc_id)
            if len(deduped) >= settings.max_references:
                break

        return deduped

    def _validate_reference_field(
        self,
        value: any,
        index: int,
        field_name: str,
        max_length: int = 200,
        allow_empty: bool = True,
    ) -> str:
        """Validate individual reference field.

        Args:
            value: The field value to validate.
            index: Index of the reference in the references array.
            field_name: Name of the field being validated.
            max_length: Maximum allowed length. Defaults to 200.
            allow_empty: Whether empty values are allowed. Defaults to True.

        Returns:
            str: Cleaned and validated field value.

        Raises:
            ValidationError: When validation fails.
        """
        if not isinstance(value, str):
            raise ValidationError(
                f"Reference at index {index}, field '{field_name}' must be a string",
                "references",
                f"index {index}.{field_name}: {type(value)}",
            )

        value_clean = value.strip()
        if not allow_empty and not value_clean:
            raise ValidationError(
                f"Reference at index {index}, field '{field_name}' cannot be empty",
                "references",
                f"index {index}.{field_name}: empty",
            )

        if len(value_clean) > max_length:
            raise ValidationError(
                f"Reference at index {index}, field '{field_name}' exceeds maximum length ({max_length} chars)",
                "references",
                f"index {index}.{field_name}: length {len(value_clean)}",
            )

        return value_clean
