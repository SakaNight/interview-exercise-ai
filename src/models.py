"""
Machine learning model abstractions and wrappers.

This module provides high-level abstractions for embedding models and LLM models,
encapsulating the complexity of different model providers and APIs. It includes
error handling, logging, and configuration management for seamless integration
with the RAG pipeline.
"""

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from exceptions import (
    EmbeddingModelError,
    LLMConfigError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    OpenAIAPIAuthenticationError,
    OpenAIAPIServerError,
)
from logging_config import get_logger
from settings import settings

logger = get_logger(__name__)


class EmbeddingModel:
    """Embedding model wrapper for text-to-vector conversion.

    This class provides a unified interface for sentence transformer models,
    handling model initialization, text encoding, and query embedding generation.
    It supports different embedding models through the sentence-transformers library.
    """

    def __init__(self, model_name: str = None):
        """Initialize embedding model.

        Args:
            model_name: Name of the embedding model. Defaults to settings.model_name.

        Raises:
            EmbeddingModelError: If model initialization fails.
        """
        self.model_name = model_name or settings.model_name
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Initialized embedding model: {self.model_name}")
        if not self.model:
            raise EmbeddingModelError(
                f"Failed to initialize embedding model: {self.model_name}"
            )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            np.ndarray: Embeddings array with shape (len(texts), embedding_dim).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        logger.info(
            f"Encoded {len(texts)} texts to embeddings with shape: {embeddings.shape}"
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query for retrieval.

        Args:
            query: Query string to encode.

        Returns:
            np.ndarray: Query embedding with shape (1, embedding_dim).
        """
        query_embeddings = self.model.encode([query], convert_to_numpy=True)
        logger.info(f"Encoded query to embedding with shape: {query_embeddings.shape}")
        return query_embeddings


class LLMModel:
    """LLM model wrapper for text generation.

    This class provides a unified interface for Large Language Models,
    handling model initialization, API client setup, and text generation.
    It supports different LLM providers with consistent error handling
    and configuration management.
    """

    def __init__(self, model_name: str = None, provider: str = None):
        """Initialize LLM model.

        Args:
            model_name: Name of the LLM model. Defaults to settings.llm_model.
            provider: LLM provider. Defaults to settings.llm_provider.

        Raises:
            LLMConfigError: If API key is not configured.
            LLMProviderError: If provider is not supported.
        """
        self.model_name = model_name or settings.llm_model
        self.provider = provider or settings.llm_provider
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.top_p = settings.llm_top_p
        self.frequency_penalty = settings.llm_frequency_penalty
        self.presence_penalty = settings.llm_presence_penalty

        if self.provider == "openai":
            if not settings.api_key:
                raise LLMConfigError("OpenAI API key is not configured", "api_key")

            self.client = OpenAI(api_key=settings.api_key, base_url=settings.api_host)
        else:
            raise LLMProviderError(f"Unsupported LLM provider: {self.provider}")

    def generate_with_messages(self, system_message: str, user_message: str) -> str:
        """Generate response using system and user messages.

        Args:
            system_message: System prompt that defines the LLM's behavior and context.
            user_message: User message containing the actual query or request.

        Returns:
            str: Generated response text from the LLM.

        Raises:
            OpenAIAPIAuthenticationError: If API authentication fails.
            LLMRateLimitError: If rate limit is exceeded.
            LLMTimeoutError: If request times out.
            OpenAIAPIServerError: If OpenAI server returns an error.
            LLMProviderError: For other LLM-related errors.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Try to provide more specific error information
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "401" in error_msg:
                raise OpenAIAPIAuthenticationError(
                    f"OpenAI authentication failed: {error_msg}"
                )
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                raise LLMRateLimitError(f"OpenAI rate limit exceeded: {error_msg}")
            elif "timeout" in error_msg.lower():
                raise LLMTimeoutError(f"OpenAI request timed out: {error_msg}")
            elif "500" in error_msg or "server" in error_msg.lower():
                raise OpenAIAPIServerError(f"OpenAI server error: {error_msg}")
            else:
                raise LLMProviderError(f"OpenAI API error: {error_msg}")
