import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from settings import settings
import logging
from exceptions import LLMProviderError, EmbeddingModelError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    # Enable embedding model to be switched out, set model_name in settings.py
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.model_name
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Initialized embedding model: {self.model_name}")
        if not self.model:
            raise EmbeddingModelError(f"Failed to initialize embedding model: {self.model_name}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        # Encode texts to embeddings
        return self.model.encode(texts, convert_to_numpy=True)
        logger.info(f"Encoded {len(texts)} texts to embeddings with shape: {embeddings.shape}")
    
    def encode_query(self, query: str) -> np.ndarray:
        # Encode a query for retrieval
        return self.model.encode([query], convert_to_numpy=True)
        logger.info(f"Encoded query to embedding with shape: {query_embeddings.shape}")

class LLMModel:
    # Enable LLM model to be switched out, set model_name and provider in settings.py
    def __init__(self, model_name: str = None, provider: str = None):
        self.model_name = model_name or settings.llm_model
        self.provider = provider or settings.llm_provider
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.top_p = settings.llm_top_p
        self.frequency_penalty = settings.llm_frequency_penalty
        self.presence_penalty = settings.llm_presence_penalty
        
        # Initialize client based on provider (default: openai)
        if self.provider == "openai":
            self.client = OpenAI(
                api_key=settings.api_key,
                base_url=settings.api_host
            )
        else:
            raise LLMProviderError(f"Unsupported LLM provider: {self.provider}")
    
    def generate(self, prompt: str) -> str:
        # Generate raw text response from prompt
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM generation failed: {e}")