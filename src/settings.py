from pydantic import Secret
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv

class Settings(BaseSettings):
    # Embedding model
    # option 1: sentence-transformers/all-MiniLM-L6-v2
    # option 2: BAAI/bge-small-en-v1.5
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector index (FAISS, future: Qdrant)
    index_dir: str = "src/data/index"
    top_k: int = 3 # number of top results to return
    rebuild_index: bool = False # if True, will rebuild the index from scratch

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    llm_top_p: float = 1.0
    llm_frequency_penalty: float = 0.0
    llm_presence_penalty: float = 0.0

    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'extra': 'ignore',
        'case_sensitive': False
    }

    # LLM Response Constraints
    allowed_actions: List[str] = ['none', 'escalate_to_support', 'escalate_to_abuse_team', 'contact_customer']
    max_references: int = 3
    
    # API
    api_host: str = "https://api.openai.com/v1"
    api_key: str | None = None  # load from env file
    api_port: int = 8000

    # Paths
    docs_path: str = "src/data/docs.json"
    index_path: str = "faiss_index.bin"
    
settings = Settings()

# Load environment variables from .env file
load_dotenv()
settings = Settings()

# Manually set api_key from environment variable if Pydantic didn't load it
if not settings.api_key and os.getenv('OPENAI_API_KEY'):
    settings.api_key = os.getenv('OPENAI_API_KEY')

# Validate critical environment variables
if not settings.api_key:
    print("Warning: OPENAI_API_KEY environment variable is not set")
    print("Please create a .env file with your OpenAI API key:")
    print("OPENAI_API_KEY=your_actual_api_key_here")
else:
    print("OpenAI API key loaded successfully")