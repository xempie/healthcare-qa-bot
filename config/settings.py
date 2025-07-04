"""
Configuration settings for the Healthcare Q&A Bot.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.3
    
    # Embedding Configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_chunk_size: int = 1000
    embedding_chunk_overlap: int = 100
    
    # Vector Store Configuration
    vector_store_type: str = "chromadb"
    vector_store_path: str = "./data/vectorstore"
    collection_name: str = "healthcare_qa"
    
    # Application Configuration
    app_name: str = "Healthcare Q&A Bot"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"
    
    # Medical Safety Configuration
    enable_medical_disclaimer: bool = True
    enable_content_filtering: bool = True
    max_response_length: int = 2000
    
    # Rate Limiting
    max_queries_per_minute: int = 10
    max_queries_per_hour: int = 100
    
    # Document Processing
    max_document_size: int = 10485760  # 10MB
    supported_formats: List[str] = ["pdf", "docx", "txt", "pptx", "xlsx"]
    
    # Paths
    data_dir: str = "./data"
    documents_dir: str = "./data/documents"
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        if not v or v == "your_openai_api_key_here":
            raise ValueError("OpenAI API key must be provided")
        return v
    
    @validator('vector_store_path')
    def create_vector_store_path(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('documents_dir')
    def create_documents_dir(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Medical disclaimers and safety messages
MEDICAL_DISCLAIMER = """
**Important Medical Disclaimer:**
This information is for educational purposes only and should not replace professional medical advice. 
Always consult with qualified healthcare professionals for medical concerns, diagnosis, or treatment decisions.
"""

EMERGENCY_MESSAGE = """
**Emergency Notice:**
If you are experiencing a medical emergency, please contact:
- Emergency Services: 911 (US) or your local emergency number
- Poison Control: 1-800-222-1222 (US)
- Crisis Hotline: 988 (US) or your local crisis line
"""

# Content filtering keywords
EMERGENCY_KEYWORDS = [
    "emergency", "urgent", "chest pain", "heart attack", "stroke", 
    "bleeding", "overdose", "suicide", "self-harm", "poisoning",
    "severe pain", "can't breathe", "unconscious"
]

RESTRICTED_KEYWORDS = [
    "diagnose", "diagnosis", "prescribe", "prescription", "medication dosage",
    "treatment plan", "medical advice", "should I take", "how much medicine"
]
