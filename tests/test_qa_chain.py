"""
Test cases for Q&A chains.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chains.qa_chain import HealthcareQAChain, MedicalTerminologyChain, ConversationChain
from document_processing.vectorstore import MedicalRetriever
from config.settings import settings


class TestHealthcareQAChain:
    """Test cases for HealthcareQAChain."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create a mock medical retriever."""
        mock_retriever = Mock(spec=MedicalRetriever)
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        mock_retriever.vectorstore.vectorstore = mock_vectorstore
        return mock_retriever
    
    @pytest.fixture
    def qa_chain(self, mock_retriever):
        """Create a QA chain instance."""
        with patch('chains.qa_chain.ChatOpenAI') as mock_llm:
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            
            chain = HealthcareQAChain(mock_retriever)
            return chain
    
    def test_safety_classification_safe(self, qa_chain):
        """Test safety classification for safe queries."""
        with patch.object(qa_chain.safety_chain, 'run') as mock_run:
            mock_run.return_value = "Classification: SAFE\nExplanation: General health information question"
            
            result = qa_chain._classify_safety("What is diabetes?")
            
            assert result["classification"] == "SAFE"
            assert "General health information" in result["explanation"]
    
    def test_safety_classification_emergency(self, qa_chain):
        """Test safety classification for emergency queries."""
        with patch.object(qa_chain.safety_chain, 'run') as mock_run:
            mock_run.return_value = "Classification: EMERGENCY\nExplanation: Requires immediate medical attention"
            
            result = qa_chain._classify_safety("I'm having chest pain")
            
            assert result["classification"] == "EMERGENCY"
            assert "immediate medical attention" in result["explanation"]
    
    def test_safety_classification_restricted(self, qa_chain):
        """Test safety classification for restricted queries."""
        with patch.object(qa_chain.safety_chain, 'run') as mock_run:
            mock_run.return_value = "Classification: RESTRICTED\nExplanation: Requires professional medical consultation"
            
            result = qa_chain._classify_safety("Should I take this medication?")
            
            assert result["classification"] == "RESTRICTED"
            assert "professional medical consultation" in result["explanation"]
    
    def test_handle_emergency_query(self, qa_chain):
        """Test handling of emergency queries."""
        safety_result = {
            "classification": "EMERGENCY",
            "explanation": "Chest pain requires immediate attention",
            "token_usage": {"total_tokens": 50}
        }
        
        result = qa_chain._handle_emergency_query("chest pain", safety_result)
        
        assert result["safety_classification"] == "EMERGENCY"
        assert result["requires_immediate_attention"] is True
        assert "911" in result["answer"] or "emergency" in result["answer"]
    
    def test_handle_restricted_query(self, qa_chain):
        """Test handling of restricted queries."""
        safety_result = {
            "classification": "RESTRICTED",
            "explanation": "Medication advice requires professional consultation",
            "token_usage": {"total_tokens": 40}
        }
        
        result = qa_chain._handle_restricted_query("medication dosage", safety_result)
