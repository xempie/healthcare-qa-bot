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
        
        assert result["safety_classification"] == "RESTRICTED"
        assert result["requires_professional_consultation"] is True
        assert "healthcare provider" in result["answer"] or "medical advice" in result["answer"]
    
    def test_process_safe_query(self, qa_chain):
        """Test processing of safe queries."""
        mock_qa_result = {
            "result": "Diabetes is a condition where blood sugar levels are too high.",
            "source_documents": [
                Document(
                    page_content="Diabetes information content",
                    metadata={"source": "diabetes_guide.pdf", "document_type": "medical_guideline"}
                )
            ]
        }
        
        with patch.object(qa_chain.qa_chain, '__call__') as mock_call:
            mock_call.return_value = mock_qa_result
            
            result = qa_chain._process_safe_query("What is diabetes?", True, None)
            
            assert result["safety_classification"] == "SAFE"
            assert "blood sugar" in result["answer"]
            assert len(result["sources"]) > 0
    
    def test_add_context_specific_disclaimers(self, qa_chain):
        """Test addition of context-specific disclaimers."""
        # Test medication disclaimer
        answer = "Take this medication with food."
        question = "How should I take my medication?"
        
        result = qa_chain._add_context_specific_disclaimers(answer, question)
        
        assert "medication" in result.lower() or "disclaimer" in result.lower()
        
        # Test symptom disclaimer
        answer = "These are common symptoms."
        question = "What are the symptoms of flu?"
        
        result = qa_chain._add_context_specific_disclaimers(answer, question)
        
        assert len(result) > len(answer)  # Disclaimer added
    
    def test_generate_follow_up_questions(self, qa_chain):
        """Test generation of follow-up questions."""
        with patch('chains.qa_chain.LLMChain') as mock_chain_class:
            mock_chain = Mock()
            mock_chain.run.return_value = "1. What causes diabetes?\n2. How is diabetes treated?\n3. Can diabetes be prevented?"
            mock_chain_class.return_value = mock_chain
            
            questions = qa_chain._generate_follow_up_questions(
                "What is diabetes?",
                "Diabetes is a condition affecting blood sugar."
            )
            
            assert len(questions) == 3
            assert "causes diabetes" in questions[0]
            assert "treated" in questions[1]
            assert "prevented" in questions[2]
    
    def test_answer_question_integration(self, qa_chain):
        """Test the complete answer_question workflow."""
        # Mock safety classification as SAFE
        with patch.object(qa_chain, '_classify_safety') as mock_classify:
            mock_classify.return_value = {
                "classification": "SAFE",
                "explanation": "Safe health information question",
                "token_usage": {"total_tokens": 30}
            }
            
            # Mock the QA chain call
            with patch.object(qa_chain, '_process_safe_query') as mock_process:
                mock_process.return_value = {
                    "answer": "Diabetes is a metabolic condition.",
                    "sources": [],
                    "safety_classification": "SAFE",
                    "follow_up_questions": ["What causes diabetes?"],
                    "token_usage": {"total_tokens": 100}
                }
                
                result = qa_chain.answer_question("What is diabetes?")
                
                assert result["safety_classification"] == "SAFE"
                assert "metabolic condition" in result["answer"]
                assert len(result["follow_up_questions"]) > 0


class TestMedicalTerminologyChain:
    """Test cases for MedicalTerminologyChain."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create a mock medical retriever."""
        mock_retriever = Mock(spec=MedicalRetriever)
        return mock_retriever
    
    @pytest.fixture
    def terminology_chain(self, mock_retriever):
        """Create a terminology chain instance."""
        with patch('chains.qa_chain.ChatOpenAI') as mock_llm:
            chain = MedicalTerminologyChain(mock_retriever)
            return chain
    
    def test_explain_term_with_context(self, terminology_chain):
        """Test explaining a medical term with context."""
        # Mock retriever to return relevant documents
        mock_docs = [
            Document(
                page_content="Hypertension is high blood pressure, a common cardiovascular condition.",
                metadata={"source": "cardiology_guide.pdf"}
            )
        ]
        
        terminology_chain.retriever.retrieve_medical_info.return_value = mock_docs
        
        with patch.object(terminology_chain.terminology_chain, 'run') as mock_run:
            mock_run.return_value = "Hypertension means high blood pressure. It's when the force of blood against artery walls is too high."
            
            result = terminology_chain.explain_term("hypertension")
            
            assert result["term"] == "hypertension"
            assert "blood pressure" in result["explanation"]
            assert result["context_used"] is True
    
    def test_explain_term_without_context(self, terminology_chain):
        """Test explaining a medical term without context."""
        # Mock retriever to return no documents
        terminology_chain.retriever.retrieve_medical_info.return_value = []
        
        with patch.object(terminology_chain.terminology_chain, 'run') as mock_run:
            mock_run.return_value = "Tachycardia means a fast heart rate, usually over 100 beats per minute."
            
            result = terminology_chain.explain_term("tachycardia")
            
            assert result["term"] == "tachycardia"
            assert "heart rate" in result["explanation"]
            assert result["context_used"] is False
    
    def test_explain_term_error_handling(self, terminology_chain):
        """Test error handling in term explanation."""
        # Mock retriever to raise an exception
        terminology_chain.retriever.retrieve_medical_info.side_effect = Exception("Database error")
        
        result = terminology_chain.explain_term("unknown_term")
        
        assert result["term"] == "unknown_term"
        assert "couldn't find" in result["explanation"]
        assert "error" in result


class TestConversationChain:
    """Test cases for ConversationChain."""
    
    @pytest.fixture
    def mock_qa_chain(self):
        """Create a mock QA chain."""
        mock_chain = Mock(spec=HealthcareQAChain)
        return mock_chain
    
    @pytest.fixture
    def conversation_chain(self, mock_qa_chain):
        """Create a conversation chain instance."""
        return ConversationChain(mock_qa_chain)
    
    def test_add_exchange(self, conversation_chain):
        """Test adding Q&A exchanges to conversation history."""
        conversation_chain.add_exchange("What is diabetes?", "Diabetes is a metabolic condition.")
        
        assert len(conversation_chain.conversation_history) == 1
        assert conversation_chain.conversation_history[0]["question"] == "What is diabetes?"
        assert conversation_chain.conversation_history[0]["answer"] == "Diabetes is a metabolic condition."
    
    def test_conversation_history_limit(self, conversation_chain):
        """Test conversation history length limiting."""
        # Add more than max_history_length exchanges
        for i in range(15):
            conversation_chain.add_exchange(f"Question {i}", f"Answer {i}")
        
        # Should be limited to max_history_length
        assert len(conversation_chain.conversation_history) == conversation_chain.max_history_length
        
        # Should contain the most recent exchanges
        assert conversation_chain.conversation_history[-1]["question"] == "Question 14"
    
    def test_add_conversation_context(self, conversation_chain):
        """Test adding conversation context to questions."""
        # Add some conversation history
        conversation_chain.add_exchange("What is diabetes?", "Diabetes is a metabolic condition affecting blood sugar.")
        
        # Test context addition for referential question
        contextual_question = conversation_chain._add_conversation_context("What about that condition's symptoms?")
        
        assert "Previous question:" in contextual_question
        assert "What is diabetes?" in contextual_question
        assert "Current question:" in contextual_question
    
    def test_no_context_addition(self, conversation_chain):
        """Test no context addition for standalone questions."""
        # Add some conversation history
        conversation_chain.add_exchange("What is diabetes?", "Diabetes is a metabolic condition.")
        
        # Test no context addition for standalone question
        standalone_question = "What is hypertension?"
        contextual_question = conversation_chain._add_conversation_context(standalone_question)
        
        assert contextual_question == standalone_question
    
    def test_get_contextual_answer(self, conversation_chain):
        """Test getting contextual answers."""
        # Mock the QA chain response
        mock_response = {
            "answer": "The symptoms include frequent urination and thirst.",
            "safety_classification": "SAFE",
            "sources": []
        }
        
        conversation_chain.qa_chain.answer_question.return_value = mock_response
        
        result = conversation_chain.get_contextual_answer("What are the symptoms?")
        
        assert result["answer"] == "The symptoms include frequent urination and thirst."
        assert len(conversation_chain.conversation_history) == 1
    
    def test_get_conversation_summary(self, conversation_chain):
        """Test getting conversation summary."""
        # Add some conversation history
        conversation_chain.add_exchange("What is diabetes?", "Diabetes is a metabolic condition.")
        conversation_chain.add_exchange("What are the symptoms?", "Symptoms include thirst and frequent urination.")
        
        summary = conversation_chain.get_conversation_summary()
        
        assert "Recent topics discussed:" in summary
        assert "What is diabetes?" in summary
        assert "What are the symptoms?" in summary
    
    def test_clear_history(self, conversation_chain):
        """Test clearing conversation history."""
        # Add some conversation history
        conversation_chain.add_exchange("Question 1", "Answer 1")
        conversation_chain.add_exchange("Question 2", "Answer 2")
        
        assert len(conversation_chain.conversation_history) == 2
        
        conversation_chain.clear_history()
        
        assert len(conversation_chain.conversation_history) == 0


class TestQASystemIntegration:
    """Integration tests for the complete Q&A system."""
    
    def test_create_qa_system_success(self):
        """Test successful creation of Q&A system."""
        from chains.qa_chain import create_qa_system
        
        mock_retriever = Mock(spec=MedicalRetriever)
        
        with patch('chains.qa_chain.HealthcareQAChain') as mock_qa_class, \
             patch('chains.qa_chain.MedicalTerminologyChain') as mock_term_class, \
             patch('chains.qa_chain.ConversationChain') as mock_conv_class:
            
            mock_qa_class.return_value = Mock()
            mock_term_class.return_value = Mock()
            mock_conv_class.return_value = Mock()
            
            result = create_qa_system(mock_retriever)
            
            assert result["status"] == "ready"
            assert result["qa_chain"] is not None
            assert result["terminology_chain"] is not None
            assert result["conversation_chain"] is not None
    
    def test_create_qa_system_error(self):
        """Test error handling in Q&A system creation."""
        from chains.qa_chain import create_qa_system
        
        mock_retriever = Mock(spec=MedicalRetriever)
        
        with patch('chains.qa_chain.HealthcareQAChain') as mock_qa_class:
            mock_qa_class.side_effect = Exception("Initialization error")
            
            result = create_qa_system(mock_retriever)
            
            assert result["status"] == "error"
            assert "Initialization error" in result["error"]


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test data fixtures
@pytest.fixture
def sample_medical_documents():
    """Sample medical documents for testing."""
    return [
        Document(
            page_content="Diabetes is a group of metabolic disorders characterized by high blood sugar levels.",
            metadata={
                "source": "diabetes_guide.pdf",
                "document_type": "medical_guideline",
                "medical_specialty": "endocrinology"
            }
        ),
        Document(
            page_content="Hypertension, also known as high blood pressure, is a common cardiovascular condition.",
            metadata={
                "source": "cardiology_handbook.pdf",
                "document_type": "clinical_information",
                "medical_specialty": "cardiology"
            }
        ),
        Document(
            page_content="Q: What are the symptoms of diabetes? A: Common symptoms include frequent urination, excessive thirst, and unexplained weight loss.",
            metadata={
                "source": "diabetes_faq.txt",
                "document_type": "faq",
                "question": "What are the symptoms of diabetes?",
                "answer": "Common symptoms include frequent urination, excessive thirst, and unexplained weight loss."
            }
        )
    ]


@pytest.fixture
def sample_emergency_queries():
    """Sample emergency queries for testing."""
    return [
        "I'm having severe chest pain",
        "Can't breathe properly",
        "Think I'm having a heart attack",
        "Severe bleeding that won't stop",
        "Someone is unconscious"
    ]


@pytest.fixture
def sample_safe_queries():
    """Sample safe queries for testing."""
    return [
        "What is diabetes?",
        "How can I prevent heart disease?",
        "What are the benefits of exercise?",
        "Tell me about healthy eating",
        "What is high blood pressure?"
    ]


@pytest.fixture
def sample_restricted_queries():
    """Sample restricted queries for testing."""
    return [
        "Should I take this medication?",
        "What dosage should I use?",
        "Can you diagnose my condition?",
        "What treatment should I get?",
        "Is it safe to stop my medication?"
    ]


if __name__ == "__main__":
    pytest.main([__file__])
