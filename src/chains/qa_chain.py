"""
Question-answering chains for healthcare Q&A bot.
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.callbacks import get_openai_callback
from loguru import logger

from config.settings import settings, MEDICAL_DISCLAIMER, EMERGENCY_MESSAGE
from src.prompts.medical_prompts import (
    MEDICAL_QA_CHAT_PROMPT,
    SAFETY_CLASSIFICATION_PROMPT,
    FOLLOW_UP_PROMPT,
    TERMINOLOGY_EXPLANATION_PROMPT,
    SYMPTOM_CHECKER_DISCLAIMER,
    MEDICATION_DISCLAIMER
)
from src.document_processing.vectorstore import MedicalRetriever
from src.utils.validators import validate_medical_query, is_emergency_query


class HealthcareQAChain:
    """Main Q&A chain for healthcare questions."""
    
    def __init__(self, retriever: MedicalRetriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model_name=settings.openai_model,
            temperature=settings.openai_temperature,
            openai_api_key=settings.openai_api_key
        )
        self.safety_chain = self._create_safety_chain()
        self.qa_chain = self._create_qa_chain()
        
    def _create_safety_chain(self) -> LLMChain:
        """Create safety classification chain."""
        return LLMChain(
            llm=self.llm,
            prompt=SAFETY_CLASSIFICATION_PROMPT,
            verbose=settings.debug
        )
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the main Q&A chain."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever.vectorstore.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={
                "prompt": MEDICAL_QA_CHAT_PROMPT,
                "verbose": settings.debug
            },
            return_source_documents=True
        )
    
    def answer_question(
        self,
        question: str,
        include_sources: bool = True,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Answer a healthcare question with safety checks."""
        
        # Validate input
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question about healthcare topics.",
                "sources": [],
                "safety_classification": "INVALID",
                "token_usage": {}
            }
        
        try:
            # Safety classification
            safety_result = self._classify_safety(question)
            
            # Handle different safety classifications
            if safety_result["classification"] == "EMERGENCY":
                return self._handle_emergency_query(question, safety_result)
            elif safety_result["classification"] == "RESTRICTED":
                return self._handle_restricted_query(question, safety_result)
            elif safety_result["classification"] == "INAPPROPRIATE":
                return self._handle_inappropriate_query(question, safety_result)
            
            # Process safe queries
            return self._process_safe_query(question, include_sources, max_tokens)
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again or contact support if the issue persists.",
                "sources": [],
                "safety_classification": "ERROR",
                "error": str(e),
                "token_usage": {}
            }
    
    def _classify_safety(self, question: str) -> Dict[str, str]:
        """Classify question safety."""
        try:
            with get_openai_callback() as cb:
                result = self.safety_chain.run(question=question)
                
            # Parse the result
            lines = result.strip().split('\n')
            classification = "SAFE"
            explanation = ""
            
            for line in lines:
                if line.startswith("Classification:"):
                    classification = line.split(":", 1)[1].strip()
                elif line.startswith("Explanation:"):
                    explanation = line.split(":", 1)[1].strip()
            
            return {
                "classification": classification,
                "explanation": explanation,
                "token_usage": {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error in safety classification: {str(e)}")
            return {
                "classification": "ERROR",
                "explanation": "Could not classify question safety",
                "token_usage": {}
            }
    
    def _handle_emergency_query(self, question: str, safety_result: Dict) -> Dict[str, Any]:
        """Handle emergency medical queries."""
        return {
            "answer": f"{EMERGENCY_MESSAGE}\n\n{safety_result['explanation']}",
            "sources": [],
            "safety_classification": "EMERGENCY",
            "token_usage": safety_result.get("token_usage", {}),
            "requires_immediate_attention": True
        }
    
    def _handle_restricted_query(self, question: str, safety_result: Dict) -> Dict[str, Any]:
        """Handle restricted medical queries."""
        disclaimer = MEDICAL_DISCLAIMER
        
        # Try to provide general educational information
        try:
            general_info = self._get_general_medical_info(question)
            answer = f"{disclaimer}\n\n{general_info}\n\n{safety_result['explanation']}"
        except:
            answer = f"{disclaimer}\n\n{safety_result['explanation']}"
        
        return {
            "answer": answer,
            "sources": [],
            "safety_classification": "RESTRICTED",
            "token_usage": safety_result.get("token_usage", {}),
            "requires_professional_consultation": True
        }
    
    def _handle_inappropriate_query(self, question: str, safety_result: Dict) -> Dict[str, Any]:
        """Handle inappropriate queries."""
        return {
            "answer": f"I'm designed to provide general healthcare information. {safety_result['explanation']} Please ask questions related to general health topics, symptoms, or medical conditions.",
            "sources": [],
            "safety_classification": "INAPPROPRIATE",
            "token_usage": safety_result.get("token_usage", {})
        }
    
    def _process_safe_query(
        self,
        question: str,
        include_sources: bool,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Process safe healthcare queries."""
        try:
            with get_openai_callback() as cb:
                # Run the Q&A chain
                result = self.qa_chain({"query": question})
                
            answer = result["result"]
            source_documents = result.get("source_documents", [])
            
            # Add appropriate disclaimers based on content
            answer = self._add_context_specific_disclaimers(answer, question)
            
            # Prepare response
            response = {
                "answer": answer,
                "safety_classification": "SAFE",
                "token_usage": {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens
                }
            }
            
            # Add sources if requested
            if include_sources and source_documents:
                response["sources"] = self._format_sources(source_documents)
            else:
                response["sources"] = []
            
            # Add follow-up suggestions
            response["follow_up_questions"] = self._generate_follow_up_questions(question, answer)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing safe query: {str(e)}")
            raise
    
    def _get_general_medical_info(self, question: str) -> str:
        """Get general medical information for restricted queries."""
        # Use retrieval to get educational content only
        docs = self.retriever.retrieve_medical_info(
            query=question,
            k=3,
            content_type="patient_education"
        )
        
        if docs:
            # Combine educational content
            educational_content = "\n\n".join([doc.page_content for doc in docs[:2]])
            return f"Here's some general educational information:\n\n{educational_content}"
        else:
            return "I recommend consulting with a healthcare professional for personalized medical advice."
    
    def _add_context_specific_disclaimers(self, answer: str, question: str) -> str:
        """Add appropriate disclaimers based on content."""
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        disclaimers = []
        
        # Add medical disclaimer if not already present
        if settings.enable_medical_disclaimer and "disclaimer" not in answer_lower:
            disclaimers.append(MEDICAL_DISCLAIMER)
        
        # Add medication disclaimer for drug-related questions
        if any(term in question_lower for term in ['medication', 'drug', 'prescription', 'dosage', 'pill']):
            disclaimers.append(MEDICATION_DISCLAIMER)
        
        # Add symptom checker disclaimer for symptom-related questions
        if any(term in question_lower for term in ['symptom', 'pain', 'ache', 'feel', 'hurt']):
            disclaimers.append(SYMPTOM_CHECKER_DISCLAIMER)
        
        # Combine answer with disclaimers
        if disclaimers:
            return f"{answer}\n\n" + "\n\n".join(disclaimers)
        
        return answer
    
    def _format_sources(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for response."""
        sources = []
        
        for i, doc in enumerate(source_documents):
            source = {
                "index": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source", "Unknown"),
                    "document_type": doc.metadata.get("document_type", "Unknown"),
                    "content_type": doc.metadata.get("content_type", "Unknown"),
                    "medical_specialty": doc.metadata.get("medical_specialty", "General")
                }
            }
            sources.append(source)
        
        return sources
    
    def _generate_follow_up_questions(self, question: str, answer: str) -> List[str]:
        """Generate follow-up questions."""
        try:
            follow_up_chain = LLMChain(llm=self.llm, prompt=FOLLOW_UP_PROMPT)
            result = follow_up_chain.run(question=question, answer=answer)
            
            # Parse follow-up questions
            lines = result.strip().split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    question_text = line.split('.', 1)[1].strip()
                    if question_text:
                        questions.append(question_text)
            
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            return []


class MedicalTerminologyChain:
    """Chain for explaining medical terminology."""
    
    def __init__(self, retriever: MedicalRetriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model_name=settings.openai_model,
            temperature=0.2,  # Lower temperature for more consistent explanations
            openai_api_key=settings.openai_api_key
        )
        self.terminology_chain = LLMChain(
            llm=self.llm,
            prompt=TERMINOLOGY_EXPLANATION_PROMPT
        )
    
    def explain_term(self, term: str, context: str = "") -> Dict[str, Any]:
        """Explain a medical term in simple language."""
        try:
            # Get relevant documents about the term
            docs = self.retriever.retrieve_medical_info(
                query=f"medical term definition {term}",
                k=3
            )
            
            # Use context from documents if available
            if docs:
                context = context or docs[0].page_content[:500]
            
            with get_openai_callback() as cb:
                explanation = self.terminology_chain.run(term=term, context=context)
            
            return {
                "term": term,
                "explanation": explanation,
                "context_used": bool(context),
                "token_usage": {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error explaining term '{term}': {str(e)}")
            return {
                "term": term,
                "explanation": f"I couldn't find a detailed explanation for '{term}'. Please consult a medical dictionary or healthcare professional for accurate information.",
                "context_used": False,
                "error": str(e)
            }


class ConversationChain:
    """Chain for managing conversational context."""
    
    def __init__(self, qa_chain: HealthcareQAChain):
        self.qa_chain = qa_chain
        self.conversation_history = []
        self.max_history_length = 10
    
    def add_exchange(self, question: str, answer: str) -> None:
        """Add a Q&A exchange to conversation history."""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": logger._core.time()
        })
        
        # Limit history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_contextual_answer(self, question: str) -> Dict[str, Any]:
        """Get answer with conversational context."""
        # Add conversation context to question if relevant
        contextual_question = self._add_conversation_context(question)
        
        # Get answer
        result = self.qa_chain.answer_question(contextual_question)
        
        # Add to conversation history
        if result.get("safety_classification") == "SAFE":
            self.add_exchange(question, result["answer"])
        
        return result
    
    def _add_conversation_context(self, question: str) -> str:
        """Add relevant conversation context to the question."""
        if not self.conversation_history:
            return question
        
        # Check if question refers to previous context
        context_indicators = ["that", "this", "it", "what about", "more about", "also"]
        
        if any(indicator in question.lower() for indicator in context_indicators):
            # Add context from last exchange
            last_exchange = self.conversation_history[-1]
            context = f"Previous question: {last_exchange['question']}\nPrevious answer: {last_exchange['answer'][:200]}...\n\nCurrent question: {question}"
            return context
        
        return question
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary_parts = []
        for i, exchange in enumerate(self.conversation_history[-5:], 1):  # Last 5 exchanges
            summary_parts.append(f"{i}. Q: {exchange['question'][:100]}...")
        
        return "Recent topics discussed:\n" + "\n".join(summary_parts)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


def create_qa_system(retriever: MedicalRetriever) -> Dict[str, Any]:
    """Create the complete Q&A system."""
    try:
        # Create main components
        qa_chain = HealthcareQAChain(retriever)
        terminology_chain = MedicalTerminologyChain(retriever)
        conversation_chain = ConversationChain(qa_chain)
        
        logger.info("Q&A system created successfully")
        
        return {
            "qa_chain": qa_chain,
            "terminology_chain": terminology_chain,
            "conversation_chain": conversation_chain,
            "status": "ready"
        }
        
    except Exception as e:
        logger.error(f"Error creating Q&A system: {str(e)}")
        return {
            "qa_chain": None,
            "terminology_chain": None,
            "conversation_chain": None,
            "status": "error",
            "error": str(e)
        }
