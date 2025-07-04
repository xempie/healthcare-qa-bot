"""
Medical agent for healthcare Q&A bot using LangChain agents.
"""

from typing import List, Dict, Any, Optional, Type
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish, BaseMessage
from langchain.tools.base import BaseTool
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field
from loguru import logger

from config.settings import settings, EMERGENCY_KEYWORDS, RESTRICTED_KEYWORDS
from src.document_processing.vectorstore import MedicalRetriever
from src.chains.qa_chain import MedicalTerminologyChain
from src.utils.validators import validate_medical_query, is_emergency_query


class MedicalSearchTool(BaseTool):
    """Tool for searching medical information."""
    
    name = "medical_search"
    description = "Search for medical information, symptoms, treatments, and conditions. Use this for general health questions."
    
    def __init__(self, retriever: MedicalRetriever):
        super().__init__()
        self.retriever = retriever
    
    def _run(self, query: str) -> str:
        """Run the medical search."""
        try:
            docs = self.retriever.retrieve_medical_info(query, k=3)
            
            if not docs:
                return "No relevant medical information found. Please try a different query or consult a healthcare professional."
            
            # Combine results
            results = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                source = doc.metadata.get('source', 'Unknown source')
                results.append(f"{i}. {content}\n   Source: {source}")
            
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in medical search: {str(e)}")
            return f"Error searching medical information: {str(e)}"


class FAQSearchTool(BaseTool):
    """Tool for searching frequently asked questions."""
    
    name = "faq_search"
    description = "Search for answers to frequently asked questions about health topics. Use this for common health questions."
    
    def __init__(self, retriever: MedicalRetriever):
        super().__init__()
        self.retriever = retriever
    
    def _run(self, query: str) -> str:
        """Run the FAQ search."""
        try:
            docs = self.retriever.retrieve_faq_answers(query, k=3)
            
            if not docs:
                return "No FAQ answers found for this query."
            
            results = []
            for doc in docs:
                if 'question' in doc.metadata and 'answer' in doc.metadata:
                    results.append(f"Q: {doc.metadata['question']}\nA: {doc.metadata['answer']}")
                else:
                    results.append(doc.page_content)
            
            return "\n\n---\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in FAQ search: {str(e)}")
            return f"Error searching FAQ: {str(e)}"


class SpecialtySearchTool(BaseTool):
    """Tool for searching by medical specialty."""
    
    name = "specialty_search"
    description = "Search for medical information by specialty (cardiology, dermatology, etc.). Specify the specialty and condition."
    
    def __init__(self, retriever: MedicalRetriever):
        super().__init__()
        self.retriever = retriever
    
    def _run(self, query: str) -> str:
        """Run the specialty search."""
        try:
            # Extract specialty from query
            specialties = {
                'cardiology': ['heart', 'cardiac', 'cardiovascular'],
                'dermatology': ['skin', 'dermatology'],
                'endocrinology': ['diabetes', 'thyroid', 'hormone'],
                'gastroenterology': ['stomach', 'digestive', 'gut'],
                'neurology': ['brain', 'neurological', 'nerve'],
                'orthopedics': ['bone', 'joint', 'muscle'],
                'pulmonology': ['lung', 'respiratory', 'breathing']
            }
            
            detected_specialty = None
            query_lower = query.lower()
            
            for specialty, keywords in specialties.items():
                if any(keyword in query_lower for keyword in keywords):
                    detected_specialty = specialty
                    break
            
            if detected_specialty:
                docs = self.retriever.retrieve_by_specialty(query, detected_specialty, k=3)
            else:
                docs = self.retriever.retrieve_medical_info(query, k=3)
            
            if not docs:
                return f"No information found for {detected_specialty or 'this specialty'} related to your query."
            
            results = []
            for doc in docs:
                specialty = doc.metadata.get('medical_specialty', 'General')
                content = doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                results.append(f"[{specialty.title()}] {content}")
            
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in specialty search: {str(e)}")
            return f"Error searching specialty information: {str(e)}"


class TerminologyTool(BaseTool):
    """Tool for explaining medical terminology."""
    
    name = "terminology_explainer"
    description = "Explain medical terms and terminology in simple language. Use this when users ask about medical terms they don't understand."
    
    def __init__(self, terminology_chain: MedicalTerminologyChain):
        super().__init__()
        self.terminology_chain = terminology_chain
    
    def _run(self, term: str) -> str:
        """Explain a medical term."""
        try:
            result = self.terminology_chain.explain_term(term)
            return result.get('explanation', f"Could not find explanation for '{term}'")
            
        except Exception as e:
            logger.error(f"Error explaining term '{term}': {str(e)}")
            return f"Error explaining medical term: {str(e)}"


class SafetyCheckTool(BaseTool):
    """Tool for safety checks and emergency detection."""
    
    name = "safety_check"
    description = "Check if a query involves emergency situations or requires immediate medical attention. Use this for urgent or concerning symptoms."
    
    def _run(self, query: str) -> str:
        """Perform safety check."""
        try:
            query_lower = query.lower()
            
            # Check for emergency keywords
            emergency_detected = any(keyword in query_lower for keyword in EMERGENCY_KEYWORDS)
            
            if emergency_detected:
                return """EMERGENCY DETECTED: This appears to be a medical emergency. 
                
IMMEDIATE ACTION REQUIRED:
- Call 911 (US) or your local emergency number immediately
- Do not delay seeking professional medical care
- If unconscious or not breathing, begin CPR if trained
                
This AI system cannot provide emergency medical care. Seek immediate professional help."""
            
            # Check for restricted content
            restricted_detected = any(keyword in query_lower for keyword in RESTRICTED_KEYWORDS)
            
            if restricted_detected:
                return """MEDICAL CONSULTATION REQUIRED: This question requires professional medical evaluation.
                
Please consult with a healthcare provider for:
- Medical diagnosis or treatment decisions
- Prescription medication advice
- Specific medical recommendations
                
I can provide general educational information only."""
            
            return "Query appears safe for general health information discussion."
            
        except Exception as e:
            logger.error(f"Error in safety check: {str(e)}")
            return f"Error performing safety check: {str(e)}"


class MedicalAgent:
    """Main medical agent coordinator."""
    
    def __init__(self, retriever: MedicalRetriever, terminology_chain: MedicalTerminologyChain):
        self.retriever = retriever
        self.terminology_chain = terminology_chain
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=settings.openai_model,
            temperature=settings.openai_temperature,
            openai_api_key=settings.openai_api_key
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=settings.debug,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent."""
        return [
            SafetyCheckTool(),
            MedicalSearchTool(self.retriever),
            FAQSearchTool(self.retriever),
            SpecialtySearchTool(self.retriever),
            TerminologyTool(self.terminology_chain)
        ]
    
    def _create_agent(self):
        """Create the OpenAI functions agent."""
        
        # System prompt for medical agent
        system_prompt = """You are a helpful healthcare information assistant. Your role is to provide accurate, evidence-based health information while prioritizing patient safety.

IMPORTANT GUIDELINES:
1. Always prioritize patient safety - use the safety_check tool first for any concerning queries
2. Provide accurate information based on medical documents and reliable sources
3. Never provide specific medical advice, diagnoses, or treatment recommendations
4. Always encourage users to consult healthcare professionals for personal medical concerns
5. Use appropriate tools to search for relevant information
6. Explain medical terms in simple, patient-friendly language
7. Include relevant disclaimers and safety information

TOOL USAGE:
- Use safety_check first for any urgent or concerning symptoms
- Use medical_search for general health information
- Use faq_search for common questions
- Use specialty_search when specific medical specialties are mentioned
- Use terminology_explainer for medical term definitions

Remember: You provide educational health information, not personalized medical advice."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        return create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a healthcare query using the agent."""
        try:
            # Validate query
            if not query or not query.strip():
                return {
                    "response": "Please provide a valid healthcare question.",
                    "tool_calls": [],
                    "safety_status": "invalid_query",
                    "token_usage": {}
                }
            
            with get_openai_callback() as cb:
                # Run the agent
                result = self.agent_executor.invoke({"input": query})
            
            return {
                "response": result["output"],
                "tool_calls": self._extract_tool_calls(result),
                "safety_status": self._determine_safety_status(result["output"]),
                "token_usage": {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens
                },
                "intermediate_steps": result.get("intermediate_steps", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing query with agent: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again or contact support if the issue persists.",
                "tool_calls": [],
                "safety_status": "error",
                "error": str(e),
                "token_usage": {}
            }
    
    def _extract_tool_calls(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract tool calls from agent result."""
        tool_calls = []
        
        intermediate_steps = result.get("intermediate_steps", [])
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                if isinstance(action, AgentAction):
                    tool_calls.append({
                        "tool": action.tool,
                        "tool_input": str(action.tool_input),
                        "observation": str(observation)[:200] + "..." if len(str(observation)) > 200 else str(observation)
                    })
        
        return tool_calls
    
    def _determine_safety_status(self, response: str) -> str:
        """Determine safety status from response."""
        response_lower = response.lower()
        
        if "emergency detected" in response_lower:
            return "emergency"
        elif "medical consultation required" in response_lower:
            return "requires_consultation"
        elif "disclaimer" in response_lower or "consult" in response_lower:
            return "safe_with_disclaimer"
        else:
            return "safe"
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]


def create_medical_agent_system(
    retriever: MedicalRetriever,
    terminology_chain: MedicalTerminologyChain
) -> MedicalAgent:
    """Create the medical agent system."""
    try:
        agent = MedicalAgent(retriever, terminology_chain)
        logger.info("Medical agent system created successfully")
        return agent
    except Exception as e:
        logger.error(f"Error creating medical agent system: {str(e)}")
        raise
