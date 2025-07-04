"""
Main application module for Healthcare Q&A Bot.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from document_processing.document_loader import load_all_documents
from document_processing.text_splitter import create_medical_text_splitter, optimize_chunks_for_retrieval
from document_processing.vectorstore import setup_vectorstore, MedicalRetriever
from chains.qa_chain import create_qa_system
from agents.medical_agent import create_medical_agent_system
from utils.validators import validate_medical_query, format_validation_response


class HealthcareQABot:
    """Main Healthcare Q&A Bot application."""
    
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.qa_system = None
        self.medical_agent = None
        self.is_initialized = False
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logger.remove()  # Remove default handler
        
        # Add console handler
        logger.add(
            sys.stdout,
            level=settings.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # Add file handler
        logger.add(
            "logs/healthcare_qa.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )
        
        logger.info(f"Healthcare Q&A Bot starting up - Version {settings.app_version}")
    
    async def initialize(self, force_rebuild: bool = False) -> bool:
        """Initialize the Q&A bot system."""
        try:
            logger.info("Initializing Healthcare Q&A Bot...")
            
            # Load documents
            logger.info("Loading medical documents...")
            documents = await load_all_documents()
            
            if not documents:
                logger.warning("No documents found. Please add medical documents to continue.")
                return False
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Split documents
            logger.info("Processing and splitting documents...")
            text_splitter = create_medical_text_splitter()
            split_docs = text_splitter.split_documents(documents)
            
            # Optimize chunks for retrieval
            optimized_chunks = optimize_chunks_for_retrieval(split_docs)
            logger.info(f"Created {len(optimized_chunks)} optimized chunks")
            
            # Setup vector store
            logger.info("Setting up vector store...")
            if force_rebuild:
                self.vectorstore = setup_vectorstore(optimized_chunks)
            else:
                self.vectorstore = setup_vectorstore()
                if not self.vectorstore.vectorstore:
                    # No existing store found, create new one
                    self.vectorstore = setup_vectorstore(optimized_chunks)
            
            # Create retriever
            self.retriever = MedicalRetriever(self.vectorstore)
            logger.info("Medical retriever created")
            
            # Create Q&A system
            logger.info("Creating Q&A system...")
            self.qa_system = create_qa_system(self.retriever)
            
            if self.qa_system["status"] != "ready":
                logger.error("Failed to create Q&A system")
                return False
            
            # Create medical agent
            logger.info("Creating medical agent...")
            self.medical_agent = create_medical_agent_system(
                self.retriever,
                self.qa_system["terminology_chain"]
            )
            
            # Get vector store stats
            stats = self.vectorstore.get_collection_stats()
            logger.info(f"Vector store stats: {stats}")
            
            self.is_initialized = True
            logger.info("Healthcare Q&A Bot initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Healthcare Q&A Bot: {str(e)}")
            return False
    
    def ask_question(
        self,
        question: str,
        use_agent: bool = True,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Ask a healthcare question."""
        
        if not self.is_initialized:
            return {
                "error": "Bot not initialized. Please call initialize() first.",
                "response": "System not ready. Please try again later."
            }
        
        try:
            # Validate question
            validation_result = validate_medical_query(question)
            
            if not validation_result.get("is_valid", True):
                return {
                    "response": format_validation_response(validation_result),
                    "validation": validation_result,
                    "source": "validation"
                }
            
            # Choose processing method
            if use_agent:
                # Use medical agent for more comprehensive handling
                result = self.medical_agent.process_query(question)
                
                return {
                    "response": result["response"],
                    "tool_calls": result.get("tool_calls", []),
                    "safety_status": result.get("safety_status", "unknown"),
                    "token_usage": result.get("token_usage", {}),
                    "validation": validation_result,
                    "source": "medical_agent"
                }
            else:
                # Use Q&A chain directly
                result = self.qa_system["qa_chain"].answer_question(
                    question,
                    include_sources=include_sources
                )
                
                return {
                    "response": result["answer"],
                    "sources": result.get("sources", []) if include_sources else [],
                    "safety_classification": result.get("safety_classification", "unknown"),
                    "follow_up_questions": result.get("follow_up_questions", []),
                    "token_usage": result.get("token_usage", {}),
                    "validation": validation_result,
                    "source": "qa_chain"
                }
                
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "error": str(e),
                "response": "I apologize, but I encountered an error while processing your question. Please try again.",
                "source": "error_handler"
            }
    
    def explain_term(self, term: str) -> Dict[str, Any]:
        """Explain a medical term."""
        
        if not self.is_initialized:
            return {
                "error": "Bot not initialized",
                "explanation": "System not ready"
            }
        
        try:
            result = self.qa_system["terminology_chain"].explain_term(term)
            return result
            
        except Exception as e:
            logger.error(f"Error explaining term '{term}': {str(e)}")
            return {
                "term": term,
                "explanation": f"Could not explain term '{term}'. Please try again or consult a medical dictionary.",
                "error": str(e)
            }
    
    def get_conversation_summary(self) -> str:
        """Get conversation summary."""
        
        if not self.is_initialized:
            return "System not initialized"
        
        try:
            return self.qa_system["conversation_chain"].get_conversation_summary()
        except Exception as e:
            logger.error(f"Error getting conversation summary: {str(e)}")
            return "Could not retrieve conversation summary"
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        
        if self.is_initialized and self.qa_system:
            try:
                self.qa_system["conversation_chain"].clear_history()
                logger.info("Conversation history cleared")
            except Exception as e:
                logger.error(f"Error clearing conversation: {str(e)}")
    
    def add_documents(self, document_paths: list) -> Dict[str, Any]:
        """Add new documents to the system."""
        
        if not self.is_initialized:
            return {
                "success": False,
                "message": "Bot not initialized"
            }
        
        try:
            from document_processing.document_loader import HealthcareDocumentLoader
            from document_processing.vectorstore import update_vectorstore_with_new_documents
            
            loader = HealthcareDocumentLoader()
            new_documents = []
            
            for path in document_paths:
                try:
                    docs = loader.load_document(path)
                    new_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load document {path}: {str(e)}")
                    continue
            
            if new_documents:
                # Process new documents
                text_splitter = create_medical_text_splitter()
                split_docs = text_splitter.split_documents(new_documents)
                optimized_chunks = optimize_chunks_for_retrieval(split_docs)
                
                # Update vector store
                update_vectorstore_with_new_documents(self.vectorstore, optimized_chunks)
                
                logger.info(f"Added {len(new_documents)} new documents ({len(optimized_chunks)} chunks)")
                
                return {
                    "success": True,
                    "message": f"Successfully added {len(new_documents)} documents",
                    "documents_added": len(new_documents),
                    "chunks_added": len(optimized_chunks)
                }
            else:
                return {
                    "success": False,
                    "message": "No valid documents found to add"
                }
                
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return {
                "success": False,
                "message": f"Error adding documents: {str(e)}"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        
        status = {
            "is_initialized": self.is_initialized,
            "version": settings.app_version,
            "config": {
                "vector_store_type": settings.vector_store_type,
                "embedding_model": settings.embedding_model,
                "openai_model": settings.openai_model,
                "chunk_size": settings.embedding_chunk_size
            }
        }
        
        if self.is_initialized:
            try:
                # Vector store stats
                status["vectorstore_stats"] = self.vectorstore.get_collection_stats()
                
                # Available tools
                status["available_tools"] = self.medical_agent.get_available_tools()
                
            except Exception as e:
                status["status_error"] = str(e)
        
        return status


# Global bot instance
healthcare_bot = None


async def initialize_bot(force_rebuild: bool = False) -> HealthcareQABot:
    """Initialize the global bot instance."""
    global healthcare_bot
    
    if healthcare_bot is None:
        healthcare_bot = HealthcareQABot()
    
    success = await healthcare_bot.initialize(force_rebuild=force_rebuild)
    
    if not success:
        raise RuntimeError("Failed to initialize Healthcare Q&A Bot")
    
    return healthcare_bot


def get_bot() -> Optional[HealthcareQABot]:
    """Get the global bot instance."""
    return healthcare_bot


# CLI Interface
def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Healthcare Q&A Bot")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vector store")
    parser.add_argument("--question", type=str, help="Ask a single question")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    async def run_cli():
        try:
            # Initialize bot
            bot = await initialize_bot(force_rebuild=args.rebuild)
            
            if args.question:
                # Single question mode
                result = bot.ask_question(args.question)
                print(f"\nQuestion: {args.question}")
                print(f"Answer: {result['response']}")
                
                if result.get('sources'):
                    print(f"\nSources: {len(result['sources'])} documents referenced")
                
            elif args.interactive:
                # Interactive mode
                print("\n Healthcare Q&A Bot - Interactive Mode")
                print("Type 'quit' to exit, 'help' for commands\n")
                
                while True:
                    try:
                        question = input("Ask a health question: ").strip()
                        
                        if question.lower() in ['quit', 'exit', 'q']:
                            break
                        
                        if question.lower() == 'help':
                            print("\nCommands:")
                            print("- Ask any health-related question")
                            print("- 'status' - Show system status")
                            print("- 'clear' - Clear conversation history")
                            print("- 'quit' - Exit")
                            continue
                        
                        if question.lower() == 'status':
                            status = bot.get_system_status()
                            print(f"\nSystem Status: {'✓ Ready' if status['is_initialized'] else '✗ Not Ready'}")
                            print(f"Documents: {status.get('vectorstore_stats', {}).get('total_documents', 'Unknown')}")
                            continue
                        
                        if question.lower() == 'clear':
                            bot.clear_conversation()
                            print("Conversation history cleared.")
                            continue
                        
                        if not question:
                            continue
                        
                        # Process question
                        result = bot.ask_question(question)
                        print(f"\n {result['response']}")
                        
                        # Show follow-up questions if available
                        if result.get('follow_up_questions'):
                            print(f"\n Related questions you might ask:")
                            for fq in result['follow_up_questions']:
                                print(f"   • {fq}")
                        
                        print()  # Empty line for readability
                        
                    except KeyboardInterrupt:
                        print("\nGoodbye!")
                        break
                    except Exception as e:
                        print(f"Error: {str(e)}")
            
            else:
                # Just show status
                status = bot.get_system_status()
                print(f"\n Healthcare Q&A Bot Status")
                print(f"Initialized: {'✓' if status['is_initialized'] else '✗'}")
                print(f"Version: {status['version']}")
                print(f"Documents: {status.get('vectorstore_stats', {}).get('total_documents', 'Unknown')}")
                print("\nUse --question 'your question' or --interactive to start")
                
        except Exception as e:
            logger.error(f"CLI error: {str(e)}")
            print(f"Error: {str(e)}")
    
    # Run the async function
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
