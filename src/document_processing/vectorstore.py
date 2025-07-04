"""
Vector store management for healthcare documents.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger

from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

from config.settings import settings


class HealthcareVectorStore:
    """Vector store manager for healthcare documents."""
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.collection_name
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        self.vectorstore = None
        self.vectorstore_path = Path(settings.vector_store_path)
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        
    def create_vectorstore(self, documents: List[Document]) -> VectorStore:
        """Create a new vector store from documents."""
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        try:
            if settings.vector_store_type.lower() == "chroma":
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    persist_directory=str(self.vectorstore_path / "chroma")
                )
                # Persist the collection
                self.vectorstore.persist()
                
            elif settings.vector_store_type.lower() == "faiss":
                self.vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                # Save FAISS index
                faiss_path = self.vectorstore_path / "faiss"
                faiss_path.mkdir(exist_ok=True)
                self.vectorstore.save_local(str(faiss_path))
                
            else:
                raise ValueError(f"Unsupported vector store type: {settings.vector_store_type}")
            
            logger.info(f"Successfully created {settings.vector_store_type} vector store")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vectorstore(self) -> Optional[VectorStore]:
        """Load existing vector store."""
        try:
            if settings.vector_store_type.lower() == "chroma":
                chroma_path = self.vectorstore_path / "chroma"
                if chroma_path.exists():
                    self.vectorstore = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=str(chroma_path)
                    )
                    logger.info("Loaded existing Chroma vector store")
                    return self.vectorstore
                    
            elif settings.vector_store_type.lower() == "faiss":
                faiss_path = self.vectorstore_path / "faiss"
                if faiss_path.exists() and (faiss_path / "index.faiss").exists():
                    self.vectorstore = FAISS.load_local(
                        str(faiss_path),
                        self.embeddings
                    )
                    logger.info("Loaded existing FAISS vector store")
                    return self.vectorstore
            
            logger.info("No existing vector store found")
            return None
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing vector store."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        if not documents:
            logger.warning("No documents to add")
            return
        
        try:
            self.vectorstore.add_documents(documents)
            
            # Persist changes
            if settings.vector_store_type.lower() == "chroma":
                self.vectorstore.persist()
            elif settings.vector_store_type.lower() == "faiss":
                faiss_path = self.vectorstore_path / "faiss"
                self.vectorstore.save_local(str(faiss_path))
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform similarity search."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            if filter_dict:
                # Apply metadata filters
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(query=query, k=k)
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with relevance scores."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query=query, k=k)
            
            logger.info(f"Found {len(results)} scored documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}")
            raise
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 10,
        lambda_mult: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform MMR search for diverse results."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            if hasattr(self.vectorstore, 'max_marginal_relevance_search'):
                results = self.vectorstore.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filter=filter_dict
                )
                logger.info(f"MMR search found {len(results)} diverse documents")
                return results
            else:
                # Fallback to regular similarity search
                return self.similarity_search(query, k, filter_dict)
                
        except Exception as e:
            logger.error(f"Error performing MMR search: {str(e)}")
            raise
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from vector store."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            if hasattr(self.vectorstore, 'delete'):
                self.vectorstore.delete(document_ids)
                logger.info(f"Deleted {len(document_ids)} documents")
            else:
                logger.warning("Vector store does not support document deletion")
                
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        try:
            stats = {
                "vector_store_type": settings.vector_store_type,
                "collection_name": self.collection_name,
                "embedding_model": settings.embedding_model,
            }
            
            if settings.vector_store_type.lower() == "chroma":
                # Get Chroma collection stats
                collection = self.vectorstore._collection
                stats.update({
                    "total_documents": collection.count(),
                    "storage_path": str(self.vectorstore_path / "chroma")
                })
                
            elif settings.vector_store_type.lower() == "faiss":
                # Get FAISS index stats
                if hasattr(self.vectorstore, 'index'):
                    stats.update({
                        "total_documents": self.vectorstore.index.ntotal,
                        "storage_path": str(self.vectorstore_path / "faiss")
                    })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}


class MedicalRetriever:
    """Enhanced retriever for medical documents."""
    
    def __init__(self, vectorstore: HealthcareVectorStore):
        self.vectorstore = vectorstore
        
    def retrieve_medical_info(
        self,
        query: str,
        k: int = 5,
        content_type: Optional[str] = None,
        medical_specialty: Optional[str] = None,
        include_scores: bool = False
    ) -> List[Document]:
        """Retrieve medical information with enhanced filtering."""
        
        # Build filter dictionary
        filter_dict = {}
        
        if content_type:
            filter_dict['content_type'] = content_type
            
        if medical_specialty:
            filter_dict['medical_specialty'] = medical_specialty
        
        # Perform search
        if include_scores:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter_dict=filter_dict if filter_dict else None
            )
            return results
        else:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter_dict=filter_dict if filter_dict else None
            )
            return results
    
    def retrieve_faq_answers(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve FAQ answers specifically."""
        return self.retrieve_medical_info(
            query=query,
            k=k,
            content_type="faq"
        )
    
    def retrieve_emergency_info(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve emergency medical information."""
        return self.retrieve_medical_info(
            query=query,
            k=k,
            content_type="emergency_information"
        )
    
    def retrieve_by_specialty(
        self,
        query: str,
        specialty: str,
        k: int = 5
    ) -> List[Document]:
        """Retrieve information by medical specialty."""
        return self.retrieve_medical_info(
            query=query,
            k=k,
            medical_specialty=specialty
        )
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.7
    ) -> List[Document]:
        """Perform hybrid search combining similarity and MMR."""
        # Get similarity results
        similarity_results = self.vectorstore.similarity_search(query, k)
        
        # Get MMR results for diversity
        mmr_results = self.vectorstore.max_marginal_relevance_search(
            query, k, fetch_k=k*2
        )
        
        # Combine results with weighted scoring
        combined_results = []
        seen_content = set()
        
        # Add similarity results with higher weight
        for doc in similarity_results:
            if doc.page_content not in seen_content:
                combined_results.append(doc)
                seen_content.add(doc.page_content)
        
        # Add diverse MMR results
        for doc in mmr_results:
            if doc.page_content not in seen_content and len(combined_results) < k:
                combined_results.append(doc)
                seen_content.add(doc.page_content)
        
        return combined_results[:k]


def setup_vectorstore(documents: List[Document] = None) -> HealthcareVectorStore:
    """Setup vector store with documents."""
    vectorstore = HealthcareVectorStore()
    
    # Try to load existing vector store
    existing_store = vectorstore.load_vectorstore()
    
    if existing_store:
        logger.info("Using existing vector store")
        return vectorstore
    
    # Create new vector store if documents provided
    if documents:
        vectorstore.create_vectorstore(documents)
        logger.info("Created new vector store")
        return vectorstore
    
    # No existing store and no documents
    logger.warning("No vector store found and no documents provided")
    return vectorstore


def update_vectorstore_with_new_documents(
    vectorstore: HealthcareVectorStore,
    new_documents: List[Document]
) -> None:
    """Update vector store with new documents."""
    if not new_documents:
        logger.info("No new documents to add")
        return
    
    try:
        vectorstore.add_documents(new_documents)
        logger.info(f"Successfully added {len(new_documents)} new documents to vector store")
    except Exception as e:
        logger.error(f"Failed to update vector store: {str(e)}")
        raise
