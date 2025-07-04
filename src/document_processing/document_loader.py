"""
Document loading utilities for the Healthcare Q&A Bot.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader

from config.settings import settings


class HealthcareDocumentLoader:
    """Document loader for healthcare-related documents."""
    
    def __init__(self):
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader
        }
        
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.stat().st_size > settings.max_document_size:
                raise ValueError(f"File size exceeds maximum allowed size: {settings.max_document_size}")
            
            extension = file_path.suffix.lower()
            
            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file format: {extension}")
            
            loader_class = self.supported_extensions[extension]
            loader = loader_class(str(file_path))
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': extension,
                    'file_size': file_path.stat().st_size,
                    'loaded_at': str(asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0)
                })
            
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_documents = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Loaded {len(all_documents)} documents from directory {directory_path}")
        return all_documents
    
    def validate_medical_content(self, documents: List[Document]) -> List[Document]:
        """Validate and filter medical content."""
        validated_documents = []
        
        medical_keywords = [
            'medical', 'health', 'patient', 'treatment', 'diagnosis',
            'symptoms', 'medication', 'therapy', 'clinical', 'healthcare',
            'disease', 'condition', 'doctor', 'physician', 'nurse'
        ]
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            
            # Check if document contains medical content
            if any(keyword in content_lower for keyword in medical_keywords):
                # Basic content validation
                if len(doc.page_content.strip()) > 50:  # Minimum content length
                    validated_documents.append(doc)
                else:
                    logger.warning(f"Document too short, skipping: {doc.metadata.get('source', 'unknown')}")
            else:
                logger.warning(f"Document doesn't appear to contain medical content: {doc.metadata.get('source', 'unknown')}")
        
        logger.info(f"Validated {len(validated_documents)} medical documents")
        return validated_documents


class MedicalDocumentProcessor:
    """Process medical documents with healthcare-specific handling."""
    
    def __init__(self):
        self.loader = HealthcareDocumentLoader()
        
    def process_medical_guidelines(self, file_path: str) -> List[Document]:
        """Process medical guideline documents."""
        documents = self.loader.load_document(file_path)
        
        # Add medical guideline metadata
        for doc in documents:
            doc.metadata.update({
                'document_type': 'medical_guideline',
                'authority_level': 'high',
                'content_category': 'clinical_guideline'
            })
        
        return documents
    
    def process_faq_documents(self, file_path: str) -> List[Document]:
        """Process FAQ documents."""
        documents = self.loader.load_document(file_path)
        
        # Split FAQ format if needed
        processed_docs = []
        
        for doc in documents:
            content = doc.page_content
            
            # Try to identify Q&A pairs
            if "Q:" in content and "A:" in content:
                qa_pairs = self._extract_qa_pairs(content)
                
                for i, (question, answer) in enumerate(qa_pairs):
                    new_doc = Document(
                        page_content=f"Question: {question}\nAnswer: {answer}",
                        metadata={
                            **doc.metadata,
                            'document_type': 'faq',
                            'qa_pair_index': i,
                            'question': question,
                            'answer': answer
                        }
                    )
                    processed_docs.append(new_doc)
            else:
                doc.metadata.update({
                    'document_type': 'faq',
                    'content_category': 'general_information'
                })
                processed_docs.append(doc)
        
        return processed_docs
    
    def _extract_qa_pairs(self, content: str) -> List[tuple]:
        """Extract Q&A pairs from formatted text."""
        qa_pairs = []
        lines = content.split('\n')
        
        current_question = None
        current_answer = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Q:') or line.startswith('Question:'):
                if current_question and current_answer:
                    qa_pairs.append((current_question, '\n'.join(current_answer)))
                
                current_question = line.replace('Q:', '').replace('Question:', '').strip()
                current_answer = []
                
            elif line.startswith('A:') or line.startswith('Answer:'):
                answer_text = line.replace('A:', '').replace('Answer:', '').strip()
                if answer_text:
                    current_answer.append(answer_text)
                    
            elif current_question and line:
                current_answer.append(line)
        
        # Add the last Q&A pair
        if current_question and current_answer:
            qa_pairs.append((current_question, '\n'.join(current_answer)))
        
        return qa_pairs
    
    def process_patient_information(self, file_path: str) -> List[Document]:
        """Process patient information documents."""
        documents = self.loader.load_document(file_path)
        
        for doc in documents:
            doc.metadata.update({
                'document_type': 'patient_information',
                'content_category': 'patient_education',
                'target_audience': 'patients'
            })
        
        return documents


async def load_all_documents(documents_dir: str = None) -> List[Document]:
    """Load all medical documents from the documents directory."""
    if documents_dir is None:
        documents_dir = settings.documents_dir
    
    processor = MedicalDocumentProcessor()
    all_documents = []
    
    documents_path = Path(documents_dir)
    
    if not documents_path.exists():
        logger.warning(f"Documents directory not found: {documents_dir}")
        return []
    
    # Load different types of documents
    for file_path in documents_path.iterdir():
        if file_path.is_file():
            try:
                if 'guideline' in file_path.name.lower():
                    docs = processor.process_medical_guidelines(str(file_path))
                elif 'faq' in file_path.name.lower():
                    docs = processor.process_faq_documents(str(file_path))
                elif 'patient' in file_path.name.lower():
                    docs = processor.process_patient_information(str(file_path))
                else:
                    docs = processor.loader.load_document(str(file_path))
                
                all_documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
    
    # Validate medical content
    validated_documents = processor.loader.validate_medical_content(all_documents)
    
    logger.info(f"Successfully loaded and validated {len(validated_documents)} medical documents")
    return validated_documents
