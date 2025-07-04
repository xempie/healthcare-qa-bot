"""
Text splitting utilities for healthcare documents.
"""

import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.schema import Document
from loguru import logger

from config.settings import settings


class MedicalTextSplitter:
    """Text splitter optimized for medical documents."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        length_function: callable = len,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size or settings.embedding_chunk_size
        self.chunk_overlap = chunk_overlap or settings.embedding_chunk_overlap
        self.length_function = length_function
        
        # Medical document specific separators
        self.medical_separators = separators or [
            "\n\n## ",  # Section headers
            "\n\n# ",   # Main headers  
            "\n\n### ", # Subsection headers
            "\n\n",     # Paragraph breaks
            "\n\n- ",   # List items
            "\n\n* ",   # Bullet points
            "\n\n1. ",  # Numbered lists
            "\n\nSymptoms:",
            "\n\nTreatment:",
            "\n\nDiagnosis:",
            "\n\nCauses:",
            "\n\nPrevention:",
            "\n\nQ:",   # Questions
            "\n\nA:",   # Answers
            "\n\nQuestion:",
            "\n\nAnswer:",
            "\n\n",
            "\n",
            " ",
            ""
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separators=self.medical_separators
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks optimized for medical content."""
        split_docs = []
        
        for doc in documents:
            # Pre-process the document
            processed_content = self._preprocess_medical_content(doc.page_content)
            
            # Create a new document with processed content
            processed_doc = Document(
                page_content=processed_content,
                metadata=doc.metadata.copy()
            )
            
            # Split the document
            chunks = self.splitter.split_documents([processed_doc])
            
            # Post-process chunks
            for i, chunk in enumerate(chunks):
                # Add chunk metadata
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk.page_content),
                    'parent_document': doc.metadata.get('source', 'unknown')
                })
                
                # Enhance chunk with medical context
                enhanced_chunk = self._enhance_medical_chunk(chunk)
                split_docs.append(enhanced_chunk)
        
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs
    
    def _preprocess_medical_content(self, content: str) -> str:
        """Preprocess medical content for better splitting."""
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Ensure proper spacing around medical terms
        medical_patterns = [
            (r'(\w+)(mg|mcg|ml|cc|kg|lb|oz)', r'\1 \2'),  # Units
            (r'(\d+)([a-zA-Z])', r'\1 \2'),  # Numbers and letters
            (r'([a-zA-Z])(\d+)', r'\1 \2'),  # Letters and numbers
        ]
        
        for pattern, replacement in medical_patterns:
            content = re.sub(pattern, replacement, content)
        
        # Ensure proper paragraph breaks
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Standardize medical section headers
        section_patterns = [
            (r'(?i)\bsymptoms?\b:', '\n\nSymptoms:'),
            (r'(?i)\btreatments?\b:', '\n\nTreatment:'),
            (r'(?i)\bdiagnosis\b:', '\n\nDiagnosis:'),
            (r'(?i)\bcauses?\b:', '\n\nCauses:'),
            (r'(?i)\bprevention\b:', '\n\nPrevention:'),
            (r'(?i)\bside effects?\b:', '\n\nSide Effects:'),
            (r'(?i)\bwarnings?\b:', '\n\nWarnings:'),
            (r'(?i)\bdosage\b:', '\n\nDosage:'),
        ]
        
        for pattern, replacement in section_patterns:
            content = re.sub(pattern, replacement, content)
        
        return content.strip()
    
    def _enhance_medical_chunk(self, chunk: Document) -> Document:
        """Enhance chunk with medical-specific metadata."""
        content = chunk.page_content
        
        # Identify medical entities and concepts
        medical_entities = self._extract_medical_entities(content)
        
        # Identify content type
        content_type = self._identify_content_type(content)
        
        # Add enhanced metadata
        chunk.metadata.update({
            'medical_entities': medical_entities,
            'content_type': content_type,
            'has_symptoms': 'symptoms' in content.lower(),
            'has_treatment': 'treatment' in content.lower(),
            'has_medication': any(term in content.lower() for term in ['medication', 'drug', 'prescription', 'dosage']),
            'has_diagnosis': 'diagnosis' in content.lower(),
            'has_emergency': any(term in content.lower() for term in ['emergency', 'urgent', 'immediate', 'call 911']),
            'medical_specialty': self._identify_medical_specialty(content)
        })
        
        return chunk
    
    def _extract_medical_entities(self, content: str) -> List[str]:
        """Extract medical entities from content."""
        entities = []
        
        # Common medical terms and patterns
        medical_patterns = [
            r'\b\w+itis\b',  # Conditions ending in -itis
            r'\b\w+osis\b',  # Conditions ending in -osis
            r'\b\w+emia\b',  # Blood conditions
            r'\b\w+pathy\b', # Disease conditions
            r'\b\w+mg\b',    # Dosages
            r'\b\w+mcg\b',   # Dosages
            r'\b\w+ml\b',    # Volumes
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities.extend(matches)
        
        # Common medical terms
        medical_terms = [
            'hypertension', 'diabetes', 'asthma', 'pneumonia', 'bronchitis',
            'arthritis', 'infection', 'inflammation', 'fever', 'pain',
            'headache', 'nausea', 'vomiting', 'diarrhea', 'constipation',
            'anxiety', 'depression', 'insomnia', 'fatigue', 'dizziness'
        ]
        
        content_lower = content.lower()
        for term in medical_terms:
            if term in content_lower:
                entities.append(term)
        
        return list(set(entities))
    
    def _identify_content_type(self, content: str) -> str:
        """Identify the type of medical content."""
        content_lower = content.lower()
        
        if 'question:' in content_lower and 'answer:' in content_lower:
            return 'faq'
        elif any(term in content_lower for term in ['guideline', 'protocol', 'procedure']):
            return 'clinical_guideline'
        elif any(term in content_lower for term in ['patient', 'education', 'information']):
            return 'patient_education'
        elif any(term in content_lower for term in ['symptom', 'diagnosis', 'treatment']):
            return 'clinical_information'
        elif any(term in content_lower for term in ['medication', 'drug', 'prescription']):
            return 'medication_information'
        elif any(term in content_lower for term in ['emergency', 'urgent', 'immediate']):
            return 'emergency_information'
        else:
            return 'general_medical'
    
    def _identify_medical_specialty(self, content: str) -> str:
        """Identify medical specialty based on content."""
        content_lower = content.lower()
        
        specialties = {
            'cardiology': ['heart', 'cardiac', 'cardiovascular', 'blood pressure', 'hypertension'],
            'dermatology': ['skin', 'rash', 'dermatitis', 'acne', 'eczema'],
            'endocrinology': ['diabetes', 'thyroid', 'hormone', 'insulin', 'glucose'],
            'gastroenterology': ['stomach', 'digestive', 'intestinal', 'bowel', 'liver'],
            'neurology': ['brain', 'neurological', 'seizure', 'stroke', 'migraine'],
            'orthopedics': ['bone', 'joint', 'fracture', 'arthritis', 'muscle'],
            'pediatrics': ['child', 'infant', 'pediatric', 'baby', 'adolescent'],
            'psychiatry': ['mental', 'depression', 'anxiety', 'psychiatric', 'therapy'],
            'pulmonology': ['lung', 'respiratory', 'breathing', 'asthma', 'pneumonia'],
            'urology': ['kidney', 'bladder', 'urinary', 'prostate', 'urine']
        }
        
        for specialty, keywords in specialties.items():
            if any(keyword in content_lower for keyword in keywords):
                return specialty
        
        return 'general_medicine'


class SemanticMedicalSplitter(TextSplitter):
    """Semantic text splitter for medical documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text based on medical semantic boundaries."""
        # Split by medical sections first
        sections = self._split_by_medical_sections(text)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) <= self.chunk_size:
                current_chunk += section + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If section is too long, split it further
                if len(section) > self.chunk_size:
                    sub_chunks = self._split_long_section(section)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = section + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_medical_sections(self, text: str) -> List[str]:
        """Split text by medical section boundaries."""
        # Define medical section patterns
        section_patterns = [
            r'\n\n(?=Symptoms?:)',
            r'\n\n(?=Treatment:)',
            r'\n\n(?=Diagnosis:)',
            r'\n\n(?=Causes?:)',
            r'\n\n(?=Prevention:)',
            r'\n\n(?=Side Effects?:)',
            r'\n\n(?=Warnings?:)',
            r'\n\n(?=Dosage:)',
            r'\n\n(?=Question:)',
            r'\n\n(?=Answer:)',
            r'\n\n(?=Q:)',
            r'\n\n(?=A:)',
        ]
        
        # Split by patterns
        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                new_sections.extend(re.split(pattern, section))
            sections = new_sections
        
        return [section.strip() for section in sections if section.strip()]
    
    def _split_long_section(self, section: str) -> List[str]:
        """Split long sections into smaller chunks."""
        # Use sentence boundaries for splitting
        sentences = re.split(r'(?<=[.!?])\s+', section)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


def create_medical_text_splitter(
    chunk_size: int = None,
    chunk_overlap: int = None,
    splitter_type: str = "recursive"
) -> TextSplitter:
    """Create a medical text splitter based on type."""
    
    chunk_size = chunk_size or settings.embedding_chunk_size
    chunk_overlap = chunk_overlap or settings.embedding_chunk_overlap
    
    if splitter_type == "recursive":
        return MedicalTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == "semantic":
        return SemanticMedicalSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")


def optimize_chunks_for_retrieval(chunks: List[Document]) -> List[Document]:
    """Optimize chunks for better retrieval performance."""
    optimized_chunks = []
    
    for chunk in chunks:
        # Add context from metadata
        content = chunk.page_content
        
        # Add document type context
        doc_type = chunk.metadata.get('document_type', '')
        if doc_type:
            content = f"[{doc_type.upper()}] {content}"
        
        # Add medical specialty context
        specialty = chunk.metadata.get('medical_specialty', '')
        if specialty and specialty != 'general_medicine':
            content = f"[{specialty.upper()}] {content}"
        
        # Add content type context
        content_type = chunk.metadata.get('content_type', '')
        if content_type:
            content = f"[{content_type.upper()}] {content}"
        
        # Create optimized chunk
        optimized_chunk = Document(
            page_content=content,
            metadata=chunk.metadata.copy()
        )
        
        optimized_chunks.append(optimized_chunk)
    
    logger.info(f"Optimized {len(chunks)} chunks for retrieval")
    return optimized_chunks
