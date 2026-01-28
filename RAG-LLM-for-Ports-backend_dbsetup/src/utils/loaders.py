# src/utils/loaders.py
"""
Document Loaders and Chunkers for AI Port Decision-Support System
Handles PDF processing with multilingual support and intelligent chunking
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import hashlib
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.schema import Document

# Multilingual support
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer
import langdetect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualDocumentLoader:
    """
    Advanced document loader with multilingual support for port operations.
    Handles PDF processing, language detection, and translation.
    """
    
    def __init__(self, 
                 supported_languages: List[str] = None,
                 translation_model_name: str = "Helsinki-NLP/opus-mt-en-mul",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the multilingual document loader.
        
        Args:
            supported_languages: List of supported languages
            translation_model_name: Name of the translation model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.supported_languages = supported_languages or ["en", "es", "fr", "de", "zh", "ja", "ko"]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize translation model
        try:
            self.translation_model = MarianMTModel.from_pretrained(translation_model_name)
            self.translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
            logger.info(f"Translation model loaded: {translation_model_name}")
        except Exception as e:
            logger.warning(f"Could not load translation model: {e}")
            self.translation_model = None
            self.translation_tokenizer = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info("Multilingual document loader initialized")
    
    def load_pdf_documents(self, 
                          pdf_directory: str,
                          recursive: bool = True,
                          language_detection: bool = True,
                          auto_translate: bool = False) -> List[Document]:
        """
        Load PDF documents from a directory with multilingual support.
        
        Args:
            pdf_directory: Directory containing PDF files
            recursive: Whether to search recursively
            language_detection: Whether to detect document language
            auto_translate: Whether to automatically translate non-English documents
            
        Returns:
            List of processed documents
        """
        try:
            logger.info(f"Loading PDF documents from: {pdf_directory}")
            
            # Load PDF documents
            loader = DirectoryLoader(
                pdf_directory,
                glob="**/*.pdf" if recursive else "*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} PDF documents")
            
            # Process documents with multilingual support
            processed_documents = []
            for i, doc in enumerate(documents):
                try:
                    processed_doc = self._process_document(doc, language_detection, auto_translate)
                    processed_documents.append(processed_doc)
                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
                    # Add original document with error metadata
                    doc.metadata["processing_error"] = str(e)
                    processed_documents.append(doc)
            
            logger.info(f"Successfully processed {len(processed_documents)} documents")
            return processed_documents
            
        except Exception as e:
            logger.error(f"Error loading PDF documents: {e}")
            return []
    
    def _process_document(self, 
                         document: Document, 
                         language_detection: bool,
                         auto_translate: bool) -> Document:
        """Process a single document with multilingual support"""
        
        # Extract text content
        content = document.page_content
        
        # Detect language if enabled
        detected_language = "en"  # Default to English
        if language_detection:
            try:
                detected_language = langdetect.detect(content)
                logger.info(f"Detected language: {detected_language}")
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
        
        # Translate if needed and enabled
        translated_content = content
        if auto_translate and detected_language != "en" and self.translation_model:
            try:
                translated_content = self._translate_text(content, detected_language, "en")
                logger.info(f"Translated document from {detected_language} to English")
            except Exception as e:
                logger.warning(f"Translation failed: {e}")
        
        # Add metadata
        document.metadata.update({
            "detected_language": detected_language,
            "is_translated": translated_content != content,
            "original_language": detected_language,
            "processed_at": datetime.now().isoformat(),
            "document_hash": self._calculate_document_hash(content)
        })
        
        # Update content if translated
        if translated_content != content:
            document.page_content = translated_content
        
        return document
    
    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using MarianMT model"""
        if not self.translation_model or not self.translation_tokenizer:
            return text
        
        try:
            # Prepare text for translation
            text_to_translate = f">>{target_lang}<< {text}"
            
            # Tokenize and translate
            inputs = self.translation_tokenizer(text_to_translate, return_tensors="pt", truncation=True, max_length=512)
            translated = self.translation_model.generate(**inputs)
            translated_text = self.translation_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def _calculate_document_hash(self, content: str) -> str:
        """Calculate hash for document content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for processing.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks
        """
        try:
            logger.info(f"Chunking {len(documents)} documents")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"chunk_{i}",
                    "chunk_size": len(chunk.page_content),
                    "chunked_at": datetime.now().isoformat()
                })
                processed_chunks.append(chunk)
            
            logger.info(f"Created {len(processed_chunks)} document chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            return documents  # Return original documents if chunking fails


class PortDocumentProcessor:
    """
    Specialized document processor for port operations documents.
    Handles different document types and adds port-specific metadata.
    """
    
    def __init__(self):
        """Initialize the port document processor"""
        self.document_type_patterns = self._initialize_document_patterns()
        self.metadata_extractors = self._initialize_metadata_extractors()
        
        logger.info("Port document processor initialized")
    
    def _initialize_document_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for document type classification"""
        return {
            "sop": ["standard operating procedure", "sop", "procedure", "protocol"],
            "tos_logs": ["terminal operating system", "tos", "log", "system log"],
            "edi_messages": ["edi", "electronic data interchange", "message", "manifest"],
            "ais_data": ["ais", "automatic identification", "vessel tracking", "position"],
            "weather_reports": ["weather", "forecast", "meteorological", "climate"],
            "berth_schedules": ["berth", "schedule", "allocation", "dock", "pier"],
            "vessel_manifests": ["manifest", "cargo", "vessel", "ship", "loading"],
            "safety_protocols": ["safety", "emergency", "hazard", "risk", "incident"],
            "regulatory_docs": ["regulation", "compliance", "legal", "authority"],
            "maintenance_logs": ["maintenance", "repair", "equipment", "service"]
        }
    
    def _initialize_metadata_extractors(self) -> Dict[str, callable]:
        """Initialize metadata extraction functions"""
        return {
            "extract_vessel_info": self._extract_vessel_info,
            "extract_timestamps": self._extract_timestamps,
            "extract_port_info": self._extract_port_info,
            "extract_safety_info": self._extract_safety_info
        }
    
    def classify_document_type(self, document: Document) -> str:
        """Classify document type based on content and metadata"""
        try:
            content_lower = document.page_content.lower()
            source_lower = document.metadata.get("source", "").lower()
            
            # Check content patterns
            for doc_type, patterns in self.document_type_patterns.items():
                if any(pattern in content_lower for pattern in patterns):
                    return doc_type
            
            # Check source file patterns
            for doc_type, patterns in self.document_type_patterns.items():
                if any(pattern in source_lower for pattern in patterns):
                    return doc_type
            
            return "general"
            
        except Exception as e:
            logger.warning(f"Error classifying document type: {e}")
            return "general"
    
    def extract_port_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract port-specific metadata from document"""
        try:
            metadata = {}
            
            # Extract various types of metadata
            for extractor_name, extractor_func in self.metadata_extractors.items():
                try:
                    extracted_data = extractor_func(document)
                    metadata.update(extracted_data)
                except Exception as e:
                    logger.warning(f"Error in {extractor_name}: {e}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting port metadata: {e}")
            return {}
    
    def _extract_vessel_info(self, document: Document) -> Dict[str, Any]:
        """Extract vessel-related information"""
        import re
        
        content = document.page_content
        vessel_info = {}
        
        # Extract IMO numbers
        imo_pattern = r'IMO\s*:?\s*(\d{7})'
        imo_matches = re.findall(imo_pattern, content, re.IGNORECASE)
        if imo_matches:
            vessel_info["imo_numbers"] = imo_matches
        
        # Extract vessel names
        vessel_name_pattern = r'Vessel\s*:?\s*([A-Za-z\s]+)'
        vessel_matches = re.findall(vessel_name_pattern, content, re.IGNORECASE)
        if vessel_matches:
            vessel_info["vessel_names"] = vessel_matches
        
        return vessel_info
    
    def _extract_timestamps(self, document: Document) -> Dict[str, Any]:
        """Extract timestamp information"""
        import re
        
        content = document.page_content
        timestamp_info = {}
        
        # Extract various timestamp patterns
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
            r'(\d{2}:\d{2}:\d{2})',  # HH:MM:SS
        ]
        
        timestamps = []
        for pattern in timestamp_patterns:
            matches = re.findall(pattern, content)
            timestamps.extend(matches)
        
        if timestamps:
            timestamp_info["timestamps"] = timestamps
        
        return timestamp_info
    
    def _extract_port_info(self, document: Document) -> Dict[str, Any]:
        """Extract port-related information"""
        import re
        
        content = document.page_content
        port_info = {}
        
        # Extract port names
        port_pattern = r'Port\s*:?\s*([A-Za-z\s]+)'
        port_matches = re.findall(port_pattern, content, re.IGNORECASE)
        if port_matches:
            port_info["port_names"] = port_matches
        
        # Extract berth information
        berth_pattern = r'Berth\s*:?\s*([A-Za-z0-9\s]+)'
        berth_matches = re.findall(berth_pattern, content, re.IGNORECASE)
        if berth_matches:
            port_info["berth_info"] = berth_matches
        
        return port_info
    
    def _extract_safety_info(self, document: Document) -> Dict[str, Any]:
        """Extract safety-related information"""
        import re
        
        content = document.page_content.lower()
        safety_info = {}
        
        # Safety keywords
        safety_keywords = ["emergency", "hazard", "danger", "caution", "warning", "incident"]
        found_keywords = [keyword for keyword in safety_keywords if keyword in content]
        
        if found_keywords:
            safety_info["safety_keywords"] = found_keywords
        
        # Emergency contact patterns
        emergency_pattern = r'emergency\s*:?\s*(\d{3}[-.]?\d{3}[-.]?\d{4})'
        emergency_matches = re.findall(emergency_pattern, content, re.IGNORECASE)
        if emergency_matches:
            safety_info["emergency_contacts"] = emergency_matches
        
        return safety_info
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents with port-specific metadata extraction.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed documents with enhanced metadata
        """
        try:
            logger.info(f"Processing {len(documents)} documents with port metadata")
            
            processed_documents = []
            for i, doc in enumerate(documents):
                try:
                    # Classify document type
                    doc_type = self.classify_document_type(doc)
                    
                    # Extract port metadata
                    port_metadata = self.extract_port_metadata(doc)
                    
                    # Update document metadata
                    doc.metadata.update({
                        "document_type": doc_type,
                        "port_metadata": port_metadata,
                        "processed_by": "PortDocumentProcessor",
                        "processed_at": datetime.now().isoformat()
                    })
                    
                    processed_documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
                    # Add error metadata
                    doc.metadata["processing_error"] = str(e)
                    processed_documents.append(doc)
            
            logger.info(f"Successfully processed {len(processed_documents)} documents")
            return processed_documents
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return documents


def create_document_loader(chunk_size: int = 1000, 
                          chunk_overlap: int = 200,
                          multilingual: bool = True) -> Union[MultilingualDocumentLoader, RecursiveCharacterTextSplitter]:
    """
    Factory function to create appropriate document loader.
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        multilingual: Whether to enable multilingual support
        
    Returns:
        Configured document loader
    """
    if multilingual:
        return MultilingualDocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )


def load_and_process_documents(pdf_directory: str,
                              chunk_size: int = 1000,
                              chunk_overlap: int = 200,
                              multilingual: bool = True,
                              port_processing: bool = True) -> List[Document]:
    """
    Complete document loading and processing pipeline.
    
    Args:
        pdf_directory: Directory containing PDF files
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        multilingual: Whether to enable multilingual support
        port_processing: Whether to apply port-specific processing
        
    Returns:
        List of processed document chunks
    """
    try:
        logger.info("Starting complete document processing pipeline")
        
        # Create document loader
        loader = create_document_loader(chunk_size, chunk_overlap, multilingual)
        
        # Load documents
        if isinstance(loader, MultilingualDocumentLoader):
            documents = loader.load_pdf_documents(pdf_directory)
        else:
            # Fallback to basic loading
            basic_loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents = basic_loader.load()
        
        # Process with port-specific metadata if enabled
        if port_processing:
            processor = PortDocumentProcessor()
            documents = processor.process_documents(documents)
        
        # Chunk documents
        if isinstance(loader, MultilingualDocumentLoader):
            chunks = loader.chunk_documents(documents)
        else:
            chunks = loader.split_documents(documents)
        
        logger.info(f"Document processing pipeline completed. Created {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in document processing pipeline: {e}")
        return []
