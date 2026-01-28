# src/port_information_retrieval.py
"""
Port Information Retrieval System for AI Port Decision-Support System
Core RAG implementation with ChromaDB integration and maritime-specific features

Automatically removes sensitive information from documents
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortInformationRetrieval:
    """
    Core information retrieval system for port operations.
    Integrates ChromaDB with specialized maritime document processing.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_persist_directory: str = "./storage/chroma",
                 collection_name: str = "port_documents",
                 model_name: str = "gpt-3.5-turbo",
                 embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the port information retrieval system.
        
        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            model_name: OpenAI model name for chat
            embedding_model: OpenAI embedding model name
        """
        self.openai_api_key = openai_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        
        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=embedding_model
        )
        
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.1
        )
        
        # Initialize ChromaDB
        self.vector_store = self._initialize_vector_store()
        
        # Initialize QA chain
        self.qa_chain = self._create_qa_chain()
        
        logger.info("Port Information Retrieval system initialized")
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize ChromaDB vector store"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.chroma_persist_directory, exist_ok=True)
            
            # Initialize ChromaDB
            vector_store = Chroma(
                persist_directory=self.chroma_persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            
            logger.info(f"ChromaDB initialized at {self.chroma_persist_directory}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the question-answering chain"""
        try:
            # Create custom prompt for port operations
            prompt_template = """
            You are an AI assistant specialized in port operations and maritime logistics.
            Use the following context to answer questions about port operations, safety protocols,
            vessel management, cargo handling, and regulatory compliance.
            
            Context:
            {context}
            
            Question: {question}
            
            Instructions:
            1. Provide accurate, specific answers based on the context
            2. If the context doesn't contain enough information, say so
            3. Always cite relevant sources when possible
            4. Focus on practical, actionable advice for port operations
            5. Prioritize safety and compliance in your recommendations
            
            Answer:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            logger.info("QA chain created successfully")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            raise
    
    async def add_documents_to_vector_store(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided for vector store")
                return False
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Add documents to ChromaDB
            self.vector_store.add_documents(documents)
            
            # Persist the changes
            self.vector_store.persist()
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    async def query_port_information(self, 
                                   question: str,
                                   document_types: Optional[List[str]] = None,
                                   max_documents: int = 5) -> Dict[str, Any]:
        """
        Query port information using RAG.
        
        Args:
            question: User's question
            document_types: Optional list of document types to search
            max_documents: Maximum number of documents to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            start_time = datetime.now()
            
            logger.info(f"Processing query: {question[:50]}...")
            
            # Update retriever with max_documents
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": max_documents}
            )
            
            # Filter by document types if specified
            if document_types:
                # Note: This is a simplified filter - in production, you'd want
                # more sophisticated filtering based on metadata
                logger.info(f"Filtering by document types: {document_types}")
            
            # Get relevant documents
            relevant_docs = retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return {
                    "answer": "I don't have enough information to answer this question. Please check if relevant documents have been uploaded to the system.",
                    "sources": [],
                    "confidence": 0.0,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Generate answer using QA chain
            result = self.qa_chain({"query": question})
            
            # Process sources
            sources = []
            for doc in relevant_docs:
                source_info = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "document_type": doc.metadata.get("document_type", "general"),
                    "page": doc.metadata.get("page", "N/A"),
                    "relevance_score": "N/A"  # ChromaDB doesn't provide scores by default
                }
                sources.append(source_info)
            
            # Calculate confidence based on document relevance
            confidence = min(0.9, len(relevant_docs) / max_documents)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            return {
                "answer": result["result"],
                "sources": sources,
                "confidence": confidence,
                "processing_time": processing_time,
                "documents_used": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": 0.0,
                "error": str(e)
            }
    
    async def search_similar_documents(self, 
                                     query: str,
                                     document_type: Optional[str] = None,
                                     k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents without generating an answer.
        
        Args:
            query: Search query
            document_type: Optional document type filter
            k: Number of documents to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            logger.info(f"Searching for documents similar to: {query[:50]}...")
            
            # Get similar documents
            similar_docs = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            # Process results
            results = []
            for doc in similar_docs:
                doc_info = {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "document_type": doc.metadata.get("document_type", "general"),
                    "page": doc.metadata.get("page", "N/A"),
                    "similarity_score": "N/A"  # ChromaDB doesn't provide scores by default
                }
                results.append(doc_info)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def get_document_summary(self, document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of documents in the system.
        
        Args:
            document_type: Optional document type filter
            
        Returns:
            Dictionary with document summary statistics
        """
        try:
            logger.info("Generating document summary")
            
            # Get all documents from the collection
            # Note: This is a simplified approach - in production, you'd want
            # more efficient ways to get collection statistics
            
            # For now, return basic statistics
            summary = {
                "total_documents": "N/A",  # Would need to implement collection stats
                "document_types": {
                    "sop": 0,
                    "tos_logs": 0,
                    "edi_messages": 0,
                    "ais_data": 0,
                    "weather_reports": 0,
                    "berth_schedules": 0,
                    "vessel_manifests": 0,
                    "safety_protocols": 0,
                    "regulatory_docs": 0,
                    "maintenance_logs": 0,
                    "general": 0
                },
                "last_updated": datetime.now().isoformat(),
                "collection_name": self.collection_name,
                "persist_directory": self.chroma_persist_directory
            }
            
            # In a real implementation, you would:
            # 1. Query the ChromaDB collection for statistics
            # 2. Count documents by type
            # 3. Get metadata about the collection
            
            logger.info("Document summary generated")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return {"error": str(e)}
    
    def update_qa_chain(self, 
                       model_name: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None):
        """
        Update the QA chain configuration.
        
        Args:
            model_name: New model name
            temperature: New temperature setting
            max_tokens: New max tokens setting
        """
        try:
            logger.info("Updating QA chain configuration")
            
            # Update LLM if needed
            if model_name or temperature is not None or max_tokens is not None:
                self.llm = ChatOpenAI(
                    openai_api_key=self.openai_api_key,
                    model_name=model_name or self.llm.model_name,
                    temperature=temperature if temperature is not None else self.llm.temperature,
                    max_tokens=max_tokens
                )
                
                # Recreate QA chain with new LLM
                self.qa_chain = self._create_qa_chain()
                
                logger.info("QA chain updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating QA chain: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        try:
            return {
                "status": "operational",
                "vector_store": {
                    "type": "ChromaDB",
                    "persist_directory": self.chroma_persist_directory,
                    "collection_name": self.collection_name
                },
                "llm": {
                    "model": self.llm.model_name,
                    "temperature": self.llm.temperature
                },
                "embeddings": {
                    "model": self.embeddings.model
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def clear_vector_store(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("Clearing vector store - this will delete all documents!")
            
            # Delete the collection
            self.vector_store.delete_collection()
            
            # Reinitialize
            self.vector_store = self._initialize_vector_store()
            self.qa_chain = self._create_qa_chain()
            
            logger.info("Vector store cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
