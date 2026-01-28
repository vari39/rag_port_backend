#!/usr/bin/env python3
"""
Unified Setup Script for AI Port Decision-Support System
Processes all data sources: PDFs (multilingual) and CSV operational data
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.port_information_retrieval import PortInformationRetrieval
from src.utils.loaders import create_document_loader, PortDocumentProcessor
from src.utils.csv_loader import CSVDataLoader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_unified_database(
    openai_api_key: str,
    pdf_directories: Optional[List[str]] = None,
    csv_data_dir: Optional[str] = None,
    enable_multilingual: bool = True,
    enable_ocr: bool = True,
    enable_csv: bool = True,
    vessel_limit: Optional[int] = None,
    berth_limit: Optional[int] = None,
    crane_limit: Optional[int] = None,
    chroma_persist_directory: str = "./storage/chroma",
    collection_name: str = "port_documents"
):
    """
    Unified setup function that processes all data sources.
    
    Args:
        openai_api_key: OpenAI API key
        pdf_directories: List of directories containing PDF files
        csv_data_dir: Directory containing CSV files
        enable_multilingual: Enable multilingual processing
        enable_ocr: Enable OCR for scanned documents
        enable_csv: Enable CSV data loading
        vessel_limit: Limit vessel records (None for all)
        berth_limit: Limit berth records (None for all)
        crane_limit: Limit crane call_ids (None for all)
        chroma_persist_directory: ChromaDB storage directory
        collection_name: ChromaDB collection name
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print("=" * 80)
        print("Unified Database Setup for AI Port Decision-Support System")
        print("=" * 80)
        
        # Initialize Port Information Retrieval system
        logger.info("Initializing Port Information Retrieval system...")
        port_retrieval = PortInformationRetrieval(
            openai_api_key=openai_api_key,
            chroma_persist_directory=chroma_persist_directory,
            collection_name=collection_name
        )
        
        all_documents: List[Document] = []
        
        # =====================================================================
        # Process PDF Documents (Multilingual)
        # =====================================================================
        if pdf_directories:
            print("\n" + "=" * 80)
            print("STEP 1: Processing PDF Documents (Multilingual Support)")
            print("=" * 80)
            
            # Create enhanced document loader
            loader = create_document_loader(
                chunk_size=1000,
                chunk_overlap=200,
                multilingual=enable_multilingual,
                enhanced=True,
                enable_ocr=enable_ocr
            )
            
            # Process each PDF directory
            for pdf_dir in pdf_directories:
                pdf_path = Path(pdf_dir)
                if not pdf_path.exists():
                    logger.warning(f"PDF directory not found: {pdf_dir}")
                    continue
                
                logger.info(f"Processing PDFs from: {pdf_dir}")
                
                try:
                    if hasattr(loader, 'load_pdf_documents'):
                        # Enhanced multilingual loader
                        docs = loader.load_pdf_documents(
                            pdf_directory=str(pdf_path),
                            recursive=True,
                            auto_translate=True
                        )
                    else:
                        # Fallback to basic loader
                        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
                        basic_loader = DirectoryLoader(
                            str(pdf_path),
                            glob="**/*.pdf",
                            loader_cls=PyPDFLoader
                        )
                        docs = basic_loader.load()
                    
                    # Process with port-specific metadata
                    processor = PortDocumentProcessor()
                    docs = processor.process_documents(docs)
                    
                    all_documents.extend(docs)
                    logger.info(f"  ✓ Processed {len(docs)} documents from {pdf_dir}")
                    
                except Exception as e:
                    logger.error(f"  ✗ Error processing {pdf_dir}: {e}")
                    continue
            
            print(f"\n✓ Total PDF documents processed: {len([d for d in all_documents if d.metadata.get('data_source') != 'csv'])}")
        
        # =====================================================================
        # Process CSV Operational Data
        # =====================================================================
        if enable_csv and csv_data_dir:
            print("\n" + "=" * 80)
            print("STEP 2: Processing CSV Operational Data")
            print("=" * 80)
            
            csv_loader = CSVDataLoader(data_dir=csv_data_dir)
            
            try:
                csv_docs = csv_loader.load_all_documents(
                    vessel_limit=vessel_limit,
                    berth_limit=berth_limit,
                    crane_limit=crane_limit,
                    filter_year=2015  # Filter environment data to 2015
                )
                
                all_documents.extend(csv_docs)
                logger.info(f"✓ Processed {len(csv_docs)} CSV documents")
                
            except Exception as e:
                logger.error(f"✗ Error processing CSV data: {e}")
                import traceback
                traceback.print_exc()
        
        # =====================================================================
        # Add All Documents to Vector Store
        # =====================================================================
        if not all_documents:
            logger.error("No documents to add to vector store!")
            return False
        
        print("\n" + "=" * 80)
        print(f"STEP 3: Adding {len(all_documents)} Documents to Vector Store")
        print("=" * 80)
        
        import asyncio
        success = asyncio.run(port_retrieval.add_documents_to_vector_store(all_documents))
        
        if not success:
            logger.error("Failed to add documents to vector store")
            return False
        
        # =====================================================================
        # Test Queries
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 4: Testing System with Sample Queries")
        print("=" * 80)
        
        test_queries = [
            "What are the safety protocols for vessel berthing?",
            "Tell me about vessel operations at the port",
            "What were the weather conditions in 2015?",
            "How are cranes used for container operations?"
        ]
        
        for query in test_queries:
            try:
                result = asyncio.run(port_retrieval.query_port_information(query, max_documents=3))
                print(f"\n🧪 Query: '{query}'")
                print(f"   Answer: {result['answer'][:200]}...")
                print(f"   Sources: {len(result['sources'])} documents")
            except Exception as e:
                logger.warning(f"Test query failed: {e}")
        
        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 80)
        print("🎉 Unified Database Setup Completed Successfully!")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  - Total documents processed: {len(all_documents)}")
        print(f"  - PDF documents: {len([d for d in all_documents if d.metadata.get('data_source') != 'csv'])}")
        print(f"  - CSV documents: {len([d for d in all_documents if d.metadata.get('data_source') == 'csv'])}")
        print(f"  - Database location: {chroma_persist_directory}")
        print(f"  - Collection name: {collection_name}")
        print("\nYou can now:")
        print("  - Query the system using PortInformationRetrieval")
        print("  - Start the FastAPI server: python src/api/server.py")
        print("  - Run example scripts: python src/examples/example_usage.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in unified setup: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for unified setup"""
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file or environment")
        return False
    
    # Default data directories (relative to project root)
    project_root = Path(__file__).parent
    data_root = project_root.parent / "Data"
    
    pdf_directories = [
        str(data_root / "Yokohama Port"),
        str(data_root / "Cross-Port Knowledge (Dataset B)" / "Database B" / "port_of_LB_data"),
        str(data_root / "Digital-Port Database")
    ]
    
    csv_data_dir = str(data_root / "Data-Long Beach Port (Dataset C)")
    
    # Check which directories exist
    existing_pdf_dirs = [d for d in pdf_directories if Path(d).exists()]
    
    print(f"\nFound {len(existing_pdf_dirs)} PDF directories:")
    for d in existing_pdf_dirs:
        print(f"  ✓ {d}")
    
    if not Path(csv_data_dir).exists():
        print(f"\n⚠ CSV data directory not found: {csv_data_dir}")
        csv_data_dir = None
    
    # Run unified setup
    success = setup_unified_database(
        openai_api_key=openai_api_key,
        pdf_directories=existing_pdf_dirs if existing_pdf_dirs else None,
        csv_data_dir=csv_data_dir,
        enable_multilingual=True,
        enable_ocr=True,
        enable_csv=True,
        vessel_limit=100,  # Limit for demo - remove for production
        berth_limit=100,  # Limit for demo - remove for production
        crane_limit=100   # Limit for demo - remove for production
    )
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

