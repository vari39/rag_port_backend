#!/usr/bin/env python3
"""
Setup script for ChromaDB database initialization
Creates sample documents for testing the AI Port Decision-Support System
"""

import os
import chromadb
from datetime import datetime
import json

def setup_chroma_database():
    """Initialize ChromaDB with sample port documents"""
    
    # Create storage directory
    storage_dir = "./storage/chroma"
    os.makedirs(storage_dir, exist_ok=True)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=storage_dir)
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="port_documents",
        metadata={"description": "Port operations documents for AI decision support"}
    )
    
    # Sample port documents
    sample_documents = [
        {
            "content": "Port Safety Protocol: All vessels must maintain a minimum distance of 200 meters from other vessels during berthing operations. Emergency procedures must be followed in case of equipment failure or adverse weather conditions.",
            "metadata": {
                "document_type": "safety_protocols",
                "source": "Port_Safety_Manual_v2.1.pdf",
                "page": 15,
                "timestamp": "2024-01-15T10:30:00Z",
                "port": "Los Angeles Port"
            }
        },
        {
            "content": "Weather Advisory: Current wind speed is 25 knots with gusts up to 35 knots. Visibility reduced to 2 nautical miles due to fog. All berthing operations should be suspended until conditions improve.",
            "metadata": {
                "document_type": "weather_reports",
                "source": "Weather_Report_2024-01-15.pdf",
                "page": 1,
                "timestamp": "2024-01-15T08:00:00Z",
                "port": "Los Angeles Port"
            }
        },
        {
            "content": "Berth Allocation Schedule: Berth 1A available from 14:00-18:00 for container vessel MV Horizon. Berth 2B reserved for bulk carrier MV Pacific from 16:00-20:00. Priority given to vessels with confirmed cargo operations.",
            "metadata": {
                "document_type": "berth_schedules",
                "source": "Berth_Schedule_2024-01-15.pdf",
                "page": 3,
                "timestamp": "2024-01-15T06:00:00Z",
                "port": "Los Angeles Port"
            }
        },
        {
            "content": "Cargo Handling Procedure: Container operations require crane capacity of minimum 40 tons. Bulk cargo handling follows standard procedures with conveyor system. All cargo must be properly secured before vessel departure.",
            "metadata": {
                "document_type": "sop",
                "source": "Cargo_Handling_SOP_v3.0.pdf",
                "page": 8,
                "timestamp": "2024-01-10T14:20:00Z",
                "port": "Los Angeles Port"
            }
        },
        {
            "content": "Emergency Response Plan: In case of fire, activate alarm system immediately. Evacuate personnel to designated assembly points. Contact emergency services at +1-555-911. Use fire suppression systems as per training protocols.",
            "metadata": {
                "document_type": "safety_protocols",
                "source": "Emergency_Response_Plan.pdf",
                "page": 12,
                "timestamp": "2024-01-05T09:15:00Z",
                "port": "Los Angeles Port"
            }
        },
        {
            "content": "Equipment Maintenance Log: Crane #3 scheduled for maintenance on 2024-01-20. Current status: Operational. Last inspection: 2024-01-10. Next inspection due: 2024-02-10. No issues reported.",
            "metadata": {
                "document_type": "maintenance_logs",
                "source": "Equipment_Maintenance_Log.pdf",
                "page": 5,
                "timestamp": "2024-01-12T11:45:00Z",
                "port": "Los Angeles Port"
            }
        },
        {
            "content": "Regulatory Compliance: All vessels must comply with IMO regulations for ballast water management. Port state control inspections required for vessels over 400 GT. Documentation must be available for customs clearance.",
            "metadata": {
                "document_type": "regulatory_docs",
                "source": "IMO_Compliance_Guide.pdf",
                "page": 22,
                "timestamp": "2024-01-08T16:30:00Z",
                "port": "Los Angeles Port"
            }
        },
        {
            "content": "Vessel Manifest: MV Horizon arriving 14:00 with 1200 TEU containers. Cargo includes electronics, textiles, and machinery. Hazardous materials: 50 containers Class 3 (flammable liquids). Special handling required.",
            "metadata": {
                "document_type": "vessel_manifests",
                "source": "MV_Horizon_Manifest_2024-01-15.pdf",
                "page": 1,
                "timestamp": "2024-01-15T12:00:00Z",
                "port": "Los Angeles Port"
            }
        }
    ]
    
    # Add documents to collection
    documents = []
    metadatas = []
    ids = []
    
    for i, doc in enumerate(sample_documents):
        documents.append(doc["content"])
        metadatas.append(doc["metadata"])
        ids.append(f"doc_{i+1}")
    
    # Add to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Successfully added {len(sample_documents)} documents to ChromaDB")
    print(f"📁 Database stored in: {storage_dir}")
    print(f"📊 Collection: port_documents")
    
    # Test the collection
    test_query = "What are the safety procedures for berthing operations?"
    results = collection.query(
        query_texts=[test_query],
        n_results=3
    )
    
    print(f"\n🧪 Test query: '{test_query}'")
    print(f"📋 Found {len(results['documents'][0])} relevant documents")
    
    return True

if __name__ == "__main__":
    try:
        setup_chroma_database()
        print("\n🎉 Database setup completed successfully!")
        print("You can now run the AI Port Decision-Support System")
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        raise
