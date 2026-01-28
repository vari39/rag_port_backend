#!/usr/bin/env python3
"""
Enhanced ChromaDB setup script for Port of Long Beach operational data
Loads CSV data files and creates searchable document embeddings
"""

import os
import chromadb
import pandas as pd
from datetime import datetime
import json
from typing import List, Dict, Tuple


def load_csv_data(data_dir: str = "./data") -> Dict[str, pd.DataFrame]:
    """Load all CSV files from the data directory"""
    csv_files = {
        'vessel_calls': 'POLB_vessel_calls_2015.csv',
        'berth_operations': 'POLB_berth_operations_2015.csv',
        'crane_operations': 'POLB_crane_operations_2015.csv',
        'yard_operations': 'POLB_yard_operations_2015.csv',
        'gate_operations': 'POLB_gate_operations_2015.csv',
        'environment': 'environment_timeline_2015_2024.csv'
    }

    dataframes = {}
    for key, filename in csv_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            dataframes[key] = pd.read_csv(filepath)
            print(f"  ✓ Loaded {len(dataframes[key])} records")
        else:
            print(f"  ⚠ Warning: {filename} not found")

    return dataframes


def create_vessel_documents(df: pd.DataFrame) -> Tuple[List[str], List[Dict], List[str]]:
    """Create searchable documents from vessel calls data"""
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        # Create a descriptive document from each vessel call
        doc = f"""Vessel Call Record:
Vessel: {row['VesselName']} (MMSI: {row['MMSI']})
Shipping Line: {row.get('ShippingLine', 'N/A')}
Capacity: {row.get('VesselCapacityTEU', 'N/A')} TEU
Size Category: {row.get('SizeCategory', 'N/A')}
Arrival: {row['arrival_time']}
Departure: {row['departure_time']}
Dwell Time: {row['dwell_time_hours']:.1f} hours
Visit Number: {row.get('visit_id', 0)}"""

        metadata = {
            'document_type': 'vessel_call',
            'vessel_name': str(row['VesselName']),
            'mmsi': str(row['MMSI']),
            'shipping_line': str(row.get('ShippingLine', 'Unknown')),
            'arrival_time': str(row['arrival_time']),
            'dwell_time_hours': float(row['dwell_time_hours']),
            'capacity_teu': int(row.get('VesselCapacityTEU', 0)),
            'size_category': str(row.get('SizeCategory', 'Unknown')),
            'source': 'POLB_vessel_calls_2015.csv'
        }

        documents.append(doc)
        metadatas.append(metadata)
        ids.append(f"vessel_{row['MMSI']}_{idx}")

    return documents, metadatas, ids


def create_berth_documents(df: pd.DataFrame) -> Tuple[List[str], List[Dict], List[str]]:
    """Create searchable documents from berth operations data"""
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        doc = f"""Berth Operation Record:
Call ID: {row.get('call_id', 'N/A')}
Vessel: {row.get('vessel_name', 'N/A')}
Terminal: {row.get('terminal_name', 'N/A')} ({row.get('terminal_code', 'N/A')})
Berth: {row.get('berth_id', 'N/A')} at {row.get('pier', 'N/A')}
Operator: {row.get('operator', 'N/A')}
Actual Time at Berth: {row.get('atb', 'N/A')}
Actual Time Departed: {row.get('atd', 'N/A')}
Container Operations: {row.get('containers_actual', 0)} TEU total
  - Discharge: {row.get('discharge_actual', 0)} TEU
  - Load: {row.get('load_actual', 0)} TEU
  - Restows: {row.get('restows', 0)}
Special Cargo:
  - Reefer: {row.get('reefer_containers', 0)} containers
  - Hazmat: {row.get('hazmat_containers', 0)} containers
  - Oversize: {row.get('oversize_containers', 0)} containers
Productivity: {row.get('berth_productivity_mph', 0):.2f} moves per hour
Duration: {row.get('actual_duration_hours', 0):.1f} hours"""

        metadata = {
            'document_type': 'berth_operation',
            'call_id': str(row.get('call_id', '')),
            'vessel_name': str(row.get('vessel_name', '')),
            'terminal_code': str(row.get('terminal_code', '')),
            'berth_id': str(row.get('berth_id', '')),
            'containers_total': int(row.get('containers_actual', 0)),
            'productivity_mph': float(row.get('berth_productivity_mph', 0)),
            'duration_hours': float(row.get('actual_duration_hours', 0)),
            'hazmat_containers': int(row.get('hazmat_containers', 0)),
            'reefer_containers': int(row.get('reefer_containers', 0)),
            'source': 'POLB_berth_operations_2015.csv'
        }

        documents.append(doc)
        metadatas.append(metadata)
        ids.append(f"berth_{row.get('call_id', idx)}")

    return documents, metadatas, ids


def create_environment_summaries(df: pd.DataFrame, chunk_size: int = 24) -> Tuple[List[str], List[Dict], List[str]]:
    """Create daily weather summaries from hourly environment data"""
    documents = []
    metadatas = []
    ids = []

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Group by date for daily summaries
    for date, group in df.groupby('date'):
        # Calculate daily statistics
        doc = f"""Environmental Conditions Summary for {date}:
Temperature: {group['air_temp_c'].mean():.1f}°C (min: {group['air_temp_c'].min():.1f}°C, max: {group['air_temp_c'].max():.1f}°C)
Water Temperature: {group['water_temp_c'].mean():.1f}°C
Pressure: {group['pressure_hpa'].mean():.1f} hPa
Wind: {group['wind_speed_ms'].mean():.1f} m/s average, gusts up to {group['wind_gust_ms'].max():.1f} m/s
Wave Height: {group['wave_height_m'].mean():.2f} m average (max: {group['wave_height_m'].max():.2f} m)
Wave Period: {group['dominant_period_s'].mean():.1f} seconds
Tide Range: {group['tide_ft'].min():.2f} ft to {group['tide_ft'].max():.2f} ft

Events:
  - Storm: {'Yes' if group['event_storm'].sum() > 0 else 'No'}
  - High Tide: {'Yes' if group['event_high_tide'].sum() > 0 else 'No'}
  - Low Tide: {'Yes' if group['event_low_tide'].sum() > 0 else 'No'}
  - Heat Event: {'Yes' if group['event_heat'].sum() > 0 else 'No'}
  - Pressure Drop: {'Yes' if group['event_pressure_drop'].sum() > 0 else 'No'}"""

        metadata = {
            'document_type': 'environmental_conditions',
            'date': str(date),
            'avg_air_temp': float(group['air_temp_c'].mean()),
            'max_wind_speed': float(group['wind_gust_ms'].max()),
            'avg_wave_height': float(group['wave_height_m'].mean()),
            'max_wave_height': float(group['wave_height_m'].max()),
            'has_storm': int(group['event_storm'].sum() > 0),
            'has_high_tide': int(group['event_high_tide'].sum() > 0),
            'source': 'environment_timeline_2015_2024.csv'
        }

        documents.append(doc)
        metadatas.append(metadata)
        ids.append(f"env_{date}")

    return documents, metadatas, ids


def create_crane_summaries(df: pd.DataFrame) -> Tuple[List[str], List[Dict], List[str]]:
    """Create summaries of crane operations by vessel call"""
    documents = []
    metadatas = []
    ids = []

    # Group by call_id
    for call_id, group in df.groupby('call_id'):
        crane_list = ', '.join(group['crane_id'].unique())
        total_moves = group['total_moves'].sum()
        avg_productivity = group['crane_productivity_mph'].mean()
        total_hours = group['crane_hours'].sum()

        doc = f"""Crane Operations Summary for Call {call_id}:
Number of Cranes: {len(group)}
Cranes Used: {crane_list}
Total Moves: {total_moves} containers
Average Productivity: {avg_productivity:.2f} moves per hour
Total Crane Hours: {total_hours:.1f} hours
Breakdown Minutes: {group['breakdown_minutes'].sum():.0f} minutes
Average Utilization: {group['utilization_pct'].mean():.1f}%"""

        metadata = {
            'document_type': 'crane_operations',
            'call_id': str(call_id),
            'num_cranes': int(len(group)),
            'total_moves': int(total_moves),
            'avg_productivity': float(avg_productivity),
            'total_hours': float(total_hours),
            'source': 'POLB_crane_operations_2015.csv'
        }

        documents.append(doc)
        metadatas.append(metadata)
        ids.append(f"crane_{call_id}")

    return documents, metadatas, ids


def setup_chroma_database(data_dir: str = "./data"):
    """Initialize ChromaDB with Port of Long Beach operational data"""

    print("=" * 60)
    print("Setting up ChromaDB for Port Operations Data")
    print("=" * 60)

    # Create storage directory
    storage_dir = "./storage/chroma"
    os.makedirs(storage_dir, exist_ok=True)

    # Load CSV data
    print("\n1. Loading CSV files...")
    dataframes = load_csv_data(data_dir)

    if not dataframes:
        print("❌ No data files found!")
        return False

    # Initialize ChromaDB client
    print("\n2. Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=storage_dir)

    # Create or reset collection
    try:
        client.delete_collection("port_documents")
        print("  ✓ Cleared existing collection")
    except:
        pass

    collection = client.create_collection(
        name="port_documents",
        metadata={"description": "Port of Long Beach operational data for AI decision support"}
    )

    # Process each data type
    all_documents = []
    all_metadatas = []
    all_ids = []

    print("\n3. Processing data files...")

    # Vessel calls (sample first 100 for demo, remove limit for production)
    if 'vessel_calls' in dataframes:
        print("  Processing vessel calls...")
        docs, metas, ids = create_vessel_documents(dataframes['vessel_calls'].head(100))
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    ✓ Created {len(docs)} vessel call documents")

    # Berth operations (sample first 100)
    if 'berth_operations' in dataframes:
        print("  Processing berth operations...")
        docs, metas, ids = create_berth_documents(dataframes['berth_operations'].head(100))
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    ✓ Created {len(docs)} berth operation documents")

    # Environment data (create daily summaries for 2015)
    if 'environment' in dataframes:
        print("  Processing environmental data...")
        env_2015 = dataframes['environment'][
            dataframes['environment']['timestamp'].str.startswith('2015')
        ]
        docs, metas, ids = create_environment_summaries(env_2015)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    ✓ Created {len(docs)} environmental summary documents")

    # Crane operations (summarize by call_id, sample first 100 calls)
    if 'crane_operations' in dataframes:
        print("  Processing crane operations...")
        crane_sample = dataframes['crane_operations'][
            dataframes['crane_operations']['call_id'].isin(
                dataframes['crane_operations']['call_id'].unique()[:100]
            )
        ]
        docs, metas, ids = create_crane_summaries(crane_sample)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    ✓ Created {len(docs)} crane operation summaries")

    # Add to ChromaDB in batches
    print(f"\n4. Adding {len(all_documents)} documents to ChromaDB...")
    batch_size = 100
    for i in range(0, len(all_documents), batch_size):
        batch_docs = all_documents[i:i+batch_size]
        batch_metas = all_metadatas[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]

        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        print(f"  ✓ Added batch {i//batch_size + 1} ({len(batch_docs)} documents)")

    print(f"\n✅ Successfully loaded {len(all_documents)} documents into ChromaDB")
    print(f"📁 Database stored in: {storage_dir}")
    print(f"📊 Collection: port_documents")

    # Test queries
    print("\n5. Testing queries...")
    test_queries = [
        "What vessels arrived at the port in January 2015?",
        "Tell me about berth operations and productivity",
        "What were the weather conditions in January 2015?",
        "Which cranes were used for vessel operations?"
    ]

    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        print(f"\n🧪 Query: '{query}'")
        print(f"   Found {len(results['documents'][0])} results")
        if results['documents'][0]:
            print(f"   Top result type: {results['metadatas'][0][0]['document_type']}")

    return True


if __name__ == "__main__":
    try:
        success = setup_chroma_database()
        if success:
            print("\n" + "=" * 60)
            print("🎉 Database setup completed successfully!")
            print("=" * 60)
            print("\nYou can now:")
            print("  - Run queries against the port_documents collection")
            print("  - Use this data with your AI Port Decision-Support System")
            print("  - Access the database at ./storage/chroma")
    except Exception as e:
        print(f"\n❌ Error setting up database: {e}")
        import traceback
        traceback.print_exc()
        raise
