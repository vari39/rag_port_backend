#!/usr/bin/env python3
"""
CSV Data Loader for Port Operations Data
Converts CSV operational data into searchable document embeddings
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVDataLoader:
    """
    Loads CSV operational data and converts it to searchable documents.
    """
    
    def __init__(self, data_dir: str = "./Data/Data-Long Beach Port (Dataset C)"):
        """
        Initialize CSV data loader.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
        self.csv_files = {
            'vessel_calls': 'POLB_vessel_calls_2015.csv',
            'berth_operations': 'POLB_berth_operations_2015.csv',
            'crane_operations': 'POLB_crane_operations_2015.csv',
            'yard_operations': 'POLB_yard_operations_2015.csv',
            'gate_operations': 'POLB_gate_operations_2015.csv',
            'environment': 'environment_timeline_2015_2024.csv'
        }
        
    def load_csv_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the data directory"""
        dataframes = {}
        for key, filename in self.csv_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"Loading {filename}...")
                try:
                    dataframes[key] = pd.read_csv(filepath)
                    logger.info(f"  ✓ Loaded {len(dataframes[key])} records")
                except Exception as e:
                    logger.error(f"  ✗ Error loading {filename}: {e}")
            else:
                logger.warning(f"  ⚠ Warning: {filename} not found at {filepath}")
        
        return dataframes
    
    def create_vessel_documents(self, df: pd.DataFrame, limit: Optional[int] = None) -> List[Document]:
        """Create searchable documents from vessel calls data"""
        documents = []
        
        df_to_process = df.head(limit) if limit else df
        
        for idx, row in df_to_process.iterrows():
            # Create a descriptive document from each vessel call
            doc_content = f"""Vessel Call Record:
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
                'capacity_teu': int(row.get('VesselCapacityTEU', 0)) if pd.notna(row.get('VesselCapacityTEU')) else 0,
                'size_category': str(row.get('SizeCategory', 'Unknown')),
                'source': 'POLB_vessel_calls_2015.csv',
                'data_source': 'csv',
                'processed_at': datetime.now().isoformat()
            }

            documents.append(Document(page_content=doc_content, metadata=metadata))

        return documents
    
    def create_berth_documents(self, df: pd.DataFrame, limit: Optional[int] = None) -> List[Document]:
        """Create searchable documents from berth operations data"""
        documents = []
        
        df_to_process = df.head(limit) if limit else df
        
        for idx, row in df_to_process.iterrows():
            doc_content = f"""Berth Operation Record:
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
                'source': 'POLB_berth_operations_2015.csv',
                'data_source': 'csv',
                'processed_at': datetime.now().isoformat()
            }

            documents.append(Document(page_content=doc_content, metadata=metadata))

        return documents
    
    def create_environment_summaries(self, df: pd.DataFrame) -> List[Document]:
        """Create daily weather summaries from hourly environment data"""
        documents = []

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date

        # Group by date for daily summaries
        for date, group in df.groupby('date'):
            # Calculate daily statistics
            doc_content = f"""Environmental Conditions Summary for {date}:
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
                'source': 'environment_timeline_2015_2024.csv',
                'data_source': 'csv',
                'processed_at': datetime.now().isoformat()
            }

            documents.append(Document(page_content=doc_content, metadata=metadata))

        return documents
    
    def create_crane_summaries(self, df: pd.DataFrame, limit: Optional[int] = None) -> List[Document]:
        """Create summaries of crane operations by vessel call"""
        documents = []

        # Group by call_id
        unique_calls = df['call_id'].unique()
        if limit:
            unique_calls = unique_calls[:limit]
        
        for call_id in unique_calls:
            group = df[df['call_id'] == call_id]
            crane_list = ', '.join(group['crane_id'].unique())
            total_moves = group['total_moves'].sum()
            avg_productivity = group['crane_productivity_mph'].mean()
            total_hours = group['crane_hours'].sum()

            doc_content = f"""Crane Operations Summary for Call {call_id}:
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
                'source': 'POLB_crane_operations_2015.csv',
                'data_source': 'csv',
                'processed_at': datetime.now().isoformat()
            }

            documents.append(Document(page_content=doc_content, metadata=metadata))

        return documents
    
    def load_all_documents(self, 
                          vessel_limit: Optional[int] = None,
                          berth_limit: Optional[int] = None,
                          crane_limit: Optional[int] = None,
                          filter_year: Optional[int] = None) -> List[Document]:
        """
        Load all CSV data and convert to documents.
        
        Args:
            vessel_limit: Limit number of vessel records (None for all)
            berth_limit: Limit number of berth records (None for all)
            crane_limit: Limit number of crane call_ids (None for all)
            filter_year: Filter environment data by year (e.g., 2015)
        
        Returns:
            List of Document objects
        """
        all_documents = []
        
        logger.info("Loading CSV data files...")
        dataframes = self.load_csv_data()
        
        if not dataframes:
            logger.warning("No CSV data files found!")
            return []
        
        # Process vessel calls
        if 'vessel_calls' in dataframes:
            logger.info("Processing vessel calls...")
            docs = self.create_vessel_documents(dataframes['vessel_calls'], vessel_limit)
            all_documents.extend(docs)
            logger.info(f"  ✓ Created {len(docs)} vessel call documents")
        
        # Process berth operations
        if 'berth_operations' in dataframes:
            logger.info("Processing berth operations...")
            docs = self.create_berth_documents(dataframes['berth_operations'], berth_limit)
            all_documents.extend(docs)
            logger.info(f"  ✓ Created {len(docs)} berth operation documents")
        
        # Process environment data
        if 'environment' in dataframes:
            logger.info("Processing environmental data...")
            env_df = dataframes['environment']
            if filter_year:
                env_df = env_df[env_df['timestamp'].str.startswith(str(filter_year))]
            docs = self.create_environment_summaries(env_df)
            all_documents.extend(docs)
            logger.info(f"  ✓ Created {len(docs)} environmental summary documents")
        
        # Process crane operations
        if 'crane_operations' in dataframes:
            logger.info("Processing crane operations...")
            docs = self.create_crane_summaries(dataframes['crane_operations'], crane_limit)
            all_documents.extend(docs)
            logger.info(f"  ✓ Created {len(docs)} crane operation summaries")
        
        logger.info(f"Total documents created: {len(all_documents)}")
        return all_documents

