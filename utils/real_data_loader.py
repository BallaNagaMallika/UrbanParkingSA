import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os

class RealDataLoader:
    """
    Loads and processes real parking lot data from CSV file.
    """
    
    def __init__(self, data_file: str = "dataset.csv"):
        self.data_file = data_file
        self.raw_data = None
        self.processed_data = None
        self.unique_lots = None
        self.lot_info = {}
        self.current_index = 0
        
        # Load and process data
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """Load raw data from CSV file."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file {self.data_file} not found")
        
        self.raw_data = pd.read_csv(self.data_file)
        print(f"Loaded {len(self.raw_data)} records from {self.data_file}")
        
    def preprocess_data(self):
        """Preprocess the raw data for simulation."""
        # Convert traffic conditions to numerical values
        traffic_map = {
            'low': 0.3,
            'average': 0.6,
            'high': 0.9
        }
        
        # Convert vehicle types to standardized format
        vehicle_map = {
            'car': 'car',
            'truck': 'truck',
            'bike': 'bike',
            'cycle': 'bike'  # Treat cycle as bike
        }
        
        # Process the data
        self.processed_data = self.raw_data.copy()
        
        # Convert traffic conditions
        self.processed_data['TrafficLevel'] = self.processed_data['TrafficConditionNearby'].map(traffic_map)
        
        # Standardize vehicle types
        self.processed_data['VehicleType'] = self.processed_data['VehicleType'].map(vehicle_map)
        
        # Convert datetime
        self.processed_data['DateTime'] = pd.to_datetime(
            self.processed_data['LastUpdatedDate'] + ' ' + self.processed_data['LastUpdatedTime'],
            format='%d-%m-%Y %H:%M:%S'
        )
        
        # Sort by datetime
        self.processed_data = self.processed_data.sort_values('DateTime')
        
        # Get unique parking lots
        self.unique_lots = self.processed_data['SystemCodeNumber'].unique()
        
        # Create lot information dictionary
        for lot in self.unique_lots:
            lot_data = self.processed_data[self.processed_data['SystemCodeNumber'] == lot].iloc[0]
            self.lot_info[lot] = {
                'name': lot,
                'capacity': lot_data['Capacity'],
                'latitude': lot_data['Latitude'],
                'longitude': lot_data['Longitude']
            }
        
        print(f"Found {len(self.unique_lots)} unique parking lots")
        print(f"Lots: {list(self.unique_lots)}")
    
    def get_lot_names(self) -> List[str]:
        """Get list of parking lot names."""
        return list(self.unique_lots)
    
    def get_lot_info(self) -> Dict:
        """Get parking lot information."""
        return self.lot_info
    
    def get_next_timestep_data(self) -> Dict:
        """
        Get data for the next timestep in chronological order.
        
        Returns:
            Dictionary containing parking lot data
        """
        if self.current_index >= len(self.processed_data):
            # Reset to beginning if we've reached the end
            self.current_index = 0
        
        # Get a batch of records (one per lot if available)
        batch_size = min(len(self.unique_lots), len(self.processed_data) - self.current_index)
        current_batch = self.processed_data.iloc[self.current_index:self.current_index + batch_size]
        
        # Initialize arrays for all lots
        num_lots = len(self.unique_lots)
        lot_to_index = {lot: i for i, lot in enumerate(self.unique_lots)}
        
        # Initialize with default values
        capacity = np.zeros(num_lots)
        occupancy = np.zeros(num_lots)
        queue_length = np.zeros(num_lots)
        traffic_level = np.full(num_lots, 0.5)
        special_day = np.zeros(num_lots, dtype=bool)
        vehicle_type = ['car'] * num_lots
        latitude = np.zeros(num_lots)
        longitude = np.zeros(num_lots)
        
        # Fill with lot information
        for i, lot in enumerate(self.unique_lots):
            lot_info = self.lot_info[lot]
            capacity[i] = lot_info['capacity']
            latitude[i] = lot_info['latitude']
            longitude[i] = lot_info['longitude']
        
        # Fill with current data
        for _, row in current_batch.iterrows():
            lot_name = row['SystemCodeNumber']
            if lot_name in lot_to_index:
                idx = lot_to_index[lot_name]
                occupancy[idx] = row['Occupancy']
                queue_length[idx] = row['QueueLength']
                traffic_level[idx] = row['TrafficLevel']
                special_day[idx] = bool(row['IsSpecialDay'])
                vehicle_type[idx] = row['VehicleType']
        
        # Increment index for next call
        self.current_index += batch_size
        
        # Get current timestamp
        if len(current_batch) > 0:
            current_timestamp = current_batch.iloc[0]['DateTime']
        else:
            current_timestamp = datetime.now()
        
        return {
            'capacity': capacity.astype(int),
            'occupancy': occupancy.astype(int),
            'queue_length': queue_length.astype(int),
            'traffic_level': traffic_level,
            'special_day': special_day,
            'vehicle_type': vehicle_type,
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': current_timestamp,
            'lot_names': list(self.unique_lots)
        }
    
    def get_historical_data(self, limit: int = None) -> pd.DataFrame:
        """
        Get historical data for analysis.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with historical data
        """
        data = self.processed_data.copy()
        
        if limit:
            data = data.head(limit)
        
        return data
    
    def get_data_statistics(self) -> Dict:
        """Get basic statistics about the dataset."""
        stats = {
            'total_records': len(self.processed_data),
            'unique_lots': len(self.unique_lots),
            'date_range': {
                'start': self.processed_data['DateTime'].min(),
                'end': self.processed_data['DateTime'].max()
            },
            'vehicle_type_distribution': self.processed_data['VehicleType'].value_counts().to_dict(),
            'traffic_condition_distribution': self.processed_data['TrafficConditionNearby'].value_counts().to_dict(),
            'special_day_percentage': (self.processed_data['IsSpecialDay'].sum() / len(self.processed_data)) * 100
        }
        
        return stats
    
    def reset_simulation(self):
        """Reset simulation to the beginning."""
        self.current_index = 0