import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class ParkingDataGenerator:
    """
    Generates realistic parking lot data for simulation.
    """
    
    def __init__(self, num_lots: int = 14, seed: int = 42):
        self.num_lots = num_lots
        self.seed = seed
        np.random.seed(seed)
        
        # Generate static lot characteristics
        self.lot_characteristics = self._generate_lot_characteristics()
        
        # Time-based patterns
        self.time_patterns = self._generate_time_patterns()
        
        # Special events calendar
        self.special_events = self._generate_special_events()
        
        # Current simulation state
        self.current_day = 0
        self.current_time_slot = 0
        
    def _generate_lot_characteristics(self) -> Dict:
        """Generate static characteristics for each parking lot."""
        # Generate diverse lot capacities
        capacities = np.random.choice([20, 30, 50, 75, 100], size=self.num_lots, 
                                    p=[0.2, 0.3, 0.3, 0.15, 0.05])
        
        # Generate coordinates in a realistic urban grid
        # Simulating a 5x3 grid of parking lots in a city area
        grid_positions = [(i % 5, i // 5) for i in range(self.num_lots)]
        
        # Base coordinates (simulating downtown area)
        base_lat, base_lon = 40.7589, -73.9851  # Near NYC coordinates
        
        latitudes = []
        longitudes = []
        
        for x, y in grid_positions:
            # Add some randomness to create realistic street layout
            lat = base_lat + (y * 0.002) + np.random.normal(0, 0.0005)
            lon = base_lon + (x * 0.002) + np.random.normal(0, 0.0005)
            latitudes.append(lat)
            longitudes.append(lon)
        
        # Lot type influences demand patterns
        lot_types = np.random.choice(['commercial', 'residential', 'mixed'], 
                                   size=self.num_lots, p=[0.4, 0.3, 0.3])
        
        return {
            'capacity': capacities,
            'latitude': np.array(latitudes),
            'longitude': np.array(longitudes),
            'lot_type': lot_types
        }
    
    def _generate_time_patterns(self) -> Dict:
        """Generate realistic time-based demand patterns."""
        # Different patterns for different lot types
        patterns = {}
        
        # Commercial lots: High demand during business hours
        commercial_pattern = np.array([
            0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9,  # Morning rush
            0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.3  # Afternoon decline
        ])
        
        # Residential lots: Different pattern
        residential_pattern = np.array([
            0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.25, 0.3,  # People leaving for work
            0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.85  # People returning
        ])
        
        # Mixed lots: Blend of both
        mixed_pattern = (commercial_pattern + residential_pattern) / 2
        
        patterns['commercial'] = commercial_pattern
        patterns['residential'] = residential_pattern
        patterns['mixed'] = mixed_pattern
        
        return patterns
    
    def _generate_special_events(self) -> List[int]:
        """Generate random special event days."""
        # Simulate special events on random days
        special_days = np.random.choice(73, size=int(73 * 0.1), replace=False)
        return special_days.tolist()
    
    def _calculate_base_occupancy(self, lot_idx: int, time_slot: int, day: int) -> float:
        """Calculate base occupancy rate for a lot at given time."""
        lot_type = self.lot_characteristics['lot_type'][lot_idx]
        pattern = self.time_patterns[lot_type]
        
        # Base occupancy from time pattern
        base_occupancy = pattern[time_slot]
        
        # Add day-of-week variation
        day_of_week = day % 7
        if day_of_week in [5, 6]:  # Weekend
            if lot_type == 'commercial':
                base_occupancy *= 0.6  # Less demand on weekends
            else:
                base_occupancy *= 1.2  # More demand for residential
        
        # Add some randomness
        base_occupancy += np.random.normal(0, 0.1)
        
        return np.clip(base_occupancy, 0.0, 1.0)
    
    def _generate_traffic_level(self, time_slot: int, day: int) -> float:
        """Generate traffic congestion level."""
        # Traffic patterns: higher during rush hours
        traffic_pattern = np.array([
            0.6, 0.7, 0.8, 0.9, 0.95, 0.9, 0.7, 0.6, 0.5,  # Morning rush
            0.4, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8   # Evening rush
        ])
        
        base_traffic = traffic_pattern[time_slot]
        
        # Weekend adjustment
        if day % 7 in [5, 6]:
            base_traffic *= 0.7
        
        # Add randomness
        base_traffic += np.random.normal(0, 0.1)
        
        return np.clip(base_traffic, 0.0, 1.0)
    
    def _generate_vehicle_types(self) -> List[str]:
        """Generate vehicle types for each lot."""
        vehicle_types = []
        for _ in range(self.num_lots):
            # Different probabilities for different vehicle types
            vehicle_type = np.random.choice(['car', 'truck', 'bike'], 
                                          p=[0.7, 0.2, 0.1])
            vehicle_types.append(vehicle_type)
        return vehicle_types
    
    def generate_timestep_data(self, time_step: int) -> Dict:
        """
        Generate data for a specific time step.
        
        Args:
            time_step: Current time step in simulation
            
        Returns:
            Dictionary containing all parking lot data
        """
        # Calculate current day and time slot
        day = time_step // 18  # 18 time slots per day
        time_slot = time_step % 18
        
        # Update internal state
        self.current_day = day
        self.current_time_slot = time_slot
        
        # Generate occupancy data
        occupancy = []
        queue_length = []
        
        for i in range(self.num_lots):
            # Calculate base occupancy
            base_occupancy_rate = self._calculate_base_occupancy(i, time_slot, day)
            
            # Convert to actual occupancy
            capacity = self.lot_characteristics['capacity'][i]
            occupancy_count = int(base_occupancy_rate * capacity)
            
            # Generate queue length based on demand vs capacity
            if base_occupancy_rate > 0.9:
                # High demand creates queues
                queue_count = np.random.poisson(max(0, (base_occupancy_rate - 0.9) * 20))
            else:
                queue_count = 0
            
            occupancy.append(occupancy_count)
            queue_length.append(queue_count)
        
        # Generate other factors
        traffic_level = self._generate_traffic_level(time_slot, day)
        traffic_levels = np.random.normal(traffic_level, 0.1, self.num_lots)
        traffic_levels = np.clip(traffic_levels, 0.0, 1.0)
        
        # Special day indicator
        is_special_day = day in self.special_events
        special_day_flags = np.full(self.num_lots, is_special_day)
        
        # Vehicle types
        vehicle_types = self._generate_vehicle_types()
        
        return {
            'capacity': self.lot_characteristics['capacity'],
            'occupancy': np.array(occupancy),
            'queue_length': np.array(queue_length),
            'traffic_level': traffic_levels,
            'special_day': special_day_flags,
            'vehicle_type': vehicle_types,
            'latitude': self.lot_characteristics['latitude'],
            'longitude': self.lot_characteristics['longitude'],
            'lot_type': self.lot_characteristics['lot_type'],
            'day': day,
            'time_slot': time_slot,
            'timestamp': datetime.now() - timedelta(seconds=time_step * 30)
        }
    
    def generate_historical_data(self, num_days: int = 73) -> pd.DataFrame:
        """
        Generate historical data for analysis.
        
        Args:
            num_days: Number of days to generate
            
        Returns:
            DataFrame with historical parking data
        """
        data_records = []
        
        for day in range(num_days):
            for time_slot in range(18):  # 18 time slots per day
                time_step = day * 18 + time_slot
                timestep_data = self.generate_timestep_data(time_step)
                
                # Convert to record format
                for lot_idx in range(self.num_lots):
                    record = {
                        'day': day,
                        'time_slot': time_slot,
                        'lot_id': lot_idx,
                        'capacity': timestep_data['capacity'][lot_idx],
                        'occupancy': timestep_data['occupancy'][lot_idx],
                        'queue_length': timestep_data['queue_length'][lot_idx],
                        'traffic_level': timestep_data['traffic_level'][lot_idx],
                        'special_day': timestep_data['special_day'][lot_idx],
                        'vehicle_type': timestep_data['vehicle_type'][lot_idx],
                        'latitude': timestep_data['latitude'][lot_idx],
                        'longitude': timestep_data['longitude'][lot_idx],
                        'lot_type': timestep_data['lot_type'][lot_idx],
                        'timestamp': timestep_data['timestamp']
                    }
                    data_records.append(record)
        
        return pd.DataFrame(data_records)
