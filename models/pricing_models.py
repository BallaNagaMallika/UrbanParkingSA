import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from utils.geographic_utils import calculate_distance

class BaselinePricingModel:
    """
    Simple linear pricing model based on occupancy rate.
    Price(t+1) = Price(t) + α × (Occupancy / Capacity)
    """
    
    def __init__(self, base_price: float = 10.0, alpha: float = 0.5, num_lots: int = 14):
        self.base_price = base_price
        self.alpha = alpha
        self.num_lots = num_lots
        self.current_prices = np.full(num_lots, base_price)
        self.min_price = base_price * 0.5
        self.max_price = base_price * 2.0
    
    def calculate_prices(self, data: Dict) -> np.ndarray:
        """
        Calculate new prices based on current occupancy rates.
        
        Args:
            data: Dictionary containing parking lot data
            
        Returns:
            Array of new prices for each parking lot
        """
        occupancy_rates = data['occupancy'] / data['capacity']
        
        # Calculate price adjustments
        price_adjustments = self.alpha * occupancy_rates
        
        # Update prices
        self.current_prices = self.current_prices + price_adjustments
        
        # Apply bounds
        self.current_prices = np.clip(self.current_prices, self.min_price, self.max_price)
        
        return self.current_prices.copy()
    
    def reset_prices(self):
        """Reset all prices to base price."""
        self.current_prices = np.full(self.num_lots, self.base_price)

class DemandBasedPricingModel:
    """
    Advanced pricing model based on multiple demand factors.
    
    Demand = α×(Occupancy/Capacity) + β×QueueLength - γ×Traffic + δ×SpecialDay + ε×VehicleType
    Price = BasePrice × (1 + λ × NormalizedDemand)
    """
    
    def __init__(self, base_price: float = 10.0):
        self.base_price = base_price
        self.min_price = base_price * 0.5
        self.max_price = base_price * 2.0
        
        # Default parameters
        self.alpha = 1.0    # Occupancy weight
        self.beta = 0.5     # Queue weight
        self.gamma = 0.3    # Traffic weight (negative impact)
        self.delta = 0.4    # Special day weight
        self.epsilon = 0.2  # Vehicle type weight
        self.lambda_param = 0.5  # Price sensitivity
        
        # Vehicle type weights
        self.vehicle_weights = {
            'car': 1.0,
            'truck': 1.5,
            'bike': 0.5
        }
    
    def update_parameters(self, alpha: float, beta: float, gamma: float, 
                         delta: float, epsilon: float):
        """Update model parameters."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
    
    def calculate_demand(self, data: Dict) -> np.ndarray:
        """
        Calculate demand score for each parking lot.
        
        Args:
            data: Dictionary containing parking lot data
            
        Returns:
            Array of demand scores
        """
        # Occupancy factor
        occupancy_factor = self.alpha * (data['occupancy'] / data['capacity'])
        
        # Queue factor
        queue_factor = self.beta * data['queue_length']
        
        # Traffic factor (negative impact - high traffic reduces demand)
        traffic_factor = -self.gamma * data['traffic_level']
        
        # Special day factor
        special_day_factor = self.delta * data['special_day'].astype(float)
        
        # Vehicle type factor
        vehicle_type_factor = self.epsilon * np.array([
            self.vehicle_weights.get(vtype, 1.0) for vtype in data['vehicle_type']
        ])
        
        # Calculate total demand
        demand = (occupancy_factor + queue_factor + traffic_factor + 
                 special_day_factor + vehicle_type_factor)
        
        return demand
    
    def normalize_demand(self, demand: np.ndarray) -> np.ndarray:
        """
        Normalize demand to ensure smooth price variations.
        
        Args:
            demand: Raw demand scores
            
        Returns:
            Normalized demand scores
        """
        # Use tanh normalization to keep values bounded
        normalized = np.tanh(demand)
        
        # Scale to reasonable range
        normalized = normalized * 0.5  # Max 50% price change
        
        return normalized
    
    def calculate_prices(self, data: Dict) -> np.ndarray:
        """
        Calculate prices based on demand factors.
        
        Args:
            data: Dictionary containing parking lot data
            
        Returns:
            Array of new prices for each parking lot
        """
        # Calculate demand scores
        demand = self.calculate_demand(data)
        
        # Normalize demand
        normalized_demand = self.normalize_demand(demand)
        
        # Calculate prices
        prices = self.base_price * (1 + self.lambda_param * normalized_demand)
        
        # Apply bounds
        prices = np.clip(prices, self.min_price, self.max_price)
        
        return prices

class CompetitivePricingModel:
    """
    Competitive pricing model that considers geographic proximity and competitor prices.
    """
    
    def __init__(self, base_price: float = 10.0):
        self.base_price = base_price
        self.min_price = base_price * 0.5
        self.max_price = base_price * 2.0
        
        # Initialize demand-based model as foundation
        self.demand_model = DemandBasedPricingModel(base_price)
        
        # Competitive parameters
        self.proximity_threshold = 0.5  # km
        self.competition_weight = 0.3
        
        # Store historical prices for competitive analysis
        self.price_history = []
    
    def update_parameters(self, proximity_threshold: float, competition_weight: float):
        """Update competitive model parameters."""
        self.proximity_threshold = proximity_threshold
        self.competition_weight = competition_weight
    
    def find_competitors(self, lot_idx: int, data: Dict) -> List[int]:
        """
        Find nearby competing parking lots.
        
        Args:
            lot_idx: Index of the current parking lot
            data: Dictionary containing parking lot data
            
        Returns:
            List of competitor lot indices
        """
        competitors = []
        current_lat = data['latitude'][lot_idx]
        current_lon = data['longitude'][lot_idx]
        
        for i in range(len(data['latitude'])):
            if i != lot_idx:
                distance = calculate_distance(
                    current_lat, current_lon,
                    data['latitude'][i], data['longitude'][i]
                )
                
                if distance <= self.proximity_threshold:
                    competitors.append(i)
        
        return competitors
    
    def calculate_competitive_adjustment(self, lot_idx: int, base_price: float, 
                                       competitors: List[int], competitor_prices: np.ndarray,
                                       data: Dict) -> float:
        """
        Calculate competitive price adjustment.
        
        Args:
            lot_idx: Index of current parking lot
            base_price: Base price from demand model
            competitors: List of competitor indices
            competitor_prices: Current competitor prices
            data: Dictionary containing parking lot data
            
        Returns:
            Competitive adjustment factor
        """
        if not competitors:
            return 0.0
        
        # Calculate average competitor price
        avg_competitor_price = np.mean(competitor_prices[competitors])
        
        # Current lot occupancy rate
        current_occupancy_rate = data['occupancy'][lot_idx] / data['capacity'][lot_idx]
        
        # Competitive logic
        price_difference = avg_competitor_price - base_price
        
        # If current lot is highly occupied and competitors are cheaper
        if current_occupancy_rate > 0.8 and price_difference < 0:
            # Reduce price to be competitive
            adjustment = self.competition_weight * (price_difference / base_price)
        
        # If competitors are more expensive, we can increase price
        elif price_difference > 0:
            # Increase price but stay competitive
            adjustment = self.competition_weight * min(price_difference / base_price, 0.2)
        
        else:
            # Minimal adjustment
            adjustment = 0.1 * self.competition_weight * (price_difference / base_price)
        
        return adjustment
    
    def suggest_rerouting(self, data: Dict, prices: np.ndarray) -> List[Dict]:
        """
        Suggest rerouting for overcrowded lots.
        
        Args:
            data: Dictionary containing parking lot data
            prices: Current prices
            
        Returns:
            List of rerouting suggestions
        """
        suggestions = []
        
        for i in range(len(data['occupancy'])):
            occupancy_rate = data['occupancy'][i] / data['capacity'][i]
            
            # If lot is overcrowded (>90% occupied) and has queue
            if occupancy_rate > 0.9 and data['queue_length'][i] > 0:
                # Find nearby alternatives
                competitors = self.find_competitors(i, data)
                
                # Filter for less crowded alternatives
                alternatives = []
                for comp in competitors:
                    comp_occupancy_rate = data['occupancy'][comp] / data['capacity'][comp]
                    if comp_occupancy_rate < 0.7:  # Less than 70% occupied
                        alternatives.append({
                            'lot_id': comp,
                            'occupancy_rate': comp_occupancy_rate,
                            'price': prices[comp],
                            'distance': calculate_distance(
                                data['latitude'][i], data['longitude'][i],
                                data['latitude'][comp], data['longitude'][comp]
                            )
                        })
                
                # Sort by price and occupancy
                alternatives.sort(key=lambda x: (x['price'], x['occupancy_rate']))
                
                if alternatives:
                    suggestions.append({
                        'overcrowded_lot': i,
                        'alternatives': alternatives[:3]  # Top 3 alternatives
                    })
        
        return suggestions
    
    def calculate_prices(self, data: Dict) -> np.ndarray:
        """
        Calculate competitive prices.
        
        Args:
            data: Dictionary containing parking lot data
            
        Returns:
            Array of competitive prices
        """
        # Start with demand-based prices
        base_prices = self.demand_model.calculate_prices(data)
        
        # Apply competitive adjustments
        competitive_prices = base_prices.copy()
        
        for i in range(len(competitive_prices)):
            # Find competitors
            competitors = self.find_competitors(i, data)
            
            # Calculate competitive adjustment
            adjustment = self.calculate_competitive_adjustment(
                i, base_prices[i], competitors, base_prices, data
            )
            
            # Apply adjustment
            competitive_prices[i] = base_prices[i] * (1 + adjustment)
        
        # Apply bounds
        competitive_prices = np.clip(competitive_prices, self.min_price, self.max_price)
        
        # Store prices for history
        self.price_history.append(competitive_prices.copy())
        
        # Keep only recent history
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
        
        return competitive_prices
