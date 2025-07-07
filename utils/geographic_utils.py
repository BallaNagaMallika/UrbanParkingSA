import numpy as np
from typing import Tuple

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the earth.
    Uses the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in decimal degrees
        lat2, lon2: Latitude and longitude of second point in decimal degrees
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
    
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

def find_nearby_lots(target_lat: float, target_lon: float, 
                    lot_coordinates: list, threshold_km: float = 1.0) -> list:
    """
    Find parking lots within a specified distance.
    
    Args:
        target_lat: Target latitude
        target_lon: Target longitude
        lot_coordinates: List of (lat, lon) tuples for all lots
        threshold_km: Maximum distance in kilometers
        
    Returns:
        List of indices of nearby lots
    """
    nearby_lots = []
    
    for i, (lat, lon) in enumerate(lot_coordinates):
        distance = calculate_distance(target_lat, target_lon, lat, lon)
        if distance <= threshold_km:
            nearby_lots.append(i)
    
    return nearby_lots

def create_distance_matrix(coordinates: list) -> np.ndarray:
    """
    Create a distance matrix between all parking lots.
    
    Args:
        coordinates: List of (lat, lon) tuples
        
    Returns:
        NxN distance matrix where N is number of lots
    """
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            
            distance = calculate_distance(lat1, lon1, lat2, lon2)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix
    
    return distance_matrix

def get_geographic_clusters(coordinates: list, max_distance: float = 0.5) -> list:
    """
    Group parking lots into geographic clusters.
    
    Args:
        coordinates: List of (lat, lon) tuples
        max_distance: Maximum distance within a cluster
        
    Returns:
        List of clusters, each containing lot indices
    """
    n = len(coordinates)
    distance_matrix = create_distance_matrix(coordinates)
    
    # Simple clustering based on distance threshold
    clusters = []
    assigned = set()
    
    for i in range(n):
        if i in assigned:
            continue
            
        # Start new cluster
        cluster = [i]
        assigned.add(i)
        
        # Find all lots within threshold distance
        for j in range(n):
            if j not in assigned and distance_matrix[i, j] <= max_distance:
                cluster.append(j)
                assigned.add(j)
        
        clusters.append(cluster)
    
    return clusters

def calculate_centroid(coordinates: list) -> Tuple[float, float]:
    """
    Calculate the centroid of a group of coordinates.
    
    Args:
        coordinates: List of (lat, lon) tuples
        
    Returns:
        Tuple of (centroid_lat, centroid_lon)
    """
    if not coordinates:
        return 0.0, 0.0
    
    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]
    
    centroid_lat = sum(lats) / len(lats)
    centroid_lon = sum(lons) / len(lons)
    
    return centroid_lat, centroid_lon
