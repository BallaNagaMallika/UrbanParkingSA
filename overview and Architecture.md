# Dynamic Parking Pricing System

## Overview

This is a Python-based dynamic pricing system for urban parking lots that simulates real-time pricing optimization. The system uses Streamlit for the web interface and implements three different pricing models of increasing complexity: Baseline Linear, Demand-Based, and Competitive pricing models. The application simulates 14 parking lots with realistic data patterns and provides interactive visualizations for monitoring pricing decisions.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Single-page application with interactive dashboard
- **Real-time Updates**: Session state management for continuous data simulation
- **Interactive Visualizations**: Plotly graphs and Folium maps for data visualization
- **Tabbed Interface**: Organized view with Price Trends, Occupancy Analysis, Geographic View, and Model Analysis

### Backend Architecture
- **Model-Based Pricing**: Three distinct pricing models with increasing complexity
- **Data Simulation**: Realistic parking data generation with time-based patterns
- **Geographic Calculations**: Distance-based calculations for competitive pricing
- **State Management**: Session-based state tracking for continuous simulation

### Data Processing
- **NumPy/Pandas**: Core data manipulation and mathematical operations
- **Real-time Simulation**: Thread-based data generation and price updates
- **Historical Tracking**: Price history storage for trend analysis

## Key Components

### Pricing Models (`models/pricing_models.py`)
1. **BaselinePricingModel**: Simple linear pricing based on occupancy rates
   - Formula: Price(t+1) = Price(t) + α × (Occupancy / Capacity)
   - Configurable parameters: base_price, alpha coefficient
   - Price bounds: 50% to 200% of base price

2. **DemandBasedPricingModel**: Advanced model incorporating multiple demand factors
   - Factors: Occupancy, Queue Length, Traffic, Special Events, Vehicle Type
   - More sophisticated demand calculation algorithm

3. **CompetitivePricingModel**: Market-aware pricing considering competitor prices
   - Geographic proximity analysis
   - Competitive positioning strategy

### Data Generation (`utils/data_generator.py`)
- **ParkingDataGenerator**: Simulates realistic parking lot data
- **Static Characteristics**: Lot capacities, geographic coordinates
- **Dynamic Patterns**: Time-based occupancy patterns, special events
- **Realistic Simulation**: 14 parking lots with varying characteristics

### Geographic Utilities (`utils/geographic_utils.py`)
- **Haversine Distance Calculation**: Accurate distance calculations between lots
- **Proximity Analysis**: Finding nearby competitors for pricing decisions
- **Coordinate Management**: Realistic urban grid simulation

### Visualization (`visualization/dashboard.py`)
- **PricingDashboard**: Comprehensive dashboard with multiple views
- **Real-time Metrics**: Current status indicators
- **Interactive Charts**: Price trends, occupancy correlations
- **Geographic Maps**: Folium-based location visualization
- **Model Analysis**: Comparative performance metrics

## Data Flow

1. **Data Generation**: ParkingDataGenerator creates realistic parking lot data
2. **Model Processing**: Selected pricing model processes current data
3. **Price Calculation**: New prices calculated based on model logic
4. **State Update**: Session state updated with new prices and data
5. **Visualization**: Dashboard displays updated information
6. **Historical Tracking**: Price history maintained for trend analysis

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **folium**: Geographic mapping
- **streamlit-folium**: Folium integration for Streamlit

### Standard Libraries
- **datetime**: Time-based calculations
- **threading**: Concurrent execution for real-time updates
- **time**: Simulation timing control
- **typing**: Type hints for better code documentation

## Deployment Strategy

### Development Setup
- Single-file application entry point (`app.py`)
- Modular architecture with clear separation of concerns
- Session state management for multi-user support

### Production Considerations
- Streamlit sharing/cloud deployment ready
- No external database dependencies (uses in-memory storage)
- Configurable parameters for different environments

### Scalability
- Modular design allows easy addition of new pricing models
- Geographic utilities support expansion to more parking lots
- Data generator can be extended for more complex scenarios

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- July 07, 2025. Initial setup
- July 08, 2025. Integrated real parking dataset with 500 records from 14 unique parking lots, added data source selection (Real Dataset vs Simulated Data), updated models to handle variable number of parking lots, enhanced dashboard to display real lot names and data
