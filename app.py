import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
from models.pricing_models import BaselinePricingModel, DemandBasedPricingModel, CompetitivePricingModel
from utils.data_generator import ParkingDataGenerator
from utils.real_data_loader import RealDataLoader
from utils.geographic_utils import calculate_distance
from visualization.dashboard import PricingDashboard

# Page configuration
st.set_page_config(
    page_title="Dynamic Parking Pricing System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.data_generator = ParkingDataGenerator()
    st.session_state.real_data_loader = RealDataLoader()
    st.session_state.dashboard = PricingDashboard()
    st.session_state.current_data = None
    st.session_state.price_history = []
    st.session_state.simulation_running = False
    st.session_state.current_time_step = 0
    st.session_state.use_real_data = False
    st.session_state.models_initialized = False

def main():
    st.title("🚗 Dynamic Parking Pricing System")
    st.markdown("Real-time pricing optimization for urban parking lots")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Control Panel")
        
        # Data source selection
        data_source = st.selectbox(
            "Select Data Source",
            ["Real Dataset", "Simulated Data"],
            key="data_source"
        )
        
        st.session_state.use_real_data = (data_source == "Real Dataset")
        
        # Initialize models based on data source
        if not st.session_state.models_initialized or st.session_state.get('last_data_source') != data_source:
            if st.session_state.use_real_data:
                num_lots = len(st.session_state.real_data_loader.get_lot_names())
            else:
                num_lots = 14  # Default for simulated data
            
            st.session_state.baseline_model = BaselinePricingModel(num_lots=num_lots)
            st.session_state.demand_model = DemandBasedPricingModel()
            st.session_state.competitive_model = CompetitivePricingModel()
            st.session_state.models_initialized = True
            st.session_state.last_data_source = data_source
        
        # Model selection
        selected_model = st.selectbox(
            "Select Pricing Model",
            ["Baseline Linear", "Demand-Based", "Competitive"],
            key="model_selection"
        )
        
        # Show dataset info if using real data
        if st.session_state.use_real_data:
            st.subheader("Dataset Information")
            stats = st.session_state.real_data_loader.get_data_statistics()
            st.write(f"**Total Records:** {stats['total_records']}")
            st.write(f"**Unique Lots:** {stats['unique_lots']}")
            st.write(f"**Date Range:** {stats['date_range']['start'].strftime('%Y-%m-%d')} to {stats['date_range']['end'].strftime('%Y-%m-%d')}")
            
            # Show vehicle distribution
            st.write("**Vehicle Types:**")
            for vtype, count in stats['vehicle_type_distribution'].items():
                st.write(f"  - {vtype}: {count}")
        
        # Simulation controls
        st.subheader("Simulation Controls")
        
        if st.button("Start Simulation", key="start_sim"):
            st.session_state.simulation_running = True
            st.session_state.current_time_step = 0
            st.session_state.price_history = []
            if st.session_state.use_real_data:
                st.session_state.real_data_loader.reset_simulation()
        
        if st.button("Stop Simulation", key="stop_sim"):
            st.session_state.simulation_running = False
        
        if st.button("Reset Simulation", key="reset_sim"):
            st.session_state.simulation_running = False
            st.session_state.current_time_step = 0
            st.session_state.price_history = []
            st.session_state.current_data = None
            if st.session_state.use_real_data:
                st.session_state.real_data_loader.reset_simulation()
        
        # Model parameters
        st.subheader("Model Parameters")
        
        if selected_model == "Baseline Linear":
            alpha = st.slider("Alpha (Price Sensitivity)", 0.1, 2.0, 0.5, 0.1)
            st.session_state.baseline_model.alpha = alpha
        
        elif selected_model == "Demand-Based":
            alpha = st.slider("Occupancy Weight", 0.1, 2.0, 1.0, 0.1)
            beta = st.slider("Queue Weight", 0.1, 2.0, 0.5, 0.1)
            gamma = st.slider("Traffic Weight", 0.1, 2.0, 0.3, 0.1)
            delta = st.slider("Special Day Weight", 0.1, 2.0, 0.4, 0.1)
            epsilon = st.slider("Vehicle Type Weight", 0.1, 2.0, 0.2, 0.1)
            
            st.session_state.demand_model.update_parameters(alpha, beta, gamma, delta, epsilon)
        
        elif selected_model == "Competitive":
            proximity_threshold = st.slider("Proximity Threshold (km)", 0.1, 2.0, 0.5, 0.1)
            competition_weight = st.slider("Competition Weight", 0.1, 2.0, 0.3, 0.1)
            
            st.session_state.competitive_model.update_parameters(proximity_threshold, competition_weight)
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (every 2 seconds)", value=True)
        
        # Manual refresh button
        if st.button("Refresh Data"):
            st.rerun()
    
    # Main content area
    if st.session_state.simulation_running:
        # Generate new data for current time step
        if st.session_state.use_real_data:
            current_data = st.session_state.real_data_loader.get_next_timestep_data()
        else:
            current_data = st.session_state.data_generator.generate_timestep_data(
                st.session_state.current_time_step
            )
        st.session_state.current_data = current_data
        
        # Calculate prices based on selected model
        if selected_model == "Baseline Linear":
            prices = st.session_state.baseline_model.calculate_prices(current_data)
        elif selected_model == "Demand-Based":
            prices = st.session_state.demand_model.calculate_prices(current_data)
        else:  # Competitive
            prices = st.session_state.competitive_model.calculate_prices(current_data)
        
        # Update price history
        if st.session_state.use_real_data:
            timestamp = current_data['timestamp']
        else:
            timestamp = datetime.now() - timedelta(seconds=st.session_state.current_time_step * 30)
        
        price_record = {
            'timestamp': timestamp,
            'prices': prices.copy(),
            'occupancy': current_data['occupancy'].copy(),
            'queue_length': current_data['queue_length'].copy()
        }
        
        # Add lot names if available
        if 'lot_names' in current_data:
            price_record['lot_names'] = current_data['lot_names']
        st.session_state.price_history.append(price_record)
        
        # Keep only last 50 records for performance
        if len(st.session_state.price_history) > 50:
            st.session_state.price_history = st.session_state.price_history[-50:]
        
        st.session_state.current_time_step += 1
        
        # Display current status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Time Step", st.session_state.current_time_step)
        
        with col2:
            avg_price = np.mean(prices)
            st.metric("Average Price", f"${avg_price:.2f}")
        
        with col3:
            total_occupancy = np.sum(current_data['occupancy'])
            total_capacity = np.sum(current_data['capacity'])
            occupancy_rate = (total_occupancy / total_capacity) * 100
            st.metric("Overall Occupancy", f"{occupancy_rate:.1f}%")
        
        # Display real-time visualizations
        st.session_state.dashboard.display_realtime_dashboard(
            st.session_state.current_data,
            prices,
            st.session_state.price_history,
            selected_model
        )
        
        # Auto-refresh logic
        if auto_refresh and st.session_state.simulation_running:
            time.sleep(2)
            st.rerun()
    
    else:
        # Display welcome message and instructions
        st.info("Click 'Start Simulation' to begin real-time pricing simulation")
        
        # Show sample data structure
        if st.session_state.use_real_data:
            st.subheader("Real Dataset Sample")
            
            # Show sample real data
            historical_data = st.session_state.real_data_loader.get_historical_data(limit=20)
            st.dataframe(historical_data, use_container_width=True)
            
            # Show parking lot information
            st.subheader("Parking Lot Information")
            lot_info = st.session_state.real_data_loader.get_lot_info()
            
            lot_info_df = pd.DataFrame([
                {
                    'Lot_Name': info['name'],
                    'Capacity': info['capacity'],
                    'Latitude': info['latitude'],
                    'Longitude': info['longitude']
                }
                for info in lot_info.values()
            ])
            
            st.dataframe(lot_info_df, use_container_width=True)
        else:
            st.subheader("Simulated Parking Lot Data Structure")
            sample_data = st.session_state.data_generator.generate_timestep_data(0)
            
            # Create sample dataframe for display
            sample_df = pd.DataFrame({
                'Lot_ID': range(1, 15),
                'Capacity': sample_data['capacity'],
                'Occupancy': sample_data['occupancy'],
                'Queue_Length': sample_data['queue_length'],
                'Traffic_Level': sample_data['traffic_level'],
                'Special_Day': sample_data['special_day'],
                'Vehicle_Type': sample_data['vehicle_type'],
                'Latitude': sample_data['latitude'],
                'Longitude': sample_data['longitude']
            })
            
            st.dataframe(sample_df, use_container_width=True)
        
        # Display model descriptions
        st.subheader("Pricing Models")
        
        tab1, tab2, tab3 = st.tabs(["Baseline Linear", "Demand-Based", "Competitive"])
        
        with tab1:
            st.markdown("""
            **Baseline Linear Model**
            
            A simple model where the price is adjusted based on occupancy rate:
            
            ```
            Price(t+1) = Price(t) + α × (Occupancy / Capacity)
            ```
            
            - **α**: Price sensitivity parameter
            - Simple and interpretable
            - Acts as a baseline for comparison
            """)
        
        with tab2:
            st.markdown("""
            **Demand-Based Model**
            
            Advanced model considering multiple demand factors:
            
            ```
            Demand = α×(Occupancy/Capacity) + β×QueueLength - γ×Traffic + δ×SpecialDay + ε×VehicleType
            Price = BasePrice × (1 + λ × NormalizedDemand)
            ```
            
            - **Multiple factors**: Occupancy, queue, traffic, events, vehicle type
            - **Bounded prices**: 0.5× to 2× base price
            - **Smooth adjustments**: Normalized demand function
            """)
        
        with tab3:
            st.markdown("""
            **Competitive Model**
            
            Incorporates geographic proximity and competitor pricing:
            
            ```
            CompetitorInfluence = f(distance, competitor_prices)
            Price = DemandBasedPrice × (1 + CompetitorInfluence)
            ```
            
            - **Geographic awareness**: Uses lat/long coordinates
            - **Competitor analysis**: Adjusts based on nearby lot prices
            - **Smart routing**: Suggests alternatives when overcrowded
            """)

if __name__ == "__main__":
    main()
    st.set_page_config(page_title="Urban Parking System")
