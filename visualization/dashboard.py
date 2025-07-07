import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List
import folium
from streamlit_folium import st_folium

class PricingDashboard:
    """
    Interactive dashboard for visualizing parking lot pricing and data.
    """
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
        
    def display_realtime_dashboard(self, current_data: Dict, prices: np.ndarray, 
                                 price_history: List[Dict], model_name: str):
        """
        Display the main real-time dashboard.
        
        Args:
            current_data: Current parking lot data
            prices: Current prices for all lots
            price_history: Historical price data
            model_name: Name of the pricing model being used
        """
        st.subheader(f"Real-Time Dashboard - {model_name} Model")
        
        # Current status metrics
        self._display_current_metrics(current_data, prices)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Price Trends", "Occupancy vs Price", "Geographic View", "Model Analysis"
        ])
        
        with tab1:
            self._display_price_trends(price_history)
        
        with tab2:
            self._display_occupancy_price_correlation(current_data, prices)
        
        with tab3:
            self._display_geographic_view(current_data, prices)
        
        with tab4:
            self._display_model_analysis(current_data, prices, price_history, model_name)
    
    def _display_current_metrics(self, current_data: Dict, prices: np.ndarray):
        """Display current key metrics."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_capacity = np.sum(current_data['capacity'])
            total_occupancy = np.sum(current_data['occupancy'])
            occupancy_rate = (total_occupancy / total_capacity) * 100
            st.metric("System Occupancy", f"{occupancy_rate:.1f}%")
        
        with col2:
            avg_price = np.mean(prices)
            st.metric("Average Price", f"${avg_price:.2f}")
        
        with col3:
            max_price = np.max(prices)
            st.metric("Highest Price", f"${max_price:.2f}")
        
        with col4:
            min_price = np.min(prices)
            st.metric("Lowest Price", f"${min_price:.2f}")
        
        with col5:
            total_queue = np.sum(current_data['queue_length'])
            st.metric("Total Queue", f"{total_queue} vehicles")
    
    def _display_price_trends(self, price_history: List[Dict]):
        """Display price trends over time."""
        st.subheader("Price Trends Over Time")
        
        if len(price_history) < 2:
            st.info("Insufficient data for trends. Let the simulation run for a few more steps.")
            return
        
        # Create price trend chart
        fig = go.Figure()
        
        # Extract timestamps and prices
        timestamps = [record['timestamp'] for record in price_history]
        
        # Plot price trends for each lot
        for lot_id in range(len(price_history[0]['prices'])):
            prices = [record['prices'][lot_id] for record in price_history]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=prices,
                mode='lines+markers',
                name=f'Lot {lot_id + 1}',
                line=dict(color=self.colors[lot_id % len(self.colors)], width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Price Evolution by Parking Lot",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show price change statistics
        if len(price_history) >= 2:
            current_prices = price_history[-1]['prices']
            previous_prices = price_history[-2]['prices']
            price_changes = current_prices - previous_prices
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Recent Price Changes")
                for i, change in enumerate(price_changes):
                    direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                    st.write(f"Lot {i+1}: {direction} ${abs(change):.2f}")
            
            with col2:
                # Price change distribution
                fig_hist = go.Figure(data=[go.Histogram(
                    x=price_changes,
                    nbinsx=10,
                    name="Price Changes"
                )])
                
                fig_hist.update_layout(
                    title="Distribution of Price Changes",
                    xaxis_title="Price Change ($)",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
    
    def _display_occupancy_price_correlation(self, current_data: Dict, prices: np.ndarray):
        """Display occupancy vs price correlation."""
        st.subheader("Occupancy vs Price Analysis")
        
        # Calculate occupancy rates
        occupancy_rates = (current_data['occupancy'] / current_data['capacity']) * 100
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color by queue length
        fig.add_trace(go.Scatter(
            x=occupancy_rates,
            y=prices,
            mode='markers',
            marker=dict(
                size=15,
                color=current_data['queue_length'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Queue Length")
            ),
            text=[f"Lot {i+1}<br>Queue: {q}<br>Traffic: {t:.2f}" 
                  for i, (q, t) in enumerate(zip(current_data['queue_length'], 
                                                current_data['traffic_level']))],
            hovertemplate="<b>%{text}</b><br>" +
                         "Occupancy: %{x:.1f}%<br>" +
                         "Price: $%{y:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Price vs Occupancy Rate",
            xaxis_title="Occupancy Rate (%)",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation matrix
        col1, col2 = st.columns(2)
        
        with col1:
            # Create correlation data
            corr_data = pd.DataFrame({
                'Occupancy_Rate': occupancy_rates,
                'Price': prices,
                'Queue_Length': current_data['queue_length'],
                'Traffic_Level': current_data['traffic_level']
            })
            
            correlation_matrix = corr_data.corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale='RdBu_r'
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # Show lot details table
            st.subheader("Lot Details")
            
            lot_details = pd.DataFrame({
                'Lot_ID': range(1, len(prices) + 1),
                'Occupancy': current_data['occupancy'],
                'Capacity': current_data['capacity'],
                'Occupancy_Rate': occupancy_rates,
                'Price': prices,
                'Queue': current_data['queue_length'],
                'Traffic': current_data['traffic_level']
            })
            
            # Format the dataframe
            lot_details['Occupancy_Rate'] = lot_details['Occupancy_Rate'].apply(lambda x: f"{x:.1f}%")
            lot_details['Price'] = lot_details['Price'].apply(lambda x: f"${x:.2f}")
            lot_details['Traffic'] = lot_details['Traffic'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(lot_details, use_container_width=True, height=400)
    
    def _display_geographic_view(self, current_data: Dict, prices: np.ndarray):
        """Display geographic view of parking lots."""
        st.subheader("Geographic Distribution")
        
        # Create map
        center_lat = np.mean(current_data['latitude'])
        center_lon = np.mean(current_data['longitude'])
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
        # Add markers for each parking lot
        for i in range(len(current_data['latitude'])):
            # Color based on price
            price = prices[i]
            if price < 8:
                color = 'green'
            elif price < 12:
                color = 'orange'
            else:
                color = 'red'
            
            # Create popup text
            occupancy_rate = (current_data['occupancy'][i] / current_data['capacity'][i]) * 100
            popup_text = f"""
            <b>Lot {i+1}</b><br>
            Price: ${price:.2f}<br>
            Occupancy: {current_data['occupancy'][i]}/{current_data['capacity'][i]} ({occupancy_rate:.1f}%)<br>
            Queue: {current_data['queue_length'][i]} vehicles<br>
            Traffic: {current_data['traffic_level'][i]:.2f}
            """
            
            folium.Marker(
                location=[current_data['latitude'][i], current_data['longitude'][i]],
                popup=popup_text,
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)
        
        # Show geographic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution by Location")
            
            # Create price heatmap data
            price_data = pd.DataFrame({
                'Latitude': current_data['latitude'],
                'Longitude': current_data['longitude'],
                'Price': prices,
                'Lot_ID': range(1, len(prices) + 1)
            })
            
            fig_scatter = px.scatter_mapbox(
                price_data,
                lat='Latitude',
                lon='Longitude',
                size='Price',
                color='Price',
                hover_name='Lot_ID',
                hover_data=['Price'],
                color_continuous_scale='Viridis',
                size_max=20,
                zoom=13,
                mapbox_style="open-street-map"
            )
            
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.subheader("Distance Matrix")
            
            # Calculate distance matrix
            from utils.geographic_utils import create_distance_matrix
            coordinates = [(lat, lon) for lat, lon in zip(current_data['latitude'], 
                                                         current_data['longitude'])]
            distance_matrix = create_distance_matrix(coordinates)
            
            # Create heatmap
            fig_dist = px.imshow(
                distance_matrix,
                title="Distance Between Lots (km)",
                color_continuous_scale='Blues',
                aspect="auto"
            )
            
            fig_dist.update_layout(
                xaxis_title="Lot ID",
                yaxis_title="Lot ID",
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
    
    def _display_model_analysis(self, current_data: Dict, prices: np.ndarray, 
                               price_history: List[Dict], model_name: str):
        """Display model-specific analysis."""
        st.subheader(f"{model_name} Model Analysis")
        
        if model_name == "Baseline Linear":
            self._analyze_baseline_model(current_data, prices)
        elif model_name == "Demand-Based":
            self._analyze_demand_model(current_data, prices)
        elif model_name == "Competitive":
            self._analyze_competitive_model(current_data, prices)
        
        # Common analysis for all models
        self._display_price_statistics(prices, price_history)
    
    def _analyze_baseline_model(self, current_data: Dict, prices: np.ndarray):
        """Analyze baseline model performance."""
        st.write("**Baseline Linear Model Analysis**")
        
        occupancy_rates = current_data['occupancy'] / current_data['capacity']
        
        # Show linear relationship
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=occupancy_rates,
            y=prices,
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Current Prices'
        ))
        
        # Add trend line
        z = np.polyfit(occupancy_rates, prices, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 1, 100)
        y_trend = p(x_trend)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Linear Trend'
        ))
        
        fig.update_layout(
            title="Linear Relationship: Occupancy vs Price",
            xaxis_title="Occupancy Rate",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show model parameters
        st.write(f"**Model Equation:** Price = Base + α × Occupancy Rate")
        st.write(f"**Slope (α):** {z[0]:.2f}")
        st.write(f"**Intercept:** {z[1]:.2f}")
        st.write(f"**R-squared:** {np.corrcoef(occupancy_rates, prices)[0, 1]**2:.3f}")
    
    def _analyze_demand_model(self, current_data: Dict, prices: np.ndarray):
        """Analyze demand-based model performance."""
        st.write("**Demand-Based Model Analysis**")
        
        # Calculate demand components
        occupancy_component = current_data['occupancy'] / current_data['capacity']
        queue_component = current_data['queue_length']
        traffic_component = current_data['traffic_level']
        
        # Show demand components
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Occupancy Factor', 'Queue Factor', 'Traffic Factor', 'Final Prices'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        lot_ids = list(range(1, len(prices) + 1))
        
        fig.add_trace(go.Bar(x=lot_ids, y=occupancy_component, name='Occupancy'), row=1, col=1)
        fig.add_trace(go.Bar(x=lot_ids, y=queue_component, name='Queue'), row=1, col=2)
        fig.add_trace(go.Bar(x=lot_ids, y=traffic_component, name='Traffic'), row=2, col=1)
        fig.add_trace(go.Bar(x=lot_ids, y=prices, name='Price'), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Demand Components Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show feature importance
        st.write("**Feature Contribution to Pricing:**")
        
        # Calculate normalized contributions (simplified)
        total_demand = occupancy_component + queue_component * 0.5 + traffic_component * 0.3
        
        contrib_df = pd.DataFrame({
            'Lot_ID': lot_ids,
            'Occupancy_Contrib': occupancy_component / total_demand * 100,
            'Queue_Contrib': (queue_component * 0.5) / total_demand * 100,
            'Traffic_Contrib': (traffic_component * 0.3) / total_demand * 100
        })
        
        st.dataframe(contrib_df, use_container_width=True)
    
    def _analyze_competitive_model(self, current_data: Dict, prices: np.ndarray):
        """Analyze competitive model performance."""
        st.write("**Competitive Model Analysis**")
        
        # Show competitive analysis
        from utils.geographic_utils import create_distance_matrix
        
        coordinates = [(lat, lon) for lat, lon in zip(current_data['latitude'], 
                                                     current_data['longitude'])]
        distance_matrix = create_distance_matrix(coordinates)
        
        # Find competitors for each lot
        competitive_analysis = []
        
        for i in range(len(prices)):
            # Find lots within 0.5km
            competitors = []
            for j in range(len(prices)):
                if i != j and distance_matrix[i][j] <= 0.5:
                    competitors.append(j)
            
            if competitors:
                competitor_prices = [prices[j] for j in competitors]
                avg_competitor_price = np.mean(competitor_prices)
                price_advantage = prices[i] - avg_competitor_price
            else:
                avg_competitor_price = 0
                price_advantage = 0
            
            competitive_analysis.append({
                'Lot_ID': i + 1,
                'Own_Price': prices[i],
                'Competitors': len(competitors),
                'Avg_Competitor_Price': avg_competitor_price,
                'Price_Advantage': price_advantage
            })
        
        comp_df = pd.DataFrame(competitive_analysis)
        
        # Display competitive analysis
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=comp_df['Avg_Competitor_Price'],
            y=comp_df['Own_Price'],
            mode='markers',
            marker=dict(
                size=comp_df['Competitors'] * 3 + 5,
                color=comp_df['Price_Advantage'],
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title="Price Advantage")
            ),
            text=comp_df['Lot_ID'],
            hovertemplate="<b>Lot %{text}</b><br>" +
                         "Own Price: $%{y:.2f}<br>" +
                         "Competitor Avg: $%{x:.2f}<br>" +
                         "Competitors: %{marker.size}<extra></extra>"
        ))
        
        # Add diagonal line
        max_price = max(comp_df['Own_Price'].max(), comp_df['Avg_Competitor_Price'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_price],
            y=[0, max_price],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Price Parity Line'
        ))
        
        fig.update_layout(
            title="Competitive Pricing Analysis",
            xaxis_title="Average Competitor Price ($)",
            yaxis_title="Own Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show competitive summary
        st.write("**Competitive Summary:**")
        st.dataframe(comp_df, use_container_width=True)
    
    def _display_price_statistics(self, prices: np.ndarray, price_history: List[Dict]):
        """Display general price statistics."""
        st.subheader("Price Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Price Distribution:**")
            
            fig_hist = go.Figure(data=[go.Histogram(
                x=prices,
                nbinsx=10,
                marker_color='lightblue',
                opacity=0.7
            )])
            
            fig_hist.update_layout(
                title="Price Distribution",
                xaxis_title="Price ($)",
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.write("**Price Volatility:**")
            
            if len(price_history) >= 2:
                # Calculate price volatility
                price_changes = []
                for i in range(1, len(price_history)):
                    changes = price_history[i]['prices'] - price_history[i-1]['prices']
                    price_changes.extend(changes)
                
                volatility = np.std(price_changes)
                avg_change = np.mean(np.abs(price_changes))
                
                st.metric("Price Volatility (σ)", f"{volatility:.3f}")
                st.metric("Avg Absolute Change", f"${avg_change:.2f}")
                st.metric("Min Price", f"${np.min(prices):.2f}")
                st.metric("Max Price", f"${np.max(prices):.2f}")
            else:
                st.info("Need more data points for volatility analysis")
