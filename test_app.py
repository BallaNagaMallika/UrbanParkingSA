import streamlit as st
import pandas as pd
import numpy as np

# Simple test to verify Streamlit works
st.title("🚗 Dynamic Parking Pricing System")
st.write("Testing application display...")

# Test data loading
try:
    data = pd.read_csv("dataset.csv")
    st.success(f"Successfully loaded {len(data)} records!")
    st.write("Sample data:")
    st.dataframe(data.head())
except Exception as e:
    st.error(f"Error loading data: {e}")

st.write("If you can see this, the application is working!")