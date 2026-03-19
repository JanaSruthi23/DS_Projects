import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="FoodieExpress ETA Predictor", page_icon="🚴")
st.title("🚴 Food Delivery ETA Predictor")
st.markdown("Enter order details below to estimate delivery time.")

# 2. Load the Model
@st.cache_resource 
def load_model():
    return joblib.load('models/best_delivery_model.pkl')

model = load_model()
expected_features = model.feature_names_in_

# 3. Create Sidebar for Inputs
st.sidebar.header("Order Details")

distance = st.sidebar.slider("Distance (km)", 0.5, 20.0, 5.0)
traffic = st.sidebar.select_slider("Traffic Level", options=[1, 2, 3, 4, 5], value=2)
weather = st.sidebar.selectbox("Weather Condition", options=[0, 1, 2], format_func=lambda x: ["Sunny", "Rainy", "Stormy"][x])

col1, col2 = st.columns(2)
with col1:
    hour = st.number_input("Hour (0-23)", 0, 23, 19)
    city = st.selectbox("City", ["Mumbai", "Delhi", "Chennai", "Hyderabad"])
with col2:
    day = st.number_input("Day of Week (0-6)", 0, 6, 4)
    vehicle = st.selectbox("Vehicle Type", ["Scooter", "Bike", "Car"])

# 4. Prepare the Data for Prediction
if st.button("Predict Delivery Time"):
    # Create the same 0-filled DataFrame your model expects
    input_df = pd.DataFrame(0, index=[0], columns=expected_features)
    # Add missing derived features in app.py
    input_df['is_peak'] = 1 if hour in [12, 13, 14, 19, 20, 21] else 0
    input_df['is_weekend'] = 1 if day >= 5 else 0
    
    # Fill basic features
    input_df['distance_km'] = distance
    input_df['hour'] = hour
    input_df['day'] = day
    input_df['traffic_level'] = traffic
    input_df['weather'] = weather
    
    # Interaction terms
    input_df['distance_traffic'] = distance * traffic
    input_df['distance_weather'] = distance * weather
    
    # One-Hot Encoding for City and Vehicle
    if f"city_{city}" in expected_features:
        input_df[f"city_{city}"] = 1
    if f"vehicle_type_{vehicle}" in expected_features:
        input_df[f"vehicle_type_{vehicle}"] = 1

    # 5. Make Prediction
    prediction = model.predict(input_df)[0]
    
    # Display Result
    st.success(f"### Estimated Delivery Time: {prediction:.2f} minutes")
    
    # Add a fun visual
    st.progress(min(int(prediction), 100))