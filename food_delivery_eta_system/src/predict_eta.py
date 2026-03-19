import joblib
import pandas as pd
from feature_engineering import feature_engineering # Import your existing logic

# 1. Load the model
model = joblib.load('models/best_delivery_model.pkl')

def get_delivery_eta(raw_data):
    # Convert raw input to a DataFrame
    df_raw = pd.DataFrame([raw_data])
    
    # --- IMPORTANT FIX ---
    # We must transform the raw input exactly like we did the training data.
    # We use a trick: create a dummy DF with all zeros for missing columns 
    # OR better yet, ensure your feature_engineering function handles single rows.
    
    # Get the list of features the model expects
    expected_features = model.feature_names_in_
    
    # Create the input matching those exact columns, filling missing with 0
    df_input = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # Fill in the values we actually have
    for column in raw_data:
        if column in df_input.columns:
            df_input[column] = raw_data[column]
            
    # Handle the specific One-Hot encoded columns (e.g., if city is 'Chennai')
    if 'city' in raw_data:
        city_col = f"city_{raw_data['city']}"
        if city_col in df_input.columns:
            df_input[city_col] = 1
            
    if 'vehicle_type' in raw_data:
        vehicle_col = f"vehicle_type_{raw_data['vehicle_type']}"
        if vehicle_col in df_input.columns:
            df_input[vehicle_col] = 1

    # 2. Predict
    prediction = model.predict(df_input)
    return round(prediction[0], 2)

# Now your order dictionary can be simple and "human-readable"
new_order = {
    'distance_km': 5.2,
    'hour': 19,
    'day': 4,
    'is_peak': 1,
    'is_weekend': 0,
    'traffic_level': 3,
    'weather': 1,
    'distance_traffic': 15.6,
    'distance_weather': 5.2,
    'city': 'Chennai',      # The code above will turn this into city_Chennai=1
    'vehicle_type': 'Car'   # The code above will turn this into vehicle_type_Car=1
}

eta = get_delivery_eta(new_order)
print(f"Estimated Delivery Time: {eta} minutes")