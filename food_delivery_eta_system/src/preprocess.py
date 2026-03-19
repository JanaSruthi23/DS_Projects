import pandas as pd

def load_preprocess(file_path="data/food_delivery_eta_5000.csv"):
    """
    Loads food delivery data and calculates the delivery duration in minutes.
    """
    df = pd.read_csv(file_path)

    # Convert time columns to datetime objects
    df['order_time'] = pd.to_datetime(df['order_time'])
    df['delivery_time'] = pd.to_datetime(df['delivery_time'])

    # Calculate delivery duration in minutes
    df['delivery_duration'] = (
        (df['delivery_time'] - df['order_time']).dt.total_seconds() / 60
    )
    
    return df