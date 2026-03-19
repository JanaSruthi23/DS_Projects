from sklearn.model_selection import train_test_split
import pandas as pd

def feature_engineering(df):
    # Ensure we are working on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # --- Time-based features ---
    df['hour'] = df['order_time'].dt.hour
    df['day'] = df['order_time'].dt.dayofweek

    # Peak hour feature (Lunch: 12-14, Dinner: 19-21)
    df['is_peak'] = df['hour'].isin([12, 13, 14, 19, 20, 21]).astype(int)

    # Weekend feature (Saturday=5, Sunday=6)
    df['is_weekend'] = (df['day'] >= 5).astype(int)

    # --- Interaction Features ---
    # Map categorical levels to numeric for math operations
    df['traffic_level'] = df['traffic_level'].map({'Low': 1, 'Medium': 2, 'High': 3})
    df['weather'] = df['weather'].map({'Clear': 0, 'Rainy': 1, 'Stormy': 2})

    df['distance_traffic'] = df['distance_km'] * df['traffic_level']
    df['distance_weather'] = df['distance_km'] * df['weather']

    # --- Handling Outliers (IQR Method) ---
    q1 = df['delivery_duration'].quantile(0.25)
    q3 = df['delivery_duration'].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df = df[(df['delivery_duration'] >= lower_bound) & 
            (df['delivery_duration'] <= upper_bound)].copy()

    # --- Categorical Encoding ---
    df = pd.get_dummies(df, columns=['vehicle_type', 'city'], drop_first=True)

    # --- Define Input (X) and Output (y) ---
    drop_cols = [
        'order_id', 'delivery_person_id', 'restaurant_id',
        'order_time', 'delivery_time', 'delivery_duration'
    ]
    
    X = df.drop(columns=drop_cols)
    y = df['delivery_duration']

    # --- Splitting the data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test