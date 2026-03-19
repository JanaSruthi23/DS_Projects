from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_model(X_train, y_train): 

     # Define the models in a dictionary
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    # Loop through and train each one
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    # 2. Tuned XGBoost (The "Specialist")
    print("Tuning XGBoost hyperparameters...")
    xgb = XGBRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    
    # Add the best version of XGBoost to our dictionary
    models["XGBoost_Tuned"] = grid_search.best_estimator_
    
    print(f"Best XGBoost Params: {grid_search.best_params_}")
    
    return models

def save_best_model(model, filename="best_delivery_model.pkl"):
    """
    Saves the trained model object to the models/ folder.
    """
    # Create a 'models' directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    path = os.path.join('models', filename)
    joblib.dump(model, path)
    print(f"--- SUCCESS: Best model saved to {path} ---")