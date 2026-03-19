print("--- DEBUG: Script Started ---")

try:
    from preprocess import load_preprocess
    from feature_engineering import feature_engineering
    from models import train_model
    from model_evaluation import evaluate_model
    print("--- DEBUG: Imports Successful ---")
except ImportError as e:
    print(f"--- DEBUG: Import Error! {e} ---")

def run():
    print("--- DEBUG: Entering run() function ---")
    
    # 1. Load
    df = load_preprocess("data/food_delivery_eta_5000.csv")
    
    # 2. Engineering
    X_train, X_test, y_train, y_test = feature_engineering(df)
    
    # 3. Train
    models_dict = train_model(X_train, y_train) 
    print(f"--- DEBUG: {len(models_dict)} Models Trained ---")
    
    # 4. Evaluate
    results = evaluate_model(models_dict, X_test, y_test)

    # Automatically find the model name with the lowest MAE
    best_model_name = min(results, key=results.get)
    print(f"\n The winner is: {best_model_name}")

    # Save the actual model object
    from models import save_best_model
    save_best_model(models_dict[best_model_name])

    print("--- DEBUG: Pipeline Finished Successfully ---")

if __name__ == "__main__":
    run()
