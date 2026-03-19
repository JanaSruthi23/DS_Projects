import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def plot_feature_importance(model, feat_names, model_name):

    plt.figure(figsize=(10, 6))
    
    # Check if the model is tree-based (RF/XGB) or linear
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_) # Use absolute value of coefficients
    else:
        print(f"Feature importance not available for {model_name}")
        return

    # Sort the features by importance
    indices = np.argsort(importances)

    plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [feat_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title(f'Feature Importance: {model_name}')
    plt.tight_layout()
    plt.savefig("outputs/Feature_importance.png")
    plt.close()

def evaluate_model(models_dict, X_test, y_test):
    n, k = X_test.shape[0], X_test.shape[1]
    results = {}

    print("\n" + "="*40)
    print(f"{'FINAL MODEL PERFORMANCE':^40}")
    print("="*40)

    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
        
        avg_bias = np.mean(y_test - y_pred)
        
        print(f"\nModel: {name}")
        print(f"MAE:         {mae:.2f} minutes")
        print(f"Adjusted R2: {adj_r2:.4f}")
        print(f"Avg Bias:    {avg_bias:.2f} mins")
        print("-" * 30)

       # Important_feature
        plot_feature_importance(model, X_test.columns, name)
        
        # Store results so we can pick the winner in main.py
        results[name] = mae
    
    return results