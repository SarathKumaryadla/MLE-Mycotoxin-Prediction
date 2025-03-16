import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_baseline_model(X_train, y_train):
    """Trains an optimized MLP Regressor with hyperparameter tuning."""
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),  # Increased layer complexity
        activation='relu',
        solver='adam',
        alpha=0.001,  # Regularization to reduce overfitting
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3):
    """Trains a Random Forest Regressor with regularization to prevent overfitting."""
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, dataset_type="Test"):
    """Evaluates the model using MAE, RMSE, and R² Score."""
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\n{dataset_type} Set Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    return mae, rmse, r2

if __name__ == "__main__":
    # Load preprocessed data
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").values.ravel()
    y_test = pd.read_csv("y_test.csv").values.ravel()
    
    print("Preprocessed data loaded. Training the models...")
    
    print("Training Optimized Neural Network (MLP Regressor)...")
    mlp_model = train_baseline_model(X_train, y_train)
    evaluate_model(mlp_model, X_test, y_test, "Test")
    
    print("\nTraining Random Forest Regressor with Regularization...")
    rf_model = train_random_forest(X_train, y_train)
    joblib.dump(rf_model, "random_forest_model.pkl")  # Save model
    joblib.dump(mlp_model, "mlp_model.pkl")  # Save MLP model

    
    print("\nChecking for Overfitting...")
    evaluate_model(rf_model, X_train, y_train, "Training")
    evaluate_model(rf_model, X_test, y_test, "Test")
    mlp_mae, mlp_rmse, mlp_r2 = evaluate_model(mlp_model, X_test, y_test)
    rf_mae, rf_rmse, rf_r2 = evaluate_model(rf_model, X_test, y_test)

    print("\nSummary of Model Performance:")
    print(f"{'Model':<25}{'MAE':<10}{'RMSE':<10}{'R² Score':<10}")
    print(f"{'MLP Regressor (Test)':<25}{mlp_mae:.4f} {mlp_rmse:.4f} {mlp_r2:.4f}")
    print(f"{'Random Forest (Test)':<25}{rf_mae:.4f} {rf_rmse:.4f} {rf_r2:.4f}")


