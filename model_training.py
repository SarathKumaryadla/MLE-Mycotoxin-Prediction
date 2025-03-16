import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Setup Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "model_training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("🔄 Starting Model Training...")

# Set paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data/processed")
MODEL_DIR = os.path.join(CURRENT_DIR, "models")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_baseline_model(X_train, y_train):
    """Trains an optimized MLP Regressor with hyperparameter tuning."""
    try:
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)
        logging.info("✅ MLP Regressor trained successfully.")
        return model
    except Exception as e:
        logging.error(f"❌ Error in training MLP Regressor: {str(e)}")
        raise

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3):
    """Trains a Random Forest Regressor with regularization to prevent overfitting."""
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        logging.info("✅ Random Forest Regressor trained successfully.")
        return model
    except Exception as e:
        logging.error(f"❌ Error in training Random Forest Regressor: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load preprocessed data
        X_train_path = os.path.join(DATA_DIR, "X_train.csv")
        y_train_path = os.path.join(DATA_DIR, "y_train.csv")

        if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
            logging.error(f"❌ Preprocessed data files missing in {DATA_DIR}")
            raise FileNotFoundError(f"❌ Preprocessed data files missing in {DATA_DIR}")

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path).values.ravel()
        logging.info(f"✅ Preprocessed data loaded: {X_train.shape[0]} rows, {X_train.shape[1]} features.")

        print("Training Optimized Neural Network (MLP Regressor)...")
        mlp_model = train_baseline_model(X_train, y_train)

        print("\nTraining Random Forest Regressor with Regularization...")
        rf_model = train_random_forest(X_train, y_train)

        # Save models
        joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
        joblib.dump(mlp_model, os.path.join(MODEL_DIR, "mlp_model.pkl"))

        logging.info(f"✅ Models trained and saved in {MODEL_DIR}")
        print(f"\n✅ Models trained and saved in {MODEL_DIR}")

    except Exception as e:
        logging.error(f"❌ Error in model training pipeline: {str(e)}")
        print(f"❌ ERROR: {e}")

