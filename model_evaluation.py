import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "model_evaluation.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("🔄 Starting Model Evaluation...")

# Set paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "models")
DATA_DIR = os.path.join(CURRENT_DIR, "data/processed")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    # Load trained model
    MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")  # Update if using MLP
    if not os.path.exists(MODEL_PATH):
        logging.error(f"❌ Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    logging.info("✅ Model loaded successfully.")

    # Load test data
    X_test_path = os.path.join(DATA_DIR, "X_test.csv")
    y_test_path = os.path.join(DATA_DIR, "y_test.csv")

    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        logging.error(f"❌ Test data files missing in {DATA_DIR}")
        raise FileNotFoundError(f"❌ Test data files missing in {DATA_DIR}")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    logging.info(f"✅ Test data loaded: {X_test.shape[0]} rows, {X_test.shape[1]} features.")

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    logging.info(f"📊 Model Evaluation Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

    # Save numerical results
    results = {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }

    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Evaluation completed. Results saved in {RESULTS_DIR}/evaluation_results.json")
    logging.info(f"✅ Evaluation results saved in {RESULTS_DIR}/evaluation_results.json")

    # -----------------  VISUAL EVALUATION ----------------- #

    # 1️⃣ Scatter Plot: Actual vs. Predicted
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor='black')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"))
    plt.show()
    logging.info("📊 Saved scatter plot of Actual vs. Predicted values.")

    # 2️⃣ Residual Analysis: Histogram
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=30, kde=True, color="blue")
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel("Prediction Error (Residuals)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.savefig(os.path.join(RESULTS_DIR, "residual_analysis.png"))
    plt.show()
    logging.info("📊 Saved residual analysis histogram.")

    # 3️⃣ Residuals vs. Predictions
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, edgecolor='black')
    plt.axhline(0, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Predicted Values")
    plt.savefig(os.path.join(RESULTS_DIR, "residuals_vs_predicted.png"))
    plt.show()
    logging.info("📊 Saved residuals vs. predicted values plot.")

    # -----------------  FEATURE IMPORTANCE (SHAP) ----------------- #

    # 4️⃣ SHAP Feature Importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(RESULTS_DIR, "shap_feature_importance.png"), bbox_inches="tight")
    plt.close()
    logging.info("📊 Saved SHAP feature importance plot.")

    print(f"\n✅ All visualizations saved in {RESULTS_DIR}")
    logging.info(f"✅ All evaluation visualizations saved in {RESULTS_DIR}")

except Exception as e:
    logging.error(f"❌ Error in model evaluation pipeline: {str(e)}")
    print(f"❌ ERROR: {e}")


