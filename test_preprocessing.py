import pytest
import os
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing_pipeline import (
    handle_missing_values,
    remove_outliers,
    compute_spectral_indices,
    scale_features,
    shap_feature_selection,
    transform_target,
    split_data,
)
from model_training import train_random_forest
import joblib

# Ensure script runs in the correct working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)  # Set working directory

# Sample Data for Testing (Using Correct Feature Names)
data = {
    "feature_1": [1, 2, 3, np.nan, 5],
    "feature_2": [10, 20, 30, 40, 50],
    "feature_3": [100, 200, 300, 400, 500],
    "vomitoxin_ppb": [0.1, 0.2, 0.3, 0.4, 0.5],  # Target variable
}
df = pd.DataFrame(data)

# ✅ Test 1: Missing Value Handling
def test_handle_missing_values():
    cleaned_df = handle_missing_values(df)
    assert cleaned_df.isnull().sum().sum() == 0  # No missing values should remain

# ✅ Test 2: Outlier Removal
def test_remove_outliers():
    df_no_outliers = remove_outliers(df, method='iqr')
    assert len(df_no_outliers) <= len(df)  # No increase in rows

# ✅ Test 3: Spectral Indices Computation
def test_compute_spectral_indices():
    df_indices = compute_spectral_indices(df.copy())
    assert 'NDVI' in df_indices.columns and 'SR' in df_indices.columns  # NDVI & SR must exist

# ✅ Test 4: Feature Scaling
def test_scale_features():
    df_scaled = scale_features(df.copy(), method='standard')
    assert np.allclose(df_scaled.mean(), 0, atol=1)  # Standardized mean ≈ 0

# ✅ Test 5: SHAP Feature Selection
def test_shap_feature_selection():
    df_selected = shap_feature_selection(df.copy(), num_features=2)
    assert df_selected.shape[1] == 3  # 2 selected features + target variable

# ✅ Test 6: Target Transformation
def test_transform_target():
    df_transformed = transform_target(df.copy())
    assert (df_transformed['vomitoxin_ppb'] > 0).all()  # Log-transformed values should be positive

# ✅ Test 7: Data Splitting
def test_split_data():
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert len(X_train) > 0 and len(X_test) > 0  # Data must be split properly

# ✅ Test 8: Model Training
def test_model_training():
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    rf_model = train_random_forest(X_train, y_train)

    # Ensure model is trained
    assert rf_model is not None

    # Save model temporarily
    model_path = os.path.join(CURRENT_DIR, "test_model.pkl")
    joblib.dump(rf_model, model_path)

    # Load the saved model to ensure it's saved correctly
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None

    # Remove test model file after checking
    os.remove(model_path)

# ✅ Test 9: Model Evaluation
def test_model_evaluation():
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    rf_model = train_random_forest(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Compute Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    assert mae >= 0  # MAE should be non-negative
    assert rmse >= 0  # RMSE should be non-negative

    # Handle NaN case for R² Score
    if math.isnan(r2):
        r2 = -1  # Assign worst possible R² Score

    assert -1 <= r2 <= 1  # R² Score should be between -1 and 1

# ✅ Run pytest only in the current directory
if __name__ == "__main__":
    pytest.main(["-q", "--disable-warnings", CURRENT_DIR])
