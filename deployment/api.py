import sys
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ✅ Fix Import Issue (Ensure API can find preprocessing pipeline)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing_pipeline import (
    handle_missing_values, remove_outliers, compute_spectral_indices, 
    scale_features, shap_feature_selection
)

# ✅ Load Model with Correct Path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_model.pkl")
model = joblib.load(MODEL_PATH)

# ✅ Load Expected Feature Names from Saved File (Ensures Consistency)
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "selected_features.txt")

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"❌ Feature names file not found: {FEATURES_PATH}")

with open(FEATURES_PATH, "r") as f:
    expected_features = [line.strip() for line in f.readlines()]

# ✅ Define Input Schema
class PredictionInput(BaseModel):
    hsi_id: str
    features: list[float]  # Must match the original dataset structure

# ✅ Initialize FastAPI
app = FastAPI()

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # 🔹 Convert input data to DataFrame (Ensure feature names are strings)
        df = pd.DataFrame([data.features], columns=[str(i) for i in range(len(data.features))])

        # 🔹 Apply Full Preprocessing Pipeline
        df = handle_missing_values(df)
        df = remove_outliers(df, method="iqr")
        df = compute_spectral_indices(df)
        df = scale_features(df, method="standard")
        df = shap_feature_selection(df, num_features=len(expected_features))  # Ensure consistency  

        # 🔹 Ensure Correct Features Before Prediction
        df = df.reindex(columns=expected_features, fill_value=0)  # Use the exact features from training

        # 🔹 Validate Feature Count Before Prediction
        if df.shape[1] != model.n_features_in_:
            raise ValueError(f"Model expects {model.n_features_in_} features, but received {df.shape[1]}")

        # 🔹 Make Prediction
        prediction = model.predict(df)[0]
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
