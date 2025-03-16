import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Request Body Schema
class InputData(BaseModel):
    features: list  # Expecting a list of numerical feature values

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input to numpy array
        input_array = np.array(data.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)[0]
        return {"prediction": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI Model Prediction Service is Running!"}
