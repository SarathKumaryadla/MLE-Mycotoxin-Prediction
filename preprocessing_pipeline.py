import pandas as pd
import numpy as np
import shap
import os
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Setup Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("🔄 Starting Preprocessing Pipeline...")

# Set paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data/processed")
RAW_DATA_PATH = os.path.join(CURRENT_DIR, "data/MLE-Assignment.csv")

# Ensure output directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def load_data():
    """Loads dataset from CSV."""
    if not os.path.exists(RAW_DATA_PATH):
        logging.error(f"❌ Data file not found: {RAW_DATA_PATH}")
        raise FileNotFoundError(f"❌ Data file not found: {RAW_DATA_PATH}")
    
    df = pd.read_csv(RAW_DATA_PATH)
    logging.info(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def handle_missing_values(df):
    """Handles missing values by dropping them."""
    df_cleaned = df.dropna()
    logging.info(f"🔍 After removing missing values: {df_cleaned.shape[0]} rows.")
    return df_cleaned

def remove_outliers(df, method='iqr', threshold=1.5):
    """Removes outliers using IQR or Z-score method."""
    df_cleaned = df.copy()
    
    if method == 'iqr':
        Q1 = df_cleaned.iloc[:, :-1].quantile(0.25)
        Q3 = df_cleaned.iloc[:, :-1].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df_cleaned.iloc[:, :-1] < (Q1 - threshold * IQR)) | (df_cleaned.iloc[:, :-1] > (Q3 + threshold * IQR))).any(axis=1)
    elif method == 'zscore':
        z_scores = np.abs((df_cleaned.iloc[:, :-1] - df_cleaned.iloc[:, :-1].mean()) / df_cleaned.iloc[:, :-1].std())
        mask = (z_scores < threshold).all(axis=1)
    else:
        logging.error("❌ Invalid outlier removal method.")
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    df_cleaned = df_cleaned[mask]
    logging.info(f"📉 Removed {df.shape[0] - df_cleaned.shape[0]} outliers using {method} method.")
    return df_cleaned

def compute_spectral_indices(df):
    """Computes NDVI and SR using identified RED (bands 50-80) and NIR (bands 100-140)."""

    # Convert all non-numeric columns to numeric (force conversion)
    df = df.apply(pd.to_numeric, errors='coerce')

    if df.isnull().all().any():
        logging.warning("⚠️ Some columns contain only NaN values after conversion!")

    # Compute Red & NIR bands
    red_band = df.iloc[:, 50:80].mean(axis=1)
    nir_band = df.iloc[:, 100:140].mean(axis=1)

    # Compute indices safely (handle NaN)
    df['NDVI'] = (nir_band - red_band) / (nir_band + red_band + 1e-6)
    df['SR'] = nir_band / (red_band + 1e-6)

    logging.info("✅ Computed NDVI and SR spectral indices.")
    return df

def scale_features(df, method='standard'):
    """Scales features using Standardization or Min-Max Scaling."""
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    feature_columns = [col for col in df.columns if col != 'vomitoxin_ppb']
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    logging.info("✅ Applied feature scaling using {} method.".format(method))
    return df

def shap_feature_selection(df, num_features=50):
    """Selects top features using SHAP values from a Random Forest model."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    top_features = np.argsort(shap_importance)[-num_features:]
    selected_columns = X.columns[top_features]
    df_selected = df[selected_columns].copy()
    df_selected['vomitoxin_ppb'] = y.values  # Ensures a proper copy is made
    
    logging.info(f"📌 Feature selection using SHAP reduced from {df.shape[1]-1} to {len(selected_columns)} features.")
    return df_selected

def transform_target(df):
    """Applies log transformation to the target variable."""
    df['vomitoxin_ppb'] = np.log1p(df['vomitoxin_ppb'])
    logging.info("✅ Applied log transformation to target variable.")
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Run Preprocessing
if __name__ == "__main__":
    try:
        df = load_data()
        df = df.drop(columns=['hsi_id'])
        df = handle_missing_values(df)
        df = remove_outliers(df, method='iqr')
        df = compute_spectral_indices(df)
        df = scale_features(df, method='standard')
        
        # Apply SHAP feature selection before splitting to avoid data leakage
        df = shap_feature_selection(df, num_features=50)
        
        # Split after SHAP selection
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Save processed data in correct directory
        X_train.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(DATA_DIR, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)
        df.to_csv(os.path.join(DATA_DIR, "processed.csv"), index=False)
        
        logging.info(f"✅ Preprocessing complete! Data saved in {DATA_DIR}")
        print(f"✅ Preprocessing complete! Data saved in {DATA_DIR}")
    
    except Exception as e:
        logging.error(f"❌ Error in preprocessing: {str(e)}")
        print(f"❌ ERROR: {e}")
