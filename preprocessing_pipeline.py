import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

def handle_missing_values(df):
    """Handles missing values by imputing or removing them."""
    return df.dropna()

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
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    df_cleaned = df_cleaned[mask]
    print(f"Removed {df.shape[0] - df_cleaned.shape[0]} outliers using {method} method.")
    return df_cleaned

def compute_spectral_indices(df):
    """Computes NDVI and SR using identified RED (bands 50-80) and NIR (bands 100-140)."""
    red_band = df.iloc[:, 50:80].mean(axis=1)  # Averaging across RED range
    nir_band = df.iloc[:, 100:140].mean(axis=1)  # Averaging across NIR range
    
    df['NDVI'] = (nir_band - red_band) / (nir_band + red_band + 1e-6)  # Adding small constant to avoid division by zero
    df['SR'] = nir_band / (red_band + 1e-6)
    print("Computed NDVI and SR spectral indices.")
    return df

def scale_features(df, method='standard'):
    """Scales features using Standardization or Min-Max Scaling."""
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    feature_columns = [col for col in df.columns if col != 'vomitoxin_ppb']
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
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
    df_selected = df[selected_columns]
    df_selected.loc[:, 'vomitoxin_ppb'] = y.values  # Ensures a proper copy is made # Add target variable back
    
    print(f"Feature selection using SHAP reduced from {df.shape[1]-1} to {len(selected_columns)} features.")
    return df_selected

def transform_target(df):
    """Applies log transformation to the target variable."""
    df['vomitoxin_ppb'] = np.log1p(df['vomitoxin_ppb'])
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("C:/Users/ykumar/Learnbay/IMAGOAI/MLE-Assignment.csv")
    df = df.drop(columns=['hsi_id'])
    df = handle_missing_values(df)
    df = remove_outliers(df, method='iqr')
    df = compute_spectral_indices(df)
    df = scale_features(df, method='standard')
    
    # Apply SHAP feature selection before splitting to avoid data leakage
    df = shap_feature_selection(df, num_features=50)
    
    # Split after SHAP selection
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save processed data
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    df.to_csv("processed.csv", index=False)
    
    print("Preprocessing complete! SHAP-based feature selection applied and data saved.")

    
