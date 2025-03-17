# **Machine Learning Model for Mycotoxin Prediction - Project Documentation**

## **1. Project Overview**
This project focuses on developing a robust machine learning model to predict mycotoxin levels in agricultural samples using spectral data. The end-to-end pipeline includes data preprocessing, feature selection, model training, evaluation, and deployment via a FastAPI-based API. The model aims to assist in quality control by accurately estimating contamination levels based on spectral features.

---
## **2. Project Structure**
```
MLE-Mycotoxin-Prediction/
│── data/                # Dataset storage
│   │── processed/       # Preprocessed data
│── deployment/          # API deployment scripts
│   │── api.py           # FastAPI for real-time predictions
│── logs/                # Logging information for debugging
│── models/              # Trained model storage
│   │── random_forest_model.pkl  # Saved Random Forest model
│   │── mlp_model.pkl            # Saved MLP model
│   │── selected_features.txt    # Features used in training
│── results/             # Evaluation results and plots
│   │── actual_vs_predicted.png
│   │── evaluation_results.json
│   │── residual_analysis.png
│   │── residuals_vs_predicted.png
│   │── shap_feature_importance.png
│── EDA.ipynb            # Exploratory Data Analysis Notebook
│── model_evaluation.py  # Model evaluation and interpretability
│── model_training.py    # Model training script
│── preprocessing_pipeline.py  # Data preprocessing & feature selection
│── requirements.txt     # Python dependencies
│── test_preprocessing.py # Unit tests for preprocessing & model pipeline
│── README.md            # Project documentation
│── MLE-Assignment.csv   # Raw dataset
```

---
## **3. Data Preprocessing**
The preprocessing pipeline is responsible for cleaning and preparing the raw spectral dataset for model training. It includes:
- **Handling missing values**: Dropping missing data points.
- **Outlier detection & removal**: Using IQR-based filtering.
- **Feature engineering**:
  - Computing spectral indices (NDVI, SR) using red and NIR bands.
  - Standardizing numerical features using `StandardScaler`.
- **Feature selection**: SHAP-based feature selection to reduce dimensions and improve model interpretability.

**Outputs:** Processed datasets (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) are saved in `data/processed/`.

---
## **4. Model Training**
Two machine learning models were trained and compared:
1. **Random Forest Regressor** - Selected due to its robustness and feature importance interpretability.
2. **MLP Regressor** - Used as a deep learning-based alternative for performance comparison.

Key steps:
- Training on preprocessed data.
- Hyperparameter tuning.
- Saving models (`random_forest_model.pkl`, `mlp_model.pkl`) in `models/`.
- Logging model training progress (`logs/model_training.log`).

---
## **5. Model Evaluation**
To assess model performance, the following metrics were used:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

Additionally, the following visualizations were generated:
- **Actual vs. Predicted Values** (`results/actual_vs_predicted.png`)
- **Residual Analysis Histogram** (`results/residual_analysis.png`)
- **Residuals vs. Predictions** (`results/residuals_vs_predicted.png`)
- **SHAP Feature Importance** (`results/shap_feature_importance.png`)

Results are stored in `results/evaluation_results.json`.

---
## **6. Model Deployment**
The trained model was deployed as a FastAPI service to allow real-time predictions.
- **API Endpoint:** `/predict`
- **Input Format:** JSON containing spectral features.
- **Response:** Predicted mycotoxin levels.
- **Logging:** API requests & errors recorded in `logs/`.

**Example API Request:**
```json
{
  "hsi_id": "imagoai_corn_500",
  "features": [0.45, 0.39, 0.40, ...]  # (Total 449 features)
}
```
**Response:**
```json
{
  "prediction": 0.0408
}
```

---
## **7. Unit Testing & Code Quality**
- **Unit tests** are included in `test_preprocessing.py` to validate preprocessing and model predictions.
- **Logging** ensures tracking of pipeline execution and error handling.
- **Documentation** is provided inline in the scripts and summarized here.

---
## **8. Final Deliverables**
- **GitHub Repository**: [Link to project repository](https://github.com/SarathKumaryadla/MLE-Mycotoxin-Prediction)
- **Project Code**: Well-structured Python scripts.
- **Evaluation Report**: Stored in `results/`.
- **API Deployment**: FastAPI-based inference pipeline.

---
## **9. Future Improvements**
- **Hyperparameter tuning:** Further optimizing models for accuracy.
- **Feature Engineering:** Exploring additional spectral transformations.
- **Deployment Enhancements:** Dockerizing the API for easier deployment.
- **CI/CD Integration:** Automating testing and deployment.

---
## **10. Submission Details**
**Author:** Sarath Kumar Yadla  
**Email:** [Your Email]  
**Phone:** 9542838396  
**Submission Date:** 3/16/2025  


