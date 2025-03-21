# MLE-Mycotoxin-Prediction

## 📌 Project Overview
Mycotoxins are toxic secondary metabolites produced by fungi that contaminate food and agricultural products. This project aims to build a **machine learning model** to predict **vomitoxin (Deoxynivalenol) levels** using **spectral data**. The solution integrates data preprocessing, feature selection, model training, evaluation, and deployment as an API for real-time predictions.

---

## 📁 Repository Structure
```bash
MLE-Mycotoxin-Prediction/
│── data/                # Dataset storage
│   │── processed/       # Preprocessed data
│── deployment/          # API deployment scripts
│   │── api.py           # FastAPI for real-time predictions
│── logs/                # Logging information for debugging
│── models/              # Trained model storage
│   │── random_forest_model.pkl  # Saved Random Forest model
│   │── mlp_model.pkl            # Saved MLP model
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

## ⚙️ Setup Instructions
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/SarathKumaryadla/MLE-Mycotoxin-Prediction.git
cd MLE-Mycotoxin-Prediction
```

### **2️⃣ Install Dependencies**
Make sure you have **Python 3.8+** installed. Then, run:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline
### **1️⃣ Data Preprocessing**
```bash
python preprocessing_pipeline.py
```
- Handles missing values and outliers
- Computes NDVI & SR spectral indices
- Performs SHAP-based feature selection
- Splits data into train/test sets

### **2️⃣ Model Training**
```bash
python model_training.py
```
- Trains **Random Forest** and **MLP Regressor**
- Saves trained models in `models/`

### **3️⃣ Model Evaluation**
```bash
python model_evaluation.py
```
- Evaluates models using **MAE, RMSE, R² Score**
- Generates **visualizations (scatter plots, residual analysis, SHAP importance)**

### **4️⃣ API Deployment (FastAPI)**
```bash
uvicorn api:app --reload
```
- Runs the API on `http://127.0.0.1:8000/docs`
- Accepts **new spectral data** for real-time predictions

#### **Swagger UI for API Testing**
Navigate to:
- 🌍 **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 📜 **Redoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📊 Model Performance
| Model                 | MAE   | RMSE  | R² Score |
|----------------------|-------|-------|---------|
| MLP Regressor       | 0.1172 | 0.1417 | 0.9832  |
| Random Forest       | 0.0375 | 0.1247 | 0.9870  |

📌 **SHAP Feature Importance**: Identifies most impactful features in prediction.

---

## 🧪 Unit Testing & Logging
### **Run Unit Tests**
```bash
pytest test_preprocessing.py --disable-warnings
```
- Tests **data preprocessing**, **feature selection**, and **model evaluation**.

### **Logging**
- Logs are stored in `logs/`
- Includes **error tracking & runtime details**

---

## 📌 Future Improvements
✅ Implement an **attention-based model (Transformer)** for comparison.  
✅ Develop a **Streamlit web app** for easy predictions.  
✅ Enhance **Ensemble techniques** (stacking RF + MLP + Transformer).  
✅ Deploy via **Docker & CI/CD (GitHub Actions)**.

---

## 📩 Submission Details
- 📂 **GitHub Repository:** https://github.com/SarathKumaryadla/MLE-Mycotoxin-Prediction
- 📧 **Submit to:** keerthan.shagrithaya@imagoai.com

📌 **Developed by:** Sarath Kumar Yadla 
🚀 **Date:** 16 March 2025


