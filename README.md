# House Rent Prediction – Machine Learning Application

**Assignment 2 – Machine Learning Application**  
**Topic:** Build and Analyze a Machine Learning Model (House Rent Prediction)

---

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline that predicts residential rental prices across major Indian metropolitan hubs (Mumbai, Bangalore, Chennai, Hyderabad, Delhi, Kolkata). It includes:
- **Comprehensive Data Preprocessing & Feature Engineering** (Floor hierarchy extraction, room ratios, one-hot encoding, and log-target normalization)
- **Benchmarking of 5 Machine Learning Algorithms**:
  1. Multiple Linear Regression (OLS)
  2. Ridge Regression ($L_2$ Regularization)
  3. Decision Tree Regressor
  4. Random Forest Regressor (Bagging)
  5. Gradient Boosting Regressor (Boosting)
- **Rigorous Evaluation Metrics** (MAE, RMSE, MAPE, $R^2$ Score on original and log scale, 5-Fold Cross Validation)
- **7 High-Resolution Analytical Visualizations** saved in `plots/`
- **Interactive Streamlit Web Application** (`app.py`) for live user inference and market diagnostics
- **Full Academic Project Report** in Markdown (`House_Rent_Prediction_Report.md`) and self-contained HTML (`House_Rent_Prediction_Report.html`)

---

## 📁 Repository Structure

```
├── House_Rent_Dataset.csv            # Dataset (4,746 real estate listings)
├── train_model.py                    # Complete ML training & evaluation script
├── app.py                            # Streamlit web application
├── generate_report_html.py           # Script to compile report to standalone HTML
├── house_rent_model.joblib           # Serialized Gradient Boosting model pipeline
├── model_evaluation_metrics.csv      # Model performance comparative metrics (CSV)
├── model_evaluation_metrics.json     # Model performance comparative metrics (JSON)
├── House_Rent_Prediction_Report.md   # Comprehensive academic project report
├── House_Rent_Prediction_Report.html # Publication-styled HTML report (Print-to-PDF ready)
├── README.md                         # Project documentation and quick start guide
└── plots/                            # High-resolution generated charts
    ├── 01_rent_distribution.png      # Raw vs Log-transformed rent distribution
    ├── 02_rent_by_city.png           # Rent distribution & mean vs median across cities
    ├── 03_rent_vs_size_bhk.png       # Rent vs Property Size categorized by BHK
    ├── 04_model_comparison.png       # Comparative MAE, RMSE, and R² across 5 models
    ├── 05_actual_vs_predicted.png    # Actual vs Predicted scatter with identity line
    ├── 06_residual_distribution.png  # Residual vs Predicted & error normality histogram
    └── 07_feature_importance.png     # Top 15 influential features driving house rent
```

---

## 🚀 Quick Start Guide

### 1. Requirements
Ensure Python 3.10+ is installed with the required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn streamlit joblib
```

### 2. Train the Machine Learning Models
To reproduce all data preprocessing, 5-fold cross-validation, model evaluations, plot generations, and model export:
```bash
python train_model.py
```

### 3. Launch the Interactive Streamlit Web Interface
To start the interactive web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to:
- Estimate rent dynamically by selecting City, BHK, Size, Bathrooms, and Floor levels.
- View real-time confidence intervals and comparison against city medians.
- Explore interactive dataset plots and model benchmarking tables.

### 4. View or Export the Project Report
- View the Markdown report directly: [House_Rent_Prediction_Report.md](House_Rent_Prediction_Report.md)
- Open `House_Rent_Prediction_Report.html` in any browser (Chrome/Edge/Firefox) and press `Ctrl + P` -> **Save as PDF** to generate an academic PDF report.

---

## 📊 Summary of Model Performance

| Model | MAE (INR) ↓ | RMSE (INR) ↓ | MAPE (%) ↓ | $R^2$ (Log Scale) ↑ | 5-Fold CV $R^2$ ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | ₹11,355.71 | ₹32,935.03 | 30.66% | 0.8308 | 0.8086 |
| **Ridge Regression** | ₹11,357.34 | ₹32,965.38 | 30.65% | 0.8308 | 0.8086 |
| **Decision Tree** | ₹12,434.82 | ₹43,431.38 | 33.43% | 0.8028 | 0.7801 |
| **Random Forest** | **₹10,609.37** | ₹33,949.93 | 29.96% | 0.8372 | 0.8140 |
| **Gradient Boosting** | ₹10,795.61 | ₹36,762.84 | **29.46%** | **0.8428** | **0.8255** |

*The Gradient Boosting Regressor achieved the highest overall explanatory power ($R^2 = 0.8428$, $\text{MAPE} = 29.46\%$), closely followed by the Random Forest Regressor ($\text{MAE} = ₹10,609.37$).*
