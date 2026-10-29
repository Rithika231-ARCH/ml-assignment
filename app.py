"""
================================================================================
Machine Learning Application
Streamlit Web Interface: Indian House Rent Prediction
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="House Rent Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .rent-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #15803d;
        margin: 10px 0;
    }
    .rent-range {
        font-size: 1rem;
        color: #166534;
        font-weight: 500;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-blue { background-color: #dbeafe; color: #1e40af; }
    .badge-purple { background-color: #f3e8ff; color: #6b21a8; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# LOAD MODEL AND METADATA
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model_path = 'house_rent_model.joblib'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please run 'python train_model.py' first.")
        st.stop()
    payload = joblib.load(model_path)
    return payload

@st.cache_data
def load_metrics():
    metrics_path = 'model_evaluation_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

payload = load_model_artifacts()
pipeline = payload['pipeline']
best_model_name = payload['model_name']
unique_vals = payload['unique_values']
metrics_data = load_metrics()

# City median benchmarks for quick comparison
CITY_MEDIANS = {
    'Mumbai': 38000,
    'Delhi': 16000,
    'Bangalore': 16000,
    'Chennai': 16000,
    'Hyderabad': 15000,
    'Kolkata': 9500
}

# ------------------------------------------------------------------------------
# SIDEBAR: USER INPUTS
# ------------------------------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=600&q=80", use_container_width=True)
st.sidebar.header("🏠 Property Specifications")

city = st.sidebar.selectbox("City", unique_vals['City'], index=unique_vals['City'].index('Mumbai') if 'Mumbai' in unique_vals['City'] else 0)
bhk = st.sidebar.selectbox("Number of Bedrooms (BHK)", [1, 2, 3, 4, 5, 6], index=1)
size = st.sidebar.slider("Property Size (Sq. Ft.)", min_value=150, max_value=5000, value=950, step=25)
bathroom = st.sidebar.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5, 6], index=min(bhk - 1, 5) if bhk > 0 else 0)

furnishing = st.sidebar.selectbox("Furnishing Status", unique_vals['Furnishing Status'], index=1)
tenant = st.sidebar.selectbox("Tenant Preferred", unique_vals['Tenant Preferred'], index=0)
area_type = st.sidebar.selectbox("Area Type", unique_vals['Area Type'], index=unique_vals['Area Type'].index('Carpet Area') if 'Carpet Area' in unique_vals['Area Type'] else 0)
contact = st.sidebar.selectbox("Point of Contact", unique_vals['Point of Contact'], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🏢 Floor Details")
floor_level = st.sidebar.number_input("Floor Level (0 = Ground, -1 = Basement)", min_value=-2, max_value=80, value=2, step=1)
total_floors = st.sidebar.number_input("Total Floors in Building", min_value=1, max_value=90, value=6, step=1)

if floor_level > total_floors:
    st.sidebar.warning("Note: Floor Level exceeds Total Floors.")

# ------------------------------------------------------------------------------
# MAIN CONTENT HEADER & TABS
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">🏠 AI House Rent Prediction & Analytics System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End Machine Learning Regression Pipeline with Real Estate Analytics</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Rent Estimator",
    "📊 Dataset Exploratory Analysis",
    "🤖 ML Models & Evaluation",
    "📋 Project Documentation & Architecture"
])

# ==============================================================================
# TAB 1: RENT ESTIMATOR
# ==============================================================================
with tab1:
    st.subheader("💡 Instant Rental Valuation")
    st.write("Adjust property parameters in the sidebar and view the ML model's estimation below.")

    # Feature Engineering
    size_per_bhk = size / bhk
    bathroom_to_bhk = bathroom / bhk
    is_top_floor = 1 if floor_level >= total_floors else 0
    is_ground_floor = 1 if floor_level == 0 else 0

    input_data = pd.DataFrame([{
        'BHK': bhk,
        'Size': size,
        'Bathroom': bathroom,
        'Floor_Level': floor_level,
        'Total_Floors': total_floors,
        'Size_per_BHK': size_per_bhk,
        'Bathroom_to_BHK_ratio': bathroom_to_bhk,
        'Is_Top_Floor': is_top_floor,
        'Is_Ground_Floor': is_ground_floor,
        'City': city,
        'Furnishing Status': furnishing,
        'Tenant Preferred': tenant,
        'Area Type': area_type,
        'Point of Contact': contact
    }])

    # Predict
    pred_log = pipeline.predict(input_data)[0]
    predicted_rent = np.expm1(pred_log)
    lower_bound = predicted_rent * 0.85
    upper_bound = predicted_rent * 1.15
    city_med = CITY_MEDIANS.get(city, 20000)
    pct_diff_city = ((predicted_rent - city_med) / city_med) * 100

    col_val, col_meta = st.columns([1.2, 1])

    with col_val:
        st.markdown(f"""
        <div class="metric-card">
            <span class="badge badge-blue">Model: {best_model_name}</span>
            <span class="badge badge-purple">{city}</span>
            <div style="color: #4b5563; font-size: 1rem; margin-top: 10px;">ESTIMATED MONTHLY RENT</div>
            <div class="rent-value">₹ {int(round(predicted_rent, -2)):,}</div>
            <div class="rent-range">Plausible Market Range: ₹ {int(round(lower_bound, -2)):,} – ₹ {int(round(upper_bound, -2)):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_meta:
        st.markdown("#### 📌 Key Property Characteristics")
        st.write(f"- **Property Layout:** {bhk} BHK | {bathroom} Bathrooms | {size:,} Sq. Ft.")
        st.write(f"- **Average Room Size:** `{size_per_bhk:.1f}` Sq. Ft. per BHK")
        st.write(f"- **Floor Location:** Floor {floor_level} of {total_floors} ({'Ground Floor' if is_ground_floor else 'Top Floor' if is_top_floor else 'Mid Floor'})")
        st.write(f"- **Furnishing & Tenant:** {furnishing} | {tenant}")
        
        if pct_diff_city >= 0:
            st.info(f"📈 This estimate is **{pct_diff_city:.1f}% higher** than the median rent for {city} (₹{city_med:,}).")
        else:
            st.info(f"📉 This estimate is **{abs(pct_diff_city):.1f}% lower** than the median rent for {city} (₹{city_med:,}).")

    st.markdown("---")
    st.markdown("#### 🔍 Real-Time Input Feature Vector passed to ML Pipeline:")
    st.dataframe(input_data, use_container_width=True)

# ==============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab2:
    st.subheader("📊 Exploratory Data Analysis & Feature Distributions")
    st.write("Visual patterns and empirical distributions uncovered in the 4,746 house rental listings across India.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1. Target Variable Distribution")
        st.image("plots/01_rent_distribution.png", caption="Raw Rent vs Log-Transformed log(1 + Rent)", use_container_width=True)
        st.caption("Rental prices exhibit severe right-skewness. Applying log(1 + Rent) normalizes the variance, preventing large outlier dominance during gradient descent.")

    with col2:
        st.markdown("##### 2. Inter-City Rental Variation")
        st.image("plots/02_rent_by_city.png", caption="Median & Mean Rental Prices Across 6 Indian Metros", use_container_width=True)
        st.caption("Mumbai commands significantly higher rental prices due to acute geographical land constraints, followed by Bangalore, Delhi, and Chennai.")

    st.markdown("---")
    st.markdown("##### 3. Relationship between Property Size, BHK, and Monthly Rent")
    st.image("plots/03_rent_vs_size_bhk.png", caption="Rent vs Size (Sq. Ft.) Categorized by BHK", use_container_width=True)
    st.caption("Strong positive non-linear correlation between square footage and rent, segmented distinctly by bedroom count.")

# ==============================================================================
# TAB 3: ML MODELS & EVALUATION
# ==============================================================================
with tab3:
    st.subheader("🤖 Machine Learning Model Benchmarking & Diagnostics")
    st.write("Five machine learning algorithms were trained and benchmarked using rigorous 5-fold cross-validation.")

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1. Comparative Performance Metrics")
        st.image("plots/04_model_comparison.png", caption="MAE, RMSE, and Log R² Across All 5 Algorithms", use_container_width=True)

    with col2:
        st.markdown("##### 2. Actual vs. Predicted Rent (Test Set)")
        st.image("plots/05_actual_vs_predicted.png", caption="Scatter Alignment along the Perfect Identity Line", use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### 3. Residual Diagnostics & Normality")
        st.image("plots/06_residual_distribution.png", caption="Residuals vs Predicted and Zero-Centered Error Distribution", use_container_width=True)

    with col4:
        st.markdown("##### 4. Feature Importance Ranking")
        st.image("plots/07_feature_importance.png", caption="Gini Impurity Reduction / Feature Contribution", use_container_width=True)

# ==============================================================================
# TAB 4: PROJECT SUMMARY & ARCHITECTURE
# ==============================================================================
with tab4:
    st.subheader("📋 Project Overview & Technical Architecture")
    
    st.markdown("""
    ### 1. Problem Statement
    In modern urban economies, residential house rents fluctuate dynamically based on physical attributes (square footage, bedrooms, bathrooms), spatial location (metropolitan tier, city locality), and lease terms. Accurate rent estimation is essential for tenants to prevent exploitation and for property owners to maintain competitive occupancy rates.

    ### 2. Dataset Overview
    - **Total Records:** 4,746 rental listings across 6 metropolitan cities (Mumbai, Bangalore, Chennai, Hyderabad, Delhi, Kolkata).
    - **Target Variable:** `Rent` (Monthly rent in INR, continuous numerical regression target).
    - **Features:** BHK, Size (sq.ft.), Bathroom, Floor structure, Area Type, City, Furnishing Status, Tenant Preferred, Point of Contact.

    ### 3. Machine Learning Methodology
    1. **Data Preprocessing & Scaling:** Handled raw string parsing for floor levels, engineered domain ratios (`Size_per_BHK`, `Bathroom_to_BHK_ratio`), standardized numerical attributes via `StandardScaler`, and one-hot encoded categorical variables with `OneHotEncoder(drop='first')`.
    2. **Target Transformation:** Addressed severe positive skewness by transforming rent via $y_{log} = \\ln(1 + \\text{Rent})$, allowing linear models and gradient ensembles to fit stable Gaussian residuals.
    3. **Algorithms Compared:**
       - **Linear Regression (OLS):** Standard parametric baseline.
       - **Ridge Regression:** Penalized least-squares regression with $L_2$ regularization.
       - **Decision Tree Regressor:** Non-linear recursive partitioning baseline.
       - **Random Forest Regressor:** Bagging ensemble averaging 150 randomized decision trees.
       - **Gradient Boosting Regressor:** Sequential boosting optimizing least-squares loss over residual errors.
    4. **Model Evaluation:** Evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and $R^2$ score across 5-fold cross-validation.
    """)
    st.success("Trained model pipeline serialized into 'house_rent_model.joblib' for zero-latency inference.")
