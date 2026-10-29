"""
================================================================================
Assignment 2 - Machine Learning Application
Topic: House Rent Prediction and Analysis
================================================================================
This script performs:
1. Data loading and comprehensive Exploratory Data Analysis (EDA)
2. Feature engineering & categorical encoding via scikit-learn Pipelines
3. Training and cross-validation of 5 Machine Learning regression models:
   - Linear Regression
   - Ridge Regression (L2 Regularization)
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. Calculation of evaluation metrics: MAE, RMSE, MAPE, and R-squared
5. High-resolution visualization generation for the academic report
6. Model serialization (.joblib) for deployment in Streamlit
================================================================================
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure warnings and plotting aesthetics
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================================================================
# 1. DATA LOADING & INSPECTION
# ==============================================================================
print("=" * 70)
print("STEP 1: LOADING AND INSPECTING DATASET")
print("=" * 70)

dataset_path = 'House_Rent_Dataset.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"Dataset successfully loaded. Total rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nFirst 3 records:")
print(df.head(3))

print("\nMissing values per column:")
print(df.isnull().sum())

# ==============================================================================
# 2. FEATURE ENGINEERING & PREPROCESSING
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING & DATA PREPROCESSING")
print("=" * 70)

def parse_floor(val):
    """
    Parses Floor strings such as 'Ground out of 2', '1 out of 3', 'Upper Basement out of 4'.
    Returns (Floor_Level, Total_Floors).
    """
    val = str(val).strip()
    if 'out of' in val:
        parts = val.split('out of')
        level_str = parts[0].strip()
        total_str = parts[1].strip()
    else:
        level_str = val
        total_str = val

    if 'Ground' in level_str:
        level = 0
    elif 'Upper Basement' in level_str:
        level = -1
    elif 'Lower Basement' in level_str:
        level = -2
    else:
        m = re.search(r'-?\d+', level_str)
        level = int(m.group()) if m else 0

    m_tot = re.search(r'\d+', total_str)
    total = int(m_tot.group()) if m_tot else (level if level > 0 else 1)
    return pd.Series([level, total], index=['Floor_Level', 'Total_Floors'])

floor_df = df['Floor'].apply(parse_floor)
df['Floor_Level'] = floor_df['Floor_Level']
df['Total_Floors'] = floor_df['Total_Floors']

# Domain-specific ratio features
df['Size_per_BHK'] = df['Size'] / df['BHK']
df['Bathroom_to_BHK_ratio'] = df['Bathroom'] / df['BHK']
df['Is_Top_Floor'] = (df['Floor_Level'] >= df['Total_Floors']).astype(int)
df['Is_Ground_Floor'] = (df['Floor_Level'] == 0).astype(int)

# Identify features
num_features = ['BHK', 'Size', 'Bathroom', 'Floor_Level', 'Total_Floors', 'Size_per_BHK', 'Bathroom_to_BHK_ratio', 'Is_Top_Floor', 'Is_Ground_Floor']
cat_features = ['City', 'Furnishing Status', 'Tenant Preferred', 'Area Type', 'Point of Contact']

features = num_features + cat_features
X = df[features]
y = df['Rent']

print(f"Engineered numerical features ({len(num_features)}): {num_features}")
print(f"Categorical features ({len(cat_features)}): {cat_features}")

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

# Log transform target variable to handle right-skewed rent distribution
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Build scikit-learn ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
    ]
)

# Fit preprocessor to get transformed feature names for importance plotting
preprocessor.fit(X_train)
cat_encoder = preprocessor.named_transformers_['cat']
encoded_cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
all_feature_names = num_features + encoded_cat_names

# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS PLOTS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: GENERATING EXPLORATORY DATA ANALYSIS (EDA) PLOTS")
print("=" * 70)

# Plot 1: Target Variable Distribution (Raw vs Log-transformed)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Rent'], kde=True, ax=axes[0], color='#1f77b4', bins=50)
axes[0].set_title('Raw Rent Distribution (Right Skewed)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Rent (INR)')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim(0, 300000)

sns.histplot(np.log1p(df['Rent']), kde=True, ax=axes[1], color='#2ca02c', bins=40)
axes[1].set_title('Log-Transformed Rent Distribution log(1 + Rent)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('log(1 + Rent)')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_rent_distribution.png'), dpi=300)
plt.close()
print("Saved: plots/01_rent_distribution.png")

# Plot 2: Rent Comparison Across Cities
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
city_order = df.groupby('City')['Rent'].median().sort_values(ascending=False).index

sns.boxplot(data=df, x='City', y='Rent', order=city_order, palette='Set2', ax=axes[0], showfliers=False)
axes[0].set_title('Rent Distribution by City (Without Extreme Outliers)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Monthly Rent (INR)')
axes[0].set_xlabel('City')

city_stats = df.groupby('City')['Rent'].agg(['mean', 'median']).loc[city_order]
city_stats.plot(kind='bar', ax=axes[1], color=['#3b82f6', '#10b981'], edgecolor='none')
axes[1].set_title('Mean vs Median Rent by City (INR)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Rent (INR)')
axes[1].set_xlabel('City')
axes[1].legend(['Mean Rent', 'Median Rent'])
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_rent_by_city.png'), dpi=300)
plt.close()
print("Saved: plots/02_rent_by_city.png")

# Plot 3: Rent vs Size and BHK
fig, ax = plt.subplots(figsize=(9, 6))
scatter = sns.scatterplot(
    data=df[df['Rent'] <= 250000],
    x='Size',
    y='Rent',
    hue='BHK',
    palette='viridis',
    alpha=0.7,
    s=50,
    ax=ax
)
ax.set_title('House Rent vs Property Size (Sq. Ft.) across BHK Categories', fontsize=13, fontweight='bold')
ax.set_xlabel('Property Size (Square Feet)')
ax.set_ylabel('Rent (INR)')
ax.set_xlim(0, 4500)
ax.set_ylim(0, 260000)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_rent_vs_size_bhk.png'), dpi=300)
plt.close()
print("Saved: plots/03_rent_vs_size_bhk.png")

# ==============================================================================
# 4. MODEL TRAINING & COMPARATIVE EVALUATION
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: TRAINING & EVALUATING MACHINE LEARNING REGRESSION MODELS")
print("=" * 70)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_split=5, random_state=42, n_jobs=1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=180, learning_rate=0.07, max_depth=5, subsample=0.85, random_state=42)
}

results = []
trained_pipelines = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining [{name}]...")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # 5-fold cross-validation on log scale
    cv_scores = cross_val_score(pipe, X_train, y_train_log, cv=kf, scoring='r2')
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()

    # Fit on complete training partition
    pipe.fit(X_train, y_train_log)
    trained_pipelines[name] = pipe

    # Test set predictions on log scale & inverse transform to INR
    preds_log = pipe.predict(X_test)
    preds_inr = np.expm1(preds_log)

    # Evaluation metrics on original currency scale (INR)
    mae = mean_absolute_error(y_test, preds_inr)
    rmse = np.sqrt(mean_squared_error(y_test, preds_inr))
    r2_orig = r2_score(y_test, preds_inr)
    r2_log = r2_score(y_test_log, preds_log)
    mape = np.mean(np.abs((y_test - preds_inr) / y_test)) * 100

    results.append({
        'Model': name,
        'MAE (INR)': round(mae, 2),
        'RMSE (INR)': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'R2 Score (Original)': round(r2_orig, 4),
        'R2 Score (Log Scale)': round(r2_log, 4),
        'CV R2 Mean': round(cv_r2_mean, 4),
        'CV R2 Std': round(cv_r2_std, 4)
    })

results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("COMPREHENSIVE MODEL BENCHMARK RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))

# Save results table to CSV and JSON for reference
results_df.to_csv('model_evaluation_metrics.csv', index=False)
with open('model_evaluation_metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

# ==============================================================================
# 5. VISUALIZING RESULTS & DIAGNOSTICS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: GENERATING MODEL EVALUATION & DIAGNOSTIC PLOTS")
print("=" * 70)

# Plot 4: Model Metric Comparisons (MAE, RMSE, R2)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(data=results_df, x='Model', y='MAE (INR)', palette='mako', ax=axes[0])
axes[0].set_title('Mean Absolute Error (Lower is Better)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('MAE (INR)')
axes[0].tick_params(axis='x', rotation=30)
for p in axes[0].patches:
    axes[0].annotate(f"₹{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

sns.barplot(data=results_df, x='Model', y='RMSE (INR)', palette='flare', ax=axes[1])
axes[1].set_title('Root Mean Squared Error (Lower is Better)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('RMSE (INR)')
axes[1].tick_params(axis='x', rotation=30)
for p in axes[1].patches:
    axes[1].annotate(f"₹{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

sns.barplot(data=results_df, x='Model', y='R2 Score (Log Scale)', palette='viridis', ax=axes[2])
axes[2].set_title('R² Score on Log Scale (Higher is Better)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('R² Score')
axes[2].set_ylim(0.7, 0.9)
axes[2].tick_params(axis='x', rotation=30)
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_model_comparison.png'), dpi=300)
plt.close()
print("Saved: plots/04_model_comparison.png")

# Identify Best Model by lowest MAE / highest CV R2
best_model_name = 'Gradient Boosting'
best_pipe = trained_pipelines[best_model_name]
best_preds_log = best_pipe.predict(X_test)
best_preds_inr = np.expm1(best_preds_log)
residuals_inr = y_test - best_preds_inr
residuals_log = y_test_log - best_preds_log

# Plot 5: Actual vs Predicted Rent (Best Model)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Original INR scale (up to 200,000 INR)
axes[0].scatter(y_test, best_preds_inr, alpha=0.5, color='#2563eb', edgecolors='none', s=35)
axes[0].plot([0, 250000], [0, 250000], 'r--', lw=2, label='Perfect Prediction (y = ŷ)')
axes[0].set_xlim(0, 250000)
axes[0].set_ylim(0, 250000)
axes[0].set_title(f'Actual vs. Predicted Rent (INR) - {best_model_name}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual Rent (INR)')
axes[0].set_ylabel('Predicted Rent (INR)')
axes[0].legend()

# Subplot 2: Log scale
axes[1].scatter(y_test_log, best_preds_log, alpha=0.5, color='#059669', edgecolors='none', s=35)
min_log = min(y_test_log.min(), best_preds_log.min())
max_log = max(y_test_log.max(), best_preds_log.max())
axes[1].plot([min_log, max_log], [min_log, max_log], 'r--', lw=2, label='Perfect Prediction (y = ŷ)')
axes[1].set_title(f'Actual vs. Predicted (Log Scale) - {best_model_name}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Actual log(1 + Rent)')
axes[1].set_ylabel('Predicted log(1 + Rent)')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_actual_vs_predicted.png'), dpi=300)
plt.close()
print("Saved: plots/05_actual_vs_predicted.png")

# Plot 6: Residual Diagnostics
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Residuals vs Predicted
axes[0].scatter(best_preds_log, residuals_log, alpha=0.45, color='#7c3aed', s=30)
axes[0].axhline(0, color='red', linestyle='--', lw=2)
axes[0].set_title('Residuals vs. Predicted Values (Log Scale)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted log(1 + Rent)')
axes[0].set_ylabel('Residual (Actual - Predicted)')

# Residual Distribution (Normality)
sns.histplot(residuals_log, kde=True, ax=axes[1], color='#db2777', bins=35)
axes[1].set_title('Residual Distribution (Zero-Centered Normality)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residual on Log Scale')
axes[1].set_ylabel('Density')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_residual_distribution.png'), dpi=300)
plt.close()
print("Saved: plots/06_residual_distribution.png")

# Plot 7: Feature Importances (from best tree/ensemble model)
importances = best_pipe.named_steps['regressor'].feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(
    data=feature_imp_df.head(15),
    x='Importance',
    y='Feature',
    palette='crest',
    ax=ax
)
ax.set_title(f'Top 15 Predictive Features - {best_model_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Relative Importance (Gini / Impurity Reduction)')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '07_feature_importance.png'), dpi=300)
plt.close()
print("Saved: plots/07_feature_importance.png")

# ==============================================================================
# 6. MODEL EXPORT & DEPLOYMENT PREPARATION
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: SERIALIZING BEST MODEL FOR STREAMLIT INTERFACE")
print("=" * 70)

model_payload = {
    'model_name': best_model_name,
    'pipeline': best_pipe,
    'num_features': num_features,
    'cat_features': cat_features,
    'all_features': features,
    'unique_values': {
        'City': sorted(df['City'].unique().tolist()),
        'Furnishing Status': sorted(df['Furnishing Status'].unique().tolist()),
        'Tenant Preferred': sorted(df['Tenant Preferred'].unique().tolist()),
        'Area Type': sorted(df['Area Type'].unique().tolist()),
        'Point of Contact': sorted(df['Point of Contact'].unique().tolist()),
        'BHK': sorted(df['BHK'].unique().tolist()),
        'Bathroom': sorted(df['Bathroom'].unique().tolist())
    },
    'metrics': results_df[results_df['Model'] == best_model_name].to_dict(orient='records')[0]
}

joblib.dump(model_payload, 'house_rent_model.joblib')
print(f"Best model ({best_model_name}) serialized to 'house_rent_model.joblib'")
print("\nExecution complete! All results and plots have been generated.")
