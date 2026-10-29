"""
Generates a dedicated, publication-quality image of the evaluation metrics table and charts.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load metrics
df = pd.read_csv('model_evaluation_metrics.csv')

fig = plt.figure(figsize=(15, 8.5), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Header Title
fig.text(0.5, 0.95, 'Machine Learning Model Evaluation Benchmark', 
         fontsize=19, fontweight='bold', ha='center', color='#1e3a8a')
fig.text(0.5, 0.915, 'Comparative Performance on Holdout Test Set (N = 950) with 5-Fold Cross-Validation', 
         fontsize=11.5, ha='center', color='#4b5563')

# 1. Table in top half
ax_table = fig.add_axes([0.04, 0.48, 0.92, 0.38])
ax_table.axis('off')

col_labels = ['Model Name', 'MAE (INR) ↓', 'RMSE (INR) ↓', 'MAPE (%) ↓', 'R² (Log) ↑', 'R² (Orig) ↑', '5-Fold CV R² ↑']
cell_data = []

for _, row in df.iterrows():
    m = row['Model']
    mae = f"₹ {row['MAE (INR)']:,.2f}"
    rmse = f"₹ {row['RMSE (INR)']:,.2f}"
    mape = f"{row['MAPE (%)']:.2f}%"
    r2_log = f"{row['R2 Score (Log Scale)']:.4f}"
    r2_orig = f"{row['R2 Score (Original)']:.4f}"
    cv = f"{row['CV R2 Mean']:.4f} ± {row['CV R2 Std']:.4f}"
    
    if 'Gradient' in m:
        m += '  ★ (Best Overall / Deployed)'
    elif 'Random Forest' in m:
        m += '  (Lowest Rupee Error)'
    
    cell_data.append([m, mae, rmse, mape, r2_log, r2_orig, cv])

table = ax_table.table(
    cellText=cell_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1.0, 2.1)

# Style header
for j in range(len(col_labels)):
    cell = table[(0, j)]
    cell.set_facecolor('#1e40af')
    cell.set_text_props(color='white', fontweight='bold')
    cell.set_edgecolor('#cbd5e1')

# Style rows
for i in range(1, len(cell_data) + 1):
    row_model = df.iloc[i-1]['Model']
    is_best = 'Gradient' in row_model
    is_rf = 'Random Forest' in row_model
    bg_color = '#dcfce7' if is_best else ('#f0fdf4' if is_rf else ('#f8fafc' if i % 2 == 0 else '#ffffff'))
    
    for j in range(len(col_labels)):
        cell = table[(i, j)]
        cell.set_facecolor(bg_color)
        cell.set_edgecolor('#cbd5e1')
        if j == 0:
            cell.set_text_props(ha='left', fontweight='bold' if (is_best or is_rf) else 'normal')
        else:
            if (is_best and j in [3, 4, 6]) or (is_rf and j in [1, 5]):
                cell.set_text_props(fontweight='bold', color='#15803d')

# 2. Mini bar charts in bottom half
ax1 = fig.add_axes([0.08, 0.08, 0.25, 0.32])
ax2 = fig.add_axes([0.40, 0.08, 0.25, 0.32])
ax3 = fig.add_axes([0.72, 0.08, 0.25, 0.32])

models_short = ['Linear', 'Ridge', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
colors_palette = ['#94a3b8', '#94a3b8', '#cbd5e1', '#3b82f6', '#10b981']

# MAE
bars1 = ax1.bar(models_short, df['MAE (INR)'], color=colors_palette, edgecolor='none', width=0.55)
ax1.set_title('Mean Absolute Error (INR) ↓', fontsize=11.5, fontweight='bold', color='#1e3a8a')
ax1.set_ylabel('INR (₹)', fontsize=9.5)
ax1.tick_params(axis='x', rotation=30, labelsize=8.5)
for b in bars1:
    ax1.annotate(f"₹{int(b.get_height()):,}", (b.get_x() + b.get_width() / 2., b.get_height()),
                 ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# RMSE
bars2 = ax2.bar(models_short, df['RMSE (INR)'], color=['#94a3b8', '#94a3b8', '#f87171', '#3b82f6', '#10b981'], edgecolor='none', width=0.55)
ax2.set_title('Root Mean Squared Error (INR) ↓', fontsize=11.5, fontweight='bold', color='#1e3a8a')
ax2.set_ylabel('INR (₹)', fontsize=9.5)
ax2.tick_params(axis='x', rotation=30, labelsize=8.5)
for b in bars2:
    ax2.annotate(f"₹{int(b.get_height()):,}", (b.get_x() + b.get_width() / 2., b.get_height()),
                 ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# R2 Score
bars3 = ax3.bar(models_short, df['R2 Score (Log Scale)'], color=['#94a3b8', '#94a3b8', '#cbd5e1', '#3b82f6', '#10b981'], edgecolor='none', width=0.55)
ax3.set_title('R² Score on Log Scale ↑', fontsize=11.5, fontweight='bold', color='#1e3a8a')
ax3.set_ylabel('R² Score', fontsize=9.5)
ax3.set_ylim(0.72, 0.88)
ax3.tick_params(axis='x', rotation=30, labelsize=8.5)
for b in bars3:
    ax3.annotate(f"{b.get_height():.4f}", (b.get_x() + b.get_width() / 2., b.get_height()),
                 ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
ax3.grid(axis='y', linestyle='--', alpha=0.4)

out_file = 'plots/evaluation_metrics_table.png'
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"Saved {out_file} successfully!")
