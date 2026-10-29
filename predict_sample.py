"""
Quick Prediction Script
Run this script to test single property predictions directly in terminal.
"""

import joblib
import numpy as np
import pandas as pd

# 1. Load the trained pipeline
payload = joblib.load('house_rent_model.joblib')
pipeline = payload['pipeline']
model_name = payload['model_name']

print(f"=== Loaded Model: {model_name} ===")

# 2. Define property inputs
city = "Mumbai"           # Mumbai, Bangalore, Chennai, Hyderabad, Delhi, Kolkata
bhk = 2                   # 1, 2, 3, 4, 5, 6
size = 950                # Square feet
bathrooms = 2             # Number of bathrooms
floor_level = 3           # Floor number (0 = Ground)
total_floors = 10         # Total floors in building
furnishing = "Semi-Furnished" # Furnished, Semi-Furnished, Unfurnished
tenant_pref = "Bachelors/Family" # Bachelors/Family, Bachelors, Family
area_type = "Carpet Area" # Super Area, Carpet Area, Built Area
contact = "Contact Owner" # Contact Owner, Contact Agent, Contact Builder

# 3. Compute engineered features
size_per_bhk = size / bhk
bathroom_to_bhk = bathrooms / bhk
is_top_floor = 1 if floor_level >= total_floors else 0
is_ground_floor = 1 if floor_level == 0 else 0

input_df = pd.DataFrame([{
    'BHK': bhk,
    'Size': size,
    'Bathroom': bathrooms,
    'Floor_Level': floor_level,
    'Total_Floors': total_floors,
    'Size_per_BHK': size_per_bhk,
    'Bathroom_to_BHK_ratio': bathroom_to_bhk,
    'Is_Top_Floor': is_top_floor,
    'Is_Ground_Floor': is_ground_floor,
    'City': city,
    'Furnishing Status': furnishing,
    'Tenant Preferred': tenant_pref,
    'Area Type': area_type,
    'Point of Contact': contact
}])

# 4. Predict
pred_log = pipeline.predict(input_df)[0]
predicted_rent = np.expm1(pred_log)

print("\n--- Property Details ---")
print(f"Location    : {city}")
print(f"Layout      : {bhk} BHK | {bathrooms} Baths | {size} sq.ft.")
print(f"Floor       : Floor {floor_level} of {total_floors}")
print(f"Furnishing  : {furnishing}")

import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("\n--- Model Output ---")
print(f"Estimated Monthly Rent : INR {int(round(predicted_rent, -2)):,}")
print(f"Plausible Range (±15%)  : INR {int(round(predicted_rent * 0.85, -2)):,} - INR {int(round(predicted_rent * 1.15, -2)):,}")
