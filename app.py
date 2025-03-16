import streamlit as st
import numpy as np
import joblib
import xgboost as xgb

st.title("🏃‍♂️ Activity Level & Calorie Prediction App")

# Load models
model = joblib.load("model1.pkl")  # Activity Level Prediction Model
model2 = joblib.load("XGBoostModel.joblib")  # Calorie Prediction Model (XGBoost)

# Define feature names used during model training
feature_names = [
    "TotalSteps", "TotalDistance", "LightActiveDistance", "VeryActiveMinutes", 
    "LightlyActiveMinutes", "SedentaryMinutes", "TotalActiveMinutes", "ActivityRatio"
]

# User Inputs
total_steps = st.number_input("🚶‍♂️ Total Steps", min_value=0)
total_distance = st.number_input("📏 Total Distance (km)", min_value=0.0)
light_active_distance = st.number_input("🚶‍♂️ Light Active Distance (km)", min_value=0.0)
very_active_minutes = st.number_input("🔥 Very Active Minutes", min_value=0)
lightly_active_minutes = st.number_input("💡 Lightly Active Minutes", min_value=0)
sedentary_minutes = st.number_input("🪑 Sedentary Minutes", min_value=0)

# Derived Features
total_active_minutes = very_active_minutes + lightly_active_minutes
activity_ratio = very_active_minutes / (sedentary_minutes + 1e-6)  # Prevents division by zero

if st.button("🔍 Predict"):
    # Prepare input for activity level prediction
    user_input_activity = np.array([[total_steps, total_distance, very_active_minutes, lightly_active_minutes]])
    prediction = model.predict(user_input_activity)[0]
    st.success(f"### 📊 Predicted Activity Level: {prediction}")

    # Prepare input for calorie prediction
    user_input_calories = np.array([[total_steps, total_distance, light_active_distance, very_active_minutes,
                                     lightly_active_minutes, sedentary_minutes, total_active_minutes, activity_ratio]])

    # Convert to DMatrix with feature names
    dmatrix_input = xgb.DMatrix(user_input_calories, feature_names=feature_names)

    # Make prediction
    calories = model2.predict(dmatrix_input)[0]
    st.success(f"### 🔥 Predicted Calories Burnt: {calories:.2f}")
