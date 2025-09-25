import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Personalized Diet Recommender",
    page_icon="🥗",
    layout="wide"  # Use the full page width
)

# --- Sidebar ---
st.sidebar.title("About the App")
st.sidebar.info(
    "This application uses an AI model (LSTM) to predict the calorie count "
    "of your next meal. It analyzes the sequence of your last three meals "
    "to forecast your dietary habits."
)
st.sidebar.image("data:image/jpeg;base64,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", caption="AI-Powered Nutrition")

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and scaler."""
    # Adjusted path for deployment
    model_path = 'lstm_calorie_predictor.h5'
    scaler_path = 'scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or scaler not found. Ensure `saved_model` folder with `lstm_calorie_predictor.h5` and `scaler.pkl` is in the repository.")
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# --- Main UI ---
st.title("Next Meal Calorie Predictor")
st.markdown("Enter the details of your last 3 meals to get a calorie prediction for your next one.")

if model and scaler:
    # Organize inputs into three columns
    col1, col2, col3 = st.columns(3)
    input_data = []

    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # --- Meal 1 Input ---
    with col1:
        with st.container(border=True):
            st.subheader(" Meal 1 (Oldest)")
            protein1 = st.number_input("Protein (g)", min_value=0, value=20, key="prot_1")
            carbs1 = st.number_input("Carbs (g)", min_value=0, value=50, key="carb_1")
            fat1 = st.number_input("Fat (g)", min_value=0, value=15, key="fat_1")
            meal_type1 = st.selectbox("Meal Type", options=meal_types, key="meal_1")
            hour1 = st.slider("Hour of Day", 0, 23, 8, key="hour_1")
            day_str1 = st.selectbox("Day", options=days_of_week, index=datetime.today().weekday(), key="day_1")

    # --- Meal 2 Input ---
    with col2:
        with st.container(border=True):
            st.subheader(" Meal 2")
            protein2 = st.number_input("Protein (g)", min_value=0, value=35, key="prot_2")
            carbs2 = st.number_input("Carbs (g)", min_value=0, value=40, key="carb_2")
            fat2 = st.number_input("Fat (g)", min_value=0, value=20, key="fat_2")
            meal_type2 = st.selectbox("Meal Type", options=meal_types, index=1, key="meal_2")
            hour2 = st.slider("Hour of Day", 0, 23, 13, key="hour_2")
            day_str2 = st.selectbox("Day", options=days_of_week, index=datetime.today().weekday(), key="day_2")

    # --- Meal 3 Input ---
    with col3:
        with st.container(border=True):
            st.subheader(" Meal 3 (Most Recent)")
            protein3 = st.number_input("Protein (g)", min_value=0, value=25, key="prot_3")
            carbs3 = st.number_input("Carbs (g)", min_value=0, value=90, key="carb_3")
            fat3 = st.number_input("Fat (g)", min_value=0, value=10, key="fat_3")
            meal_type3 = st.selectbox("Meal Type", options=meal_types, index=2, key="meal_3")
            hour3 = st.slider("Hour of Day", 0, 23, 20, key="hour_3")
            day_str3 = st.selectbox("Day", options=days_of_week, index=datetime.today().weekday(), key="day_3")
    
    # Process inputs after they are all defined
    meals = [
        (protein1, carbs1, fat1, meal_type1, hour1, day_str1),
        (protein2, carbs2, fat2, meal_type2, hour2, day_str2),
        (protein3, carbs3, fat3, meal_type3, hour3, day_str3)
    ]

    for p, c, f, mt, h, ds in meals:
        day_of_week = days_of_week.index(ds)
        meal_encoding = [1 if m == mt else 0 for m in ['Breakfast', 'Dinner', 'Lunch', 'Snack']]
        meal_features = [p, c, f, h, day_of_week] + meal_encoding
        input_data.append(meal_features)

    st.divider()

    # --- Prediction Button and Output ---
    if st.button("Predict Calories for Next Meal", type="primary", use_container_width=True):
        try:
            input_array = np.array(input_data)
            scaled_input = scaler.transform(input_array)
            reshaped_input = scaled_input.reshape(1, 3, 9)
            
            prediction = model.predict(reshaped_input)
            predicted_calories = int(prediction[0][0])
            
            st.metric(label="Predicted Calories for Your Next Meal", value=f"{predicted_calories} kcal", delta="Based on your habits")

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.warning("Could not load the prediction model.")
