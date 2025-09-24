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
st.sidebar.image("https://i.imgur.com/b7xIoB9.png", caption="AI-Powered Nutrition")

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and scaler."""
    # Adjusted path for deployment
    model_path = 'saved_model/lstm_calorie_predictor.h5'
    scaler_path = 'saved_model/scaler.pkl'
    
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
