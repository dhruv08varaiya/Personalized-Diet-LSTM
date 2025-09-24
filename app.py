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
    layout="centered"
)

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and scaler."""
    model_path = 'lstm_calorie_predictor.h5'
    scaler_path = 'scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or scaler not found. Please run the `Diet_Prediction_LSTM.ipynb` notebook first to train and save them.")
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# --- UI ---
st.title("Next Meal Calorie Predictor")
st.markdown("Enter the details of your last 3 meals to get a calorie prediction for your next one.")

if model and scaler:
    st.header("Enter Your Last 3 Meals")
    input_data = []

    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for i in range(3):
        st.subheader(f"Meal {i+1}")
        cols = st.columns(3)
        with cols[0]:
            protein = st.number_input(f"Protein (g)", min_value=0, value=20, key=f"prot_{i}")
            carbs = st.number_input(f"Carbs (g)", min_value=0, value=50, key=f"carb_{i}")
            fat = st.number_input(f"Fat (g)", min_value=0, value=15, key=f"fat_{i}")
        
        with cols[1]:
            meal_type = st.selectbox(f"Meal Type", options=meal_types, key=f"meal_{i}")
        
        with cols[2]:
            hour = st.slider(f"Hour of Day (24h)", 0, 23, 12, key=f"hour_{i}")
            day_str = st.selectbox(f"Day of Week", options=days_of_week, index=datetime.today().weekday(), key=f"day_{i}")
            day_of_week = days_of_week.index(day_str)

        # Create one-hot encoding for meal type (must match training order)
        meal_encoding = [1 if mt == meal_type else 0 for mt in ['Breakfast', 'Dinner', 'Lunch', 'Snack']]
        
        # Assemble the feature vector in the correct order
        meal_features = [protein, carbs, fat, hour, day_of_week] + meal_encoding
        input_data.append(meal_features)

    if st.button("Predict Calories for Next Meal", type="primary"):
        try:
            input_array = np.array(input_data)
            
            # Scale the input data using the loaded scaler
            scaled_input = scaler.transform(input_array)
            
            # Reshape for LSTM model [samples, timesteps, features]
            reshaped_input = scaled_input.reshape(1, 3, 9) # 1 sample, 3 time steps, 9 features
            
            # Make prediction
            prediction = model.predict(reshaped_input)
            predicted_calories = int(prediction[0][0])
            
            st.success(f"### Predicted Calories: **{predicted_calories} kcal**")
            st.info("This prediction is based on the temporal patterns of your previous meals.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.warning("Could not load the prediction model.")