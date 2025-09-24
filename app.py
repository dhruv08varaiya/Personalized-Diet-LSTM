import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import os
import pandas as pd # Added for the explanation section

# --- Page Configuration ---
st.set_page_config(
    page_title="Your Personal Diet AI",
    page_icon="🤖",
    layout="wide"
)

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and scaler."""
    model_path = 'saved_model/lstm_calorie_predictor.h5'
    scaler_path = 'saved_model/scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# --- Main Application UI ---
st.title("🤖 Your Personal Diet AI")
st.markdown("A smart assistant to predict your next meal's calories based on your habits.")

# --- Explainer Section ---
with st.expander("🤔 How does this work? (Click to learn more)"):
    st.info(
        """
        This app uses a special type of AI called an **LSTM (Long Short-Term Memory) network**. Think of it as an AI with a good memory.
        
        1.  **It Learns Your Patterns**: The AI was trained on a diary of meals. It learned the relationships between the time of day, type of meal, and its nutritional content.
        2.  **It Remembers Sequences**: It specifically looks at the *sequence* of your last three meals, not just one meal in isolation. For example, it might learn that a light breakfast and a medium lunch often lead to a heavier dinner.
        3.  **It Predicts the Future**: Based on the sequence you provide, it makes an educated guess on the calories for your *next* meal.
        """
    )

if not model or not scaler:
    st.error("Model or scaler not found. Please run the `Diet_Prediction_LSTM.ipynb` notebook first to train and save them.")
else:
    # --- Step-by-Step Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([" B️reakfast (Meal 1)", " Lunch (Meal 2)", " Dinner (Meal 3)", " 🔮 Get Prediction"])
    
    input_data = []
    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # --- Meal 1 Input ---
    with tab1:
        st.header("🍳 Your First Meal (Oldest)")
        st.markdown("Enter the details of the first meal in your sequence (e.g., your breakfast).")
        
        protein1 = st.number_input("Protein (g)", min_value=0, value=20, key="prot_1", help="Found in foods like eggs, chicken, beans, and tofu. Helps build muscle.")
        carbs1 = st.number_input("Carbohydrates (g)", min_value=0, value=50, key="carb_1", help="Found in foods like bread, rice, and fruits. The body's main source of energy.")
        fat1 = st.number_input("Fat (g)", min_value=0, value=15, key="fat_1", help="Found in foods like nuts, avocados, and oils. Important for brain health.")
        
        st.divider()
        
        meal_type1 = st.selectbox("Meal Type", options=meal_types, key="meal_1")
        hour1 = st.slider("Hour of Day (24h format)", 0, 23, 8, key="hour_1")
        day_str1 = st.selectbox("Day of Week", options=days_of_week, index=datetime.today().weekday(), key="day_1")

    # --- Meal 2 Input ---
    with tab2:
        st.header("🥪 Your Second Meal")
        st.markdown("Now, enter the details for the next meal in your sequence (e.g., your lunch).")
        
        protein2 = st.number_input("Protein (g)", min_value=0, value=35, key="prot_2", help="Found in foods like eggs, chicken, beans, and tofu. Helps build muscle.")
        carbs2 = st.number_input("Carbohydrates (g)", min_value=0, value=40, key="carb_2", help="Found in foods like bread, rice, and fruits. The body's main source of energy.")
        fat2 = st.number_input("Fat (g)", min_value=0, value=20, key="fat_2", help="Found in foods like nuts, avocados, and oils. Important for brain health.")
        
        st.divider()
        
        meal_type2 = st.selectbox("Meal Type", options=meal_types, index=1, key="meal_2")
        hour2 = st.slider("Hour of Day (24h format)", 0, 23, 13, key="hour_2")
        day_str2 = st.selectbox("Day of Week", options=days_of_week, index=datetime.today().weekday(), key="day_2")

    # --- Meal 3 Input ---
    with tab3:
        st.header("🍝 Your Third Meal (Most Recent)")
        st.markdown("Finally, enter the details for your most recent meal (e.g., your dinner).")
        
        protein3 = st.number_input("Protein (g)", min_value=0, value=25, key="prot_3", help="Found in foods like eggs, chicken, beans, and tofu. Helps build muscle.")
        carbs3 = st.number_input("Carbohydrates (g)", min_value=0, value=90, key="carb_3", help="Found in foods like bread, rice, and fruits. The body's main source of energy.")
        fat3 = st.number_input("Fat (g)", min_value=0, value=10, key="fat_3", help="Found in foods like nuts, avocados, and oils. Important for brain health.")
        
        st.divider()
        
        meal_type3 = st.selectbox("Meal Type", options=meal_types, index=2, key="meal_3")
        hour3 = st.slider("Hour of Day (24h format)", 0, 23, 20, key="hour_3")
        day_str3 = st.selectbox("Day of Week", options=days_of_week, index=datetime.today().weekday(), key="day_3")
        
    # --- Prediction Tab ---
    with tab4:
        st.header("Get Your Personalized Prediction")
        
        if st.button("✨ Predict My Next Meal's Calories", type="primary", use_container_width=True):
            # Process inputs from all tabs
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

            try:
                input_array = np.array(input_data)
                scaled_input = scaler.transform(input_array)
                reshaped_input = scaled_input.reshape(1, 3, 9)
                
                prediction = model.predict(reshaped_input)
                predicted_calories = int(prediction[0][0])
                
                st.metric(label="Predicted Calorie Count", value=f"{predicted_calories} kcal")
                
                with st.expander("How was this prediction made?"):
                    st.write(f"""
                    The AI analyzed your input:
                    - **Meal 1 ({meal_type1})**: {protein1}g protein, {carbs1}g carbs, {fat1}g fat.
                    - **Meal 2 ({meal_type2})**: {protein2}g protein, {carbs2}g carbs, {fat2}g fat.
                    - **Meal 3 ({meal_type3})**: {protein3}g protein, {carbs3}g carbs, {fat3}g fat.
                    
                    Based on patterns it learned from thousands of meal sequences, it determined that this specific sequence most often leads to a next meal of around **{predicted_calories} calories**. 
                    This could be because the total energy intake from these three meals created a pattern that the AI recognized.
                    """)
            
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
