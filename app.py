import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Solar Energy Predictor", page_icon="☀️", layout="centered")

# --- LOAD THE TRAINED MODEL ---
@st.cache_resource # Caches the model so it doesn't reload on every interaction
def load_model():
    with open('solar_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Please run 'train_model.py' first.")
    st.stop()

# --- HEADER SECTION ---
st.title("☀️ Solar Power Generation Predictor")
st.markdown("Enter weather conditions below to predict the energy output of your solar panel system.")
st.divider()

# --- INPUT SECTION (SIDEBAR) ---
st.sidebar.header("User Input Parameters")

def user_input_features():
    temp = st.sidebar.slider("Temperature (°C)", min_value=-10, max_value=50, value=30)
    humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=50)
    irradiance = st.sidebar.slider("Solar Irradiance (W/m²)", min_value=0, max_value=1200, value=800)
    cloud_cover = st.sidebar.slider("Cloud Cover (%)", min_value=0, max_value=100, value=20)
    
    data = {'Temperature': temp,
            'Humidity': humidity,
            'Irradiance': irradiance,
            'Cloud_Cover': cloud_cover}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- DISPLAY INPUTS ---
st.subheader("Current Weather Conditions")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature", f"{input_df['Temperature'][0]} °C")
col2.metric("Humidity", f"{input_df['Humidity'][0]} %")
col3.metric("Irradiance", f"{input_df['Irradiance'][0]} W/m²")
col4.metric("Cloud Cover", f"{input_df['Cloud_Cover'][0]} %")

# --- PREDICTION SECTION ---
if st.button("Predict Power Output"):
    prediction = model.predict(input_df)
    output = round(prediction[0], 2)
    
    st.divider()
    st.subheader("Prediction Result")
    
    # Visualizing the result with a gauge-style metric
    st.success(f"Estimated Power Generation: **{output} Watts**")
    
    # Contextual feedback
    if output > 300:
        st.info("✅ Excellent conditions for high energy generation.")
    elif output > 100:
        st.warning("⚠️ Moderate generation conditions.")
    else:
        st.error("❌ Poor conditions. Very low energy generation expected.")

# --- FOOTER ---
st.divider()
st.caption("Built with Python & Streamlit | Solar Prediction Project")