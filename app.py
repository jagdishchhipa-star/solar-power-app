import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import pvlib
from pvlib.location import Location

# --- CONFIGURATION ---
st.set_page_config(page_title="Pro Solar Estimator", page_icon="⚡", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return pickle.load(open('solar_model.pkl', 'rb'))

try:
    model = load_model()
except:
    st.error("Model not found. Run train_model.py first.")
    st.stop()

# --- SIDEBAR: SYSTEM SPECS ---
st.sidebar.header("1. Panel Configuration")
capacity = st.sidebar.number_input("System Size (kW)", value=5.0, step=0.5)
tilt = st.sidebar.slider("Tilt Angle (°)", 0, 90, 30, help="0=Flat, 90=Vertical Wall")
azimuth = st.sidebar.slider("Azimuth (°)", 0, 360, 180, help="180=South, 90=East, 270=West")

st.sidebar.header("2. Location")
lat = st.sidebar.number_input("Latitude", value=28.6139) # Default: New Delhi
lon = st.sidebar.number_input("Longitude", value=77.2090)

# --- MAIN APP ---
st.title("⚡ Advanced Solar Power Predictor")
st.markdown(f"**Forecasting for Location:** {lat}, {lon}")

if st.button("Get Live Forecast & Predict"):
    
    with st.spinner("Fetching Satellite Data & Calculating Physics..."):
        
        # 1. FETCH LIVE WEATHER FORECAST (Open-Meteo)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,direct_normal_irradiance,diffuse_radiation",
            "forecast_days": 1
        }
        
        try:
            response = requests.get(url, params=params).json()
            
            # Organize Data
            data = pd.DataFrame({
                'time': pd.to_datetime(response['hourly']['time']),
                'temp': response['hourly']['temperature_2m'],
                'humidity': response['hourly']['relative_humidity_2m'],
                'cloud': response['hourly']['cloud_cover'],
                'dni': response['hourly']['direct_normal_irradiance'],
                'dhi': response['hourly']['diffuse_radiation']
            })
            
        except Exception as e:
            st.error("Error fetching weather data. Check internet connection.")
            st.stop()

        # 2. PHYSICS ENGINE (Calculate "Plane of Array" Irradiance)
        # This converts "Sun in Sky" to "Sun on Panel" based on user inputs
        site = Location(lat, lon)
        solar_pos = site.get_solarposition(data['time'])
        
        poa_data = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            dni=data['dni'],
            ghi=data['dni']*0 + data['dhi'], # Simplification
            dhi=data['dhi'],
            solar_zenith=solar_pos['apparent_zenith'],
            solar_azimuth=solar_pos['azimuth']
        )
        
        data['Effective_Irradiance'] = poa_data['poa_global']

        # 3. AI ENGINE (Predict Output)
        # We scale the prediction based on the user's system size (Capacity)
        # The model was trained on a 5kW system, so we scale proportionally.
        
        features = data[['Effective_Irradiance', 'temp', 'humidity', 'cloud']]
        features.columns = ['Effective_Irradiance', 'temperature', 'humidity', 'cloud_cover'] # Rename to match training
        
        raw_prediction = model.predict(features)
        
        # Scale prediction: (Prediction / 5kW_Base) * User_Capacity
        scaled_prediction = (raw_prediction / 5.0) * capacity
        
        data['Predicted_Power_kW'] = scaled_prediction

        # --- DISPLAY RESULTS ---
        # Summary Metrics
        total_energy = data['Predicted_Power_kW'].sum()
        peak_power = data['Predicted_Power_kW'].max()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Energy (Today)", f"{total_energy:.1f} kWh")
        col2.metric("Peak Power", f"{peak_power:.1f} kW")
        col3.metric("Efficiency Score", f"{100 - data['cloud'].mean():.0f}%")
        
        # Chart
        st.subheader("Hourly Power Generation Curve")
        st.area_chart(data.set_index('time')['Predicted_Power_kW'])
        
        # Data Table
        with st.expander("View Detailed Hourly Data"):
            st.dataframe(data[['time', 'temp', 'Effective_Irradiance', 'Predicted_Power_kW']])