import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import pvlib
from pvlib.location import Location

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Solar AI Predictor", page_icon="☀️", layout="wide")

# --- 1. LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_model():
    # We use a try/except block to handle cases where the model file is missing
    try:
        return pickle.load(open('solar_model.pkl', 'rb'))
    except FileNotFoundError:
        st.error("⚠️ Model file 'solar_model.pkl' not found. Please run 'train_model.py' first.")
        st.stop()

model = load_model()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("⚙️ System Configuration")

# 1. Solar Panel Details
capacity_kw = st.sidebar.number_input("System Capacity (kW)", value=5.0, step=0.5, help="Example: 5kW system")
tilt_angle = st.sidebar.slider("Panel Tilt Angle (°)", 0, 90, 30, help="0=Flat on roof, 90=Vertical wall")
azimuth_angle = st.sidebar.slider("Azimuth Angle (°)", 0, 360, 180, help="180 = South (Best for Northern Hemisphere)")

# 2. Location (Default: New Delhi)
st.sidebar.header("📍 Location")
lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.4f")

# --- MAIN DASHBOARD ---
st.title("☀️ Solar Power AI Predictor")
st.markdown(f"**Forecasting for:** Latitude {lat}, Longitude {lon}")
st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("🚀 Get Live Forecast & Predict"):
    
    with st.spinner("Connecting to Weather Satellites..."):
        try:
            # 1. FETCH LIVE WEATHER FORECAST (Open-Meteo Free API)
            # We fetch: Temperature, Humidity, Cloud Cover, Direct Sun (DNI), Diffuse Sun (DHI)
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,direct_normal_irradiance,diffuse_radiation",
                "timezone": "auto",
                "forecast_days": 1
            }
            
            response = requests.get(url, params=params)
            data_json = response.json()
            
            # Create a DataFrame from the JSON response
            df = pd.DataFrame({
                'time': pd.to_datetime(data_json['hourly']['time']),
                'temperature': data_json['hourly']['temperature_2m'],
                'humidity': data_json['hourly']['relative_humidity_2m'],
                'cloud_cover': data_json['hourly']['cloud_cover'],
                'dni': data_json['hourly']['direct_normal_irradiance'],
                'dhi': data_json['hourly']['diffuse_radiation']
            })

            # 2. PHYSICS ENGINE: CALCULATE "EFFECTIVE IRRADIANCE"
            # The AI needs to know how much sun hits the *tilted* panel, not just the ground.
            
            site = Location(lat, lon)
            solar_pos = site.get_solarposition(df['time'])
            
            # Use PVLib to calculate Plane of Array (POA) Irradiance
            poa_data = pvlib.irradiance.get_total_irradiance(
                surface_tilt=tilt_angle,
                surface_azimuth=azimuth_angle,
                dni=df['dni'],
                ghi=df['dni'], # Approximation if GHI is missing, usually fine for forecast
                dhi=df['dhi'],
                solar_zenith=solar_pos['apparent_zenith'],
                solar_azimuth=solar_pos['azimuth']
            )
            
            # Add the calculated physics data to our table
            df['Effective_Irradiance'] = poa_data['poa_global']
            
            # 3. AI ENGINE: PREDICT POWER OUTPUT
            # Prepare the exact columns the AI was trained on
            # Note: The model expects specific column names!
            features = df[['Effective_Irradiance', 'temperature', 'humidity', 'cloud_cover']]
            
            # Run the prediction
            raw_prediction_kw = model.predict(features)
            
            # Scale the result: The model was trained on a 5kW system.
            # We adjust it based on the user's actual capacity.
            # Formula: (Prediction / 5.0) * User_Capacity
            df['Predicted_Power_kW'] = (raw_prediction_kw / 5.0) * capacity_kw
            
            # Remove negative predictions (night time noise)
            df['Predicted_Power_kW'] = df['Predicted_Power_kW'].clip(lower=0)

            # --- DISPLAY RESULTS ---
            
            # A. Summary Metrics
            total_energy_kwh = df['Predicted_Power_kW'].sum()
            peak_power_kw = df['Predicted_Power_kW'].max()
            avg_cloud = df['cloud_cover'].mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Energy (Today)", f"{total_energy_kwh:.2f} kWh", delta="Forecast")
            col2.metric("Peak Power Output", f"{peak_power_kw:.2f} kW")
            col3.metric("Avg Cloud Cover", f"{avg_cloud:.1f} %", delta_color="inverse")
            
            # B. The Graph
            st.subheader("📈 Hourly Power Generation Curve")
            st.area_chart(df.set_index('time')['Predicted_Power_kW'], color="#FFC300")
            
            # C. Data Table
            with st.expander("🔍 View Detailed Hourly Data"):
                st.dataframe(df[['time', 'temperature', 'cloud_cover', 'Predicted_Power_kW']].style.format({
                    'temperature': "{:.1f}°C",
                    'cloud_cover': "{:.0f}%",
                    'Predicted_Power_kW': "{:.2f} kW"
                }))
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Please check your internet connection or try a different location.")

else:
    st.info("👈 Adjust parameters in the sidebar and click the button to predict.")