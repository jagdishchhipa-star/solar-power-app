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
    try:
        return pickle.load(open('solar_model.pkl', 'rb'))
    except FileNotFoundError:
        st.error("⚠️ Model file 'solar_model.pkl' not found. Please run 'train_model.py' first.")
        st.stop()

model = load_model()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("⚙️ System Configuration")
capacity_kw = st.sidebar.number_input("System Capacity (kW)", value=5.0, step=0.5)
tilt_angle = st.sidebar.slider("Panel Tilt Angle (°)", 0, 90, 30)
azimuth_angle = st.sidebar.slider("Azimuth Angle (°)", 0, 360, 180)

st.sidebar.header("📍 Location")
lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.4f")

# --- MAIN DASHBOARD ---
st.title("☀️ Solar Power AI Predictor")
st.markdown(f"**Forecasting for:** {lat}, {lon}")

if st.button("🚀 Get Live Forecast & Predict"):
    
    with st.spinner("Connecting to Weather Satellites..."):
        try:
            # 1. FETCH LIVE WEATHER FORECAST
            # FIXED: Added 'shortwave_radiation' to get GHI correctly
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,direct_normal_irradiance,diffuse_radiation,shortwave_radiation",
                "timezone": "auto",
                "forecast_days": 1
            }
            
            response = requests.get(url, params=params)
            data_json = response.json()
            
            # Create DataFrame
            df = pd.DataFrame({
                'time': pd.to_datetime(data_json['hourly']['time']),
                'temperature': data_json['hourly']['temperature_2m'],
                'humidity': data_json['hourly']['relative_humidity_2m'],
                'cloud_cover': data_json['hourly']['cloud_cover'],
                'dni': data_json['hourly']['direct_normal_irradiance'], # Direct Sun
                'dhi': data_json['hourly']['diffuse_radiation'],      # Diffuse (Cloud) Light
                'ghi': data_json['hourly']['shortwave_radiation']     # Global Light (FIX)
            })

            # 2. PHYSICS ENGINE
            # We need to correctly handle timezones for the Sun Position
            tz_str = data_json.get('timezone', 'UTC')
            df['time'] = df['time'].dt.tz_localize(tz_str)
            
            site = Location(lat, lon, tz=tz_str)
            solar_pos = site.get_solarposition(df['time'])
            
            # FIXED: Passed the correct 'ghi' to the engine
            poa_data = pvlib.irradiance.get_total_irradiance(
                surface_tilt=tilt_angle,
                surface_azimuth=azimuth_angle,
                dni=df['dni'],
                ghi=df['ghi'], # Now using real GHI from API
                dhi=df['dhi'],
                solar_zenith=solar_pos['apparent_zenith'],
                solar_azimuth=solar_pos['azimuth']
            )
            
            df['Effective_Irradiance'] = poa_data['poa_global']
            
            # 3. AI ENGINE
            features = df[['Effective_Irradiance', 'temperature', 'humidity', 'cloud_cover']]
            
            raw_prediction_kw = model.predict(features)
            
            # Scale result
            df['Predicted_Power_kW'] = (raw_prediction_kw / 5.0) * capacity_kw
            df['Predicted_Power_kW'] = df['Predicted_Power_kW'].clip(lower=0)

            # --- DISPLAY RESULTS ---
            total_energy_kwh = df['Predicted_Power_kW'].sum()
            peak_power_kw = df['Predicted_Power_kW'].max()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Energy", f"{total_energy_kwh:.2f} kWh")
            col2.metric("Peak Power", f"{peak_power_kw:.2f} kW")
            col3.metric("Avg Cloud Cover", f"{df['cloud_cover'].mean():.0f}%")
            
            st.area_chart(df.set_index('time')['Predicted_Power_kW'])
            
            with st.expander("View Data"):
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Error: {e}")