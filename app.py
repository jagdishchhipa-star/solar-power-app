import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import pvlib
from pvlib.location import Location

# --- CONFIG ---
st.set_page_config(page_title="Solar AI", page_icon="⚡")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return pickle.load(open('solar_model.pkl', 'rb'))

try:
    model = load_model()
except:
    st.error("Model missing. Run 'train_model.py' first!")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
capacity = st.sidebar.number_input("System Size (kW)", value=5.0)
tilt = st.sidebar.slider("Tilt Angle", 0, 90, 30)
azimuth = st.sidebar.slider("Azimuth (180=South)", 0, 360, 180)

st.sidebar.header("📍 Location")
# Default: New Delhi
lat = st.sidebar.number_input("Latitude", value=28.6139) 
lon = st.sidebar.number_input("Longitude", value=77.2090)

# --- MAIN LOGIC ---
st.title("⚡ Solar Power Predictor (Fixed)")

if st.button("Predict Now"):
    with st.spinner("Fetching Data & Fixing Timezones..."):
        
        # 1. Get Weather (including Shortwave Radiation for GHI)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,direct_normal_irradiance,diffuse_radiation,shortwave_radiation",
            "timezone": "auto", # This asks API for Local Time
            "forecast_days": 1
        }
        resp = requests.get(url, params=params).json()
        
        # 2. Process Timezone Correctly
        # Open-Meteo returns local time string (e.g., "2023-12-30T12:00")
        # We must tell Python that this IS 'Asia/Kolkata' (or whatever local tz is)
        local_tz = resp.get('timezone', 'UTC')
        
        df = pd.DataFrame({
            'time': pd.to_datetime(resp['hourly']['time']),
            'temp': resp['hourly']['temperature_2m'],
            'humidity': resp['hourly']['relative_humidity_2m'],
            'cloud': resp['hourly']['cloud_cover'],
            'dni': resp['hourly']['direct_normal_irradiance'],
            'dhi': resp['hourly']['diffuse_radiation'],
            'ghi': resp['hourly']['shortwave_radiation'] # CRITICAL FIX
        })
        
        # Localize the time so PVLib knows where the sun is
        df['time'] = df['time'].dt.tz_localize(local_tz, ambiguous='NaT', nonexistent='shift_forward')
        
        # 3. Calculate Physics (Sun Position)
        site = Location(lat, lon, tz=local_tz)
        sol_pos = site.get_solarposition(df['time'])
        
        # DEBUG: Check Sun Altitude
        # If Altitude is < 0, it is Night.
        df['Sun_Altitude'] = sol_pos['apparent_elevation'].values
        
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            dni=df['dni'],
            ghi=df['ghi'],
            dhi=df['dhi'],
            solar_zenith=sol_pos['apparent_zenith'],
            solar_azimuth=sol_pos['azimuth']
        )
        
        df['Effective_Irradiance'] = poa['poa_global']
        
        # 4. AI Prediction
        # Rename columns to match model
        X = df[['Effective_Irradiance', 'temp', 'humidity', 'cloud']]
        X.columns = ['Effective_Irradiance', 'temperature', 'humidity', 'cloud_cover']
        
        pred = model.predict(X)
        df['Predicted_Power_kW'] = (pred / 5.0) * capacity
        df['Predicted_Power_kW'] = df['Predicted_Power_kW'].clip(lower=0)

        # --- DISPLAY ---
        total = df['Predicted_Power_kW'].sum()
        peak = df['Predicted_Power_kW'].max()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Energy", f"{total:.2f} kWh")
        col2.metric("Peak Power", f"{peak:.2f} kW")
        col3.metric("Timezone Used", local_tz)

        st.area_chart(df.set_index('time')['Predicted_Power_kW'])
        
        # Debug Table to prove Sun Altitude
        with st.expander("🔍 View Debug Data (Check Sun Altitude)"):
            st.write("If 'Sun_Altitude' is negative, it means Night.")
            st.dataframe(df[['time', 'Sun_Altitude', 'ghi', 'Predicted_Power_kW']])