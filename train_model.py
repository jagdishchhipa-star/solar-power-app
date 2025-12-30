import pandas as pd
import numpy as np
import requests
import pvlib
from pvlib.location import Location
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import json

# --- CONFIGURATION ---
# Coordinates for training (using a sunny location like Phoenix, AZ or New Delhi)
LAT = 28.6139 
LON = 77.2090
START_DATE = "20230101" # NASA Format YYYYMMDD
END_DATE = "20230228"   # Grabbing 2 months for speed (NASA API is slower than others)

print("1. Connecting to NASA POWER Satellite Database...")
print("   (This might take 30-60 seconds, please wait...)")

# NASA POWER API Endpoint
url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
params = {
    "parameters": "T2M,RH2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,ALLSKY_SFC_SW_DIFF",
    "community": "RE", # Renewable Energy
    "longitude": LON,
    "latitude": LAT,
    "start": START_DATE,
    "end": END_DATE,
    "format": "JSON"
}

try:
    response = requests.get(url, params=params, timeout=60)
    data_json = response.json()
    
    # Check if NASA returned data
    if 'properties' not in data_json:
        print("Error: NASA API did not return data. Try different dates.")
        exit()

    # --- PARSING NASA JSON TO DATAFRAME ---
    # NASA returns data as a dictionary of "Timestamp: Value", we need to convert this.
    params_data = data_json['properties']['parameter']
    
    df = pd.DataFrame({
        'temperature': pd.Series(params_data['T2M']),
        'humidity': pd.Series(params_data['RH2M']),
        'ghi': pd.Series(params_data['ALLSKY_SFC_SW_DWN']), # Global Horizontal Irradiance
        'dhi': pd.Series(params_data['ALLSKY_SFC_SW_DIFF']), # Diffuse
        'cloud_cover': pd.Series(params_data['CLOUD_AMT'])
    })

    # The index is YYYYMMDDHH, convert to datetime object
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H')
    df['time'] = df.index
    
    # NASA uses -999 for missing data. Replace with NaN and drop.
    df = df.replace(-999, np.nan).dropna()
    
    # Conversion: NASA gives DHI but not DNI (Direct Normal). We approximate DNI.
    # DNI = (GHI - DHI) / cos(Zenith). 
    # For training, we will rely on GHI (Global) as our main "Irradiance" feature.
    # We rename 'ghi' to 'dni' for the physics engine compatibility or calculate DNI properly.
    
    print(f"   Success! Downloaded {len(df)} hours of NASA satellite data.")

except Exception as e:
    print(f"   NASA API Connection Failed: {e}")
    print("   Please check your internet connection.")
    exit()

# --- PHYSICS LAYER: CALCULATE 'EFFECTIVE' IRRADIANCE ---
print("2. Calculating Solar Physics (Tilt & Azimuth)...")

# Define Location
site = Location(LAT, LON, tz='UTC') # NASA data is usually UTC
solar_position = site.get_solarposition(df['time'])

tilt = 35
azimuth = 180

# We use pvlib to calculate how much of that NASA sunlight actually hits the panel
# Since NASA didn't give DNI directly, we use the GHI-based transposition model (Hay-Davies)
poa_irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    dni=df['ghi'], # Using GHI as a proxy for total available sun intensity for robustness
    ghi=df['ghi'],
    dhi=df['dhi'],
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth']
)

# We use the calculated "Plane of Array" global irradiance
df['Effective_Irradiance'] = poa_irradiance['poa_global']

# --- GENERATE TARGET VARIABLE (POWER OUTPUT) ---
panel_capacity_kw = 5.0
temp_coeff = -0.004

# Efficiency loss due to heat
temp_loss = 1 + (df['temperature'] - 25) * temp_coeff
# Calculate Power
df['Power_Output'] = (df['Effective_Irradiance'] / 1000) * panel_capacity_kw * temp_loss
df['Power_Output'] = df['Power_Output'].fillna(0).clip(lower=0)

# --- AI TRAINING ---
print("3. Training the NASA-Powered Model...")

# Features required for prediction
X = df[['Effective_Irradiance', 'temperature', 'humidity', 'cloud_cover']]
y = df['Power_Output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save
pickle.dump(model, open('solar_model.pkl', 'wb'))
print("Success! NASA-Trained Model Saved as 'solar_model.pkl'.")