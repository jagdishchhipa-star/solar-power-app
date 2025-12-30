import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# --- SETTINGS ---
# We use Madrid coordinates for training (good sunlight variation)
LAT, LON = 40.4168, -3.7038
TZ = 'Europe/Madrid'

print("1. Generating Synthetic Scientific Data (No Internet Required)...")

# Create a time range for one year (Hourly)
times = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h', tz=TZ)

# Define the Location
site = Location(LAT, LON, tz=TZ)

# --- PHYSICS 1: CALCULATE THEORETICAL SUNLIGHT (CLEAR SKY) ---
# This calculates exactly how much sun there IS if there were no clouds
clear_sky = site.get_clearsky(times)
solar_position = site.get_solarposition(times)

# Create our dataset
df = pd.DataFrame({
    'time': times,
    'dni': clear_sky['dni'],
    'ghi': clear_sky['ghi'],
    'dhi': clear_sky['dhi'],
    'temperature': 20.0,  # Start with a base temp
    'humidity': 50.0,     # Start with base humidity
    'cloud_cover': 0.0    # Start with clear sky
})

# --- PHYSICS 2: ADD REALISTIC NOISE (SIMULATE WEATHER) ---
# We introduce random cloud events to teach the AI how to handle bad weather
np.random.seed(42)
n_rows = len(df)

# Generate random "cloud factors" (0 = Clear, 1 = Dark)
cloud_events = np.random.uniform(0, 1, n_rows)

# If cloud factor > 0.3, we reduce the sunlight intensity
# (Simple model: More clouds = Less Direct Normal Irradiance)
df['cloud_cover'] = cloud_events * 100
df['dni'] = df['dni'] * (1 - cloud_events * 0.9) # Direct sun drops heavily with clouds
df['ghi'] = df['ghi'] * (1 - cloud_events * 0.6) # Global sun drops less (ambient light)

# Add random temperature variation (Winter cooler, Summer warmer)
# Cosine wave for seasons + random daily fluctuation
day_of_year = df['time'].dt.dayofyear
seasonal_temp = 15 + 15 * -np.cos((day_of_year / 365) * 2 * np.pi) # 0 to 30C range
df['temperature'] = seasonal_temp + np.random.uniform(-5, 5, n_rows)

print(f"   Generated {len(df)} hours of physics-based training data.")

# --- PHYSICS 3: CALCULATE PANEL ANGLE (EFFECTIVE IRRADIANCE) ---
print("2. Calculating Solar Geometry...")

tilt = 35
azimuth = 180

# Calculate how much sun actually hits the angled panel
poa_irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    dni=df['dni'],
    ghi=df['ghi'],
    dhi=df['dhi'],
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth']
)

df['Effective_Irradiance'] = poa_irradiance['poa_global']

# --- GENERATE TARGET VARIABLE (POWER) ---
# The "Answer Key" for the AI
panel_capacity_kw = 5.0 
temp_coeff = -0.004

temp_loss = 1 + (df['temperature'] - 25) * temp_coeff
df['Power_Output'] = (df['Effective_Irradiance'] / 1000) * panel_capacity_kw * temp_loss
df['Power_Output'] = df['Power_Output'].fillna(0).clip(lower=0)

# Final cleanup
df = df.dropna()

# --- AI TRAINING ---
print("3. Training the AI Model...")

X = df[['Effective_Irradiance', 'temperature', 'humidity', 'cloud_cover']]
y = df['Power_Output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pickle.dump(model, open('solar_model.pkl', 'wb'))
print("Success! Model saved as 'solar_model.pkl'.")