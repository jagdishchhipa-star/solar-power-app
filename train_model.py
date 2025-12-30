import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# --- SETTINGS (MADRID used for training "Standard Physics") ---
LAT, LON = 40.4168, -3.7038
TZ = 'Europe/Madrid'

print("1. Generating Synthetic Solar Data (Physics-Based)...")

# Generate 1 year of hourly data
times = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h', tz=TZ)
site = Location(LAT, LON, tz=TZ)

# Calculate Clear Sky (Theoretical Maximum Sun)
clear_sky = site.get_clearsky(times)
solar_position = site.get_solarposition(times)

# Create Dataset
df = pd.DataFrame({
    'time': times,
    'dni': clear_sky['dni'],
    'ghi': clear_sky['ghi'],
    'dhi': clear_sky['dhi'],
    'temperature': 25.0,  # Avg temp
    'humidity': 50.0,
    'cloud_cover': 0.0
})

# Add Random Cloud Noise (Teach AI about clouds)
np.random.seed(42)
cloud_events = np.random.uniform(0, 1, len(df))
df['cloud_cover'] = cloud_events * 100
# Physics: Clouds block Direct Sun (DNI) heavily, but diffuse light (GHI) remains partially
df['dni'] = df['dni'] * (1 - cloud_events * 0.95)
df['ghi'] = df['ghi'] * (1 - cloud_events * 0.60)

# Calculate Effective Irradiance on Tilted Panel
tilt = 30
azimuth = 180
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    dni=df['dni'],
    ghi=df['ghi'],
    dhi=df['dhi'],
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth']
)
df['Effective_Irradiance'] = poa['poa_global']

# Generate Target Power (5kW System)
df['Power_Output'] = (df['Effective_Irradiance'] / 1000) * 5.0
df['Power_Output'] = df['Power_Output'].fillna(0).clip(lower=0)
df = df.dropna()

print("2. Training Model...")
X = df[['Effective_Irradiance', 'temperature', 'humidity', 'cloud_cover']]
y = df['Power_Output']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

pickle.dump(model, open('solar_model.pkl', 'wb'))
print("✅ Success! New 'solar_model.pkl' created.")