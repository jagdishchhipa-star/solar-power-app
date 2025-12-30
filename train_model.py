import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Generate Realistic Dummy Data
# (In a real project, you would load a CSV file here)
print("Generating dummy solar data...")
np.random.seed(42)
n_samples = 2000

# Random weather features
temperature = np.random.uniform(15, 45, n_samples)    # 15°C to 45°C
humidity = np.random.uniform(20, 90, n_samples)       # 20% to 90%
irradiance = np.random.uniform(0, 1000, n_samples)    # 0 to 1000 W/m²
cloud_cover = np.random.uniform(0, 100, n_samples)    # 0% to 100%

# Formula for generated power (Simulated Physics)
# Power increases with Irradiance, decreases with Heat and Clouds
efficiency = 0.18  # 18% panel efficiency
panel_area = 2.5   # 2.5 m²
power_generated = (irradiance * panel_area * efficiency)
# Efficiency drops as temperature rises above 25°C
heat_loss = (temperature - 25).clip(min=0) * 0.005 
power_generated = power_generated * (1 - heat_loss) * (1 - (cloud_cover/200))

# Add some randomness/noise
power_generated += np.random.normal(0, 15, n_samples)
power_generated = np.maximum(power_generated, 0) # No negative power

# Create DataFrame
df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Irradiance': irradiance,
    'Cloud_Cover': cloud_cover,
    'Power_Output': power_generated
})

# 2. Train the Model
print("Training the Model...")
X = df[['Temperature', 'Humidity', 'Irradiance', 'Cloud_Cover']]
y = df['Power_Output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Save the Model
filename = 'solar_model.pkl'
pickle.dump(model, open(filename, 'wb'))
print(f"Success! Model saved as '{filename}'.")