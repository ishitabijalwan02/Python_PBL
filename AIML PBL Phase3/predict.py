import pickle
import numpy as np

# Load models
with open('kmeans_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('pca.pkl', 'rb') as file:
    pca = pickle.load(file)
    print("Enter the following values:")

rainfall = float(input("Rainfall (mm): "))
temperature = float(input("Temperature (°C): "))
soil_moisture = float(input("Soil Moisture (%): "))
avg_yield = float(input("Avg Yield (ton/ha): "))
# Create input data

new_data = [[rainfall, temperature, soil_moisture, avg_yield]]
scaled = scaler.transform(new_data)
new_pca = pca.transform(scaled)

# Predict cluster

cluster = model.predict(scaled)

print("Predicted Cluster:", cluster[0])

