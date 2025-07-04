#wind_energy_analysis.py
# Wind flow trend for wind energy purpose using wind_speed of four cities

import pandas as pd
import matplotlib.pyplot as plt

cities = ['Pune', 'Mumbai', 'Aurangabad', 'Nagpur']
wind_data = {}

# Collect average hourly wind speed for each city
for city in cities:
    df = pd.read_csv(f"./data/{city}_cleaned_data.csv")
    
    # Make sure 'hour' exists
    if 'hour' in df.columns:
        hourly_avg = df.groupby('hour')['wind_speed'].mean()
        wind_data[city] = hourly_avg

# Plot all on one graph
plt.figure(figsize=(10, 6))
for city, speed in wind_data.items():
    plt.plot(speed.index, speed.values, label=city)

plt.title("Average Hourly Wind Speed - City Comparison")
plt.xlabel("Hour of Day")
plt.ylabel("Wind Speed (m/s)")
plt.legend()
plt.grid(True)
plt.show()
