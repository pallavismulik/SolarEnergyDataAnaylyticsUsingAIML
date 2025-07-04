# Cleaning of data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import shapiro, zscore
import re
import os

# --- Setup ---
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load CSV skipping metadata
csv_path = "solarEnergy.csv"
df = pd.read_csv(csv_path, skiprows=2)

# Extract metadata (first two rows)
with open(csv_path, 'r') as f:
    lines = [next(f).strip() for _ in range(2)]
metadata = {}
for line in lines:
    if "," in line:
        key, value = line.split(",", 1)
        metadata[key.strip()] = value.strip()
meta_df = pd.DataFrame([metadata])

# Clean and format
df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# --- Spectral Analysis ---
spectral_cols = [col for col in df.columns if re.match(r'^\d+\.\d+_um$', col)]
spectral_df = df[spectral_cols]
plt.figure()
spectral_df.iloc[0].plot(title="Spectral Irradiance for First Record", xlabel="Wavelength (um)", ylabel="Irradiance")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/spectral_irradiance.png")
plt.close()

# Drop high null percentage cols
null_percent = df.isnull().mean() * 100
df = df.drop(columns=null_percent[null_percent > 80].index)

# Drop unit columns with single unique value
units = {}
for col in df.columns:
    if 'unit' in col and df[col].nunique() == 1:
        units[col] = df[col].unique()[0]
        df = df.drop(columns=col)

# Drop material and non-predictive columns
material_cols = [col for col in df.columns if '(' in col]
other_cols = ['year', 'panel_tilt', 'panel_azimuth_angle']
columns_to_drop = [col for col in (material_cols + other_cols) if col in df.columns]
df = df.drop(columns=columns_to_drop)

# --- Remove outliers ---
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

num_cols = ['ghi', 'temperature']
df_clean = remove_outliers_iqr(df, num_cols)
df_clean = df_clean[df_clean['ghi'] > 0]


# Normalize
scaler = MinMaxScaler()
df_scaled = df_clean.copy()
df_scaled[num_cols] = scaler.fit_transform(df_clean[num_cols])
df_scaled.to_csv('cleaned_data.csv', index=False)


# Applay  analytics steps and work on differnt factors(column data)
# --- Visualizations ---
sns.boxplot(data=df[num_cols])
plt.title('Boxplot of Key Energy Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/boxplot_energy_features.png")
plt.show()
plt.close()

# Histograms
for col in num_cols:
    plt.figure()
    sns.histplot(df_scaled[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/distribution_{col}.png")
    plt.show()
    plt.close()

# Shapiro test to check whether Normal distribution or not
for col in num_cols:
    stat, p = shapiro(df_scaled[col])
    print(f"{col}: p-value = {p}")
    print("Probably normal" if p > 0.05 else "Probably not normal")

# Monthly GHI trend to view 
monthly_ghi = df.groupby('month')['ghi'].mean()
plt.figure()
sns.lineplot(x=monthly_ghi.index, y=monthly_ghi.values, marker='o')
plt.title("Average Monthly GHI")
plt.xlabel("Month")
plt.ylabel("GHI")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/monthly_avg_ghi.png")
plt.show()
plt.close()

# Temperature vs GHI
plt.figure()
sns.scatterplot(data=df, x='temperature', y='ghi', alpha=0.5)
sns.regplot(data=df, x='temperature', y='ghi', scatter=False, color='red')
plt.title("Temperature vs. GHI")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/temp_vs_ghi.png")
plt.close()

# Wind speed vs GHI
plt.figure()
sns.scatterplot(data=df, x='wind_speed', y='ghi', alpha=0.5)
sns.regplot(data=df, x='wind_speed', y='ghi', scatter=False, color='green')
plt.title("Wind Speed vs. GHI")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/wind_vs_ghi.png")
plt.show()
plt.close()

# Day vs Night GHI Pie
daytime = df[df['hour'].between(6, 18)]
nighttime = df[~df['hour'].between(6, 18)]
sizes = [daytime['ghi'].sum(), nighttime['ghi'].sum()]
labels = ['Daytime (6AM-6PM)', 'Nighttime']
plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['gold', 'gray'])
plt.title("Share of Solar Irradiance by Day vs. Night")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/day_vs_night_pie.png")
plt.show()
plt.close()

# Monthly temp vs GHI
monthly_stats = df.groupby('month')[['temperature', 'ghi']].mean().reset_index()
plt.figure()
sns.barplot(data=monthly_stats, x='month', y='temperature', color='orange', label='Temp (°C)')
sns.lineplot(data=monthly_stats, x='month', y='ghi', color='blue', marker='o', label='GHI')
plt.title("Monthly Average Temperature vs. GHI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/monthly_temp_vs_ghi.png")
plt.show()
plt.show()
plt.close()


# Extra visualizations
if set(['ghi', 'dhi', 'dni']).issubset(df.columns):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[['ghi', 'dhi', 'dni']].corr(), annot=True, cmap="YlGnBu")
    plt.title("GHI, DHI, DNI Correlation")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/ghi_dhi_dni_corr.png")
    plt.show()
    plt.close()

if set(['temperature', 'humidity']).issubset(df.columns):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='temperature', y='humidity', alpha=0.5)
    sns.regplot(data=df, x='temperature', y='humidity', scatter=False, color='red')
    plt.title("Temperature vs Humidity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/temp_vs_humidity.png")
    plt.show()
    plt.close()