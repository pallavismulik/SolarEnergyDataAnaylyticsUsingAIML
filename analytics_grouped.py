# grouped_analytics.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import os

# Load Cleaned Data
df = pd.read_csv("cleaned_data.csv")
df.columns = df.columns.str.lower()

# Create plots directory
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Columns of interest
num_cols = ['ghi', 'temperature', 'wind_speed']

# 1. HISTOGRAMS + NORMALITY TEST
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axs[i])
    axs[i].set_title(f"{col} distribution")
    stat, p = shapiro(df[col])
    axs[i].text(0.7, 0.9, f"p = {p:.3f}", transform=axs[i].transAxes)
plt.suptitle("Distributions with Normality Test (Shapiro-Wilk)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/distribution_normality_tests.png")
plt.show()
plt.close()

# 2. CORRELATION HEATMAP (Top 10 Features)
corr = df.corr().abs()
top_features = corr['ghi'].sort_values(ascending=False)[1:11].index
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm')
plt.title("Top 10 Feature Correlation with GHI")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/top10_corr_heatmap.png")
plt.show()
plt.close()

# 3. SCATTERPLOTS
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=df, x='temperature', y='ghi', alpha=0.5, ax=axs[0])
sns.regplot(data=df, x='temperature', y='ghi', scatter=False, color='red', ax=axs[0])
axs[0].set_title("Temperature vs GHI")

sns.scatterplot(data=df, x='wind_speed', y='ghi', alpha=0.5, ax=axs[1])
sns.regplot(data=df, x='wind_speed', y='ghi', scatter=False, color='green', ax=axs[1])
axs[1].set_title("Wind Speed vs GHI")

plt.suptitle("Effect of Temperature and Wind on GHI")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/scatter_temp_wind_vs_ghi.png")
plt.show()
plt.close()

# 4. MONTHLY GHI + DAY/NIGHT PIE CHART
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
monthly_ghi = df.groupby('month')['ghi'].mean()
sns.lineplot(x=monthly_ghi.index, y=monthly_ghi.values, marker='o', ax=axs[0])
axs[0].set_title("Monthly Average GHI")
axs[0].set_xlabel("Month")
axs[0].set_ylabel("GHI")

day = df[df['hour'].between(6, 18)]
night = df[~df['hour'].between(6, 18)]
axs[1].pie([day['ghi'].sum(), night['ghi'].sum()],
           labels=["Daytime", "Nighttime"], autopct='%1.1f%%', colors=['gold', 'gray'])
axs[1].set_title("GHI: Day vs Night")

plt.suptitle("Monthly Trend and Solar Contribution")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/monthly_trend_daynight.png")
plt.show()
plt.close()

# 5. TEMP + GHI Monthly Combo
monthly_stats = df.groupby('month')[['temperature', 'ghi']].mean().reset_index()
fig, ax1 = plt.subplots(figsize=(10, 5))

sns.barplot(data=monthly_stats, x='month', y='temperature', color='orange', ax=ax1, label='Temperature')
ax2 = ax1.twinx()
sns.lineplot(data=monthly_stats, x='month', y='ghi', color='blue', marker='o', ax=ax2, label='GHI')

ax1.set_title("Monthly Temperature (Bar) & GHI (Line)")
ax1.set_ylabel("Temperature (°C)")
ax2.set_ylabel("GHI")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/monthly_temp_ghi_combo.png")
plt.show()
plt.close()

# 6. GHI, DHI, DNI Heatmap (Optional)
if {'ghi', 'dhi', 'dni'}.issubset(df.columns):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[['ghi', 'dhi', 'dni']].corr(), annot=True, cmap="YlGnBu")
    plt.title("GHI, DHI, DNI Correlation")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/ghi_dhi_dni_corr.png")
    plt.show()
    plt.close()

