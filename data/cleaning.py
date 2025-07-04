#  for cleaning of data It is cleaning part of analytics.py
# It has data of four cities representing diff. meteorological regeions of Maharashtra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import zscore
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
import re

#def main():

# Skip metadata rows (e.g., first 2 or more lines depending on the file)
df = pd.read_csv("solarEnergy.csv", skiprows=2, usecols=range(50) )

# Preview
df.head()

# Extract metadata (first two rows as example)
with open("solarEnergy.csv", 'r') as f:
    lines = [next(f).strip() for _ in range(2)]

metadata = {}
for line in lines:
    if "," in line:
        key, value = line.split(",", 1)
        metadata[key.strip()] = value.strip()

# Convert metadata to a DataFrame if needed
meta_df = pd.DataFrame([metadata])


# Drop completely empty rows and columns
df_cleaned = df.dropna(axis=0, how='all').dropna(axis=1, how='all')


#print(df.head())
#print(df.tail())
#print(df.info())
#df.describe(include='all')

#make column name lower case and '_'
#df.columns = df.columns.strip()
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')


df.head()

# all spectarl length colmn in one sub set
#spectral_cols = [col for col in df.columns if 'um' in col]
spectral_cols = [col for col in df.columns if re.match(r'^\d+\.\d+_um$', col)]

spectral_df = df[spectral_cols]

print(spectral_cols)
print("Spectral subset shape:", spectral_df.shape)

# Check ranges# Plot spectral distribution for first row
spectral_df.iloc[0].plot(title="Spectral Irradiance for First Record", xlabel="Wavelength (um)", ylabel="Irradiance")

#print(spectral_df.describe())


# Step 1: Calculate % of nulls for each column
null_percent = df.isnull().mean() * 100

# Step 2: Drop columns with more than 80% null values
df = df.drop(columns=null_percent[null_percent > 80].index)

#  print dropped columns
print("Dropped columns due to high null percentage:")
print(null_percent[null_percent > 80])


#Remove them if they only contain one value 
units = {}
for col in df.columns:
    if 'unit' in col:
        if df[col].nunique() == 1:
            units[col] = df[col].unique()[0]
            df = df.drop(columns=col)

print("Stored Units:", units)


#If any column is text/string, check for typos or inconsistent values:
for col in df.select_dtypes(include='object'):
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))


# Spectral columns
#spectral_cols = [col for col in df.columns if '_um' in col]

# Material tech columns


print(df.dtypes)# Identify material columns
material_cols = [col for col in df.columns if '(' in col]

# Explicit columns to drop (only if they exist) refers 
other_cols = ['year', 'panel_tilt', 'panel_azimuth_angle']

# Combine and filter only those that exist
columns_to_drop = [col for col in (material_cols + other_cols) if col in df.columns]

# Drop them
df = df.drop(columns=columns_to_drop)

# Check the result
print("Remaining columns:")
print(df.columns)


print("Dropped columns:")
print(columns_to_drop)


 # check any missing values
print(df.isnull().sum())

# remove outliers values
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

num_cols = ['ghi', 'temperature', 'wind_speed']  # add more as needed
df_clean = remove_outliers_iqr(df, num_cols)


scaler = MinMaxScaler()
df_scaled = df_clean.copy()
df_scaled[num_cols] = scaler.fit_transform(df_clean[num_cols])


# columns you want to normalize
num_cols = ['ghi', 'temperature', 'wind_speed']  # can add more if needed

# make a copy and scale
scaler = MinMaxScaler()
df_scaled = df_clean.copy()
df_scaled[num_cols] = scaler.fit_transform(df_clean[num_cols])


sns.boxplot(data=df[['ghi', 'temperature']])
plt.title('Boxplot of Key Energy Features')
plt.xticks(rotation=45)
plt.show()

# saved cleaned data in new file
df_scaled.to_csv('cleaned_data.csv', index=False)


# Run this only if file is executed directly
#if __name__ == "__main__":
 #   main()

