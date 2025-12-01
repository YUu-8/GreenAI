import pandas as pd

# Load raw datasets
print("Loading data...")
gen_df = pd.read_csv('Datasets/Plant_1_Generation_Data.csv')
weather_df = pd.read_csv('Datasets/Plant_1_Weather_Sensor_Data.csv')

print(f"Original generation data shape: {gen_df.shape}")
print(f"Original weather data shape: {weather_df.shape}")

# Standardize datetime (handle dd-mm-yyyy format) and drop redundant column
print("\nConverting datetime formats...")
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], dayfirst=True)
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], dayfirst=True)

gen_df = gen_df.drop(columns=['PLANT_ID'])
weather_df = weather_df.drop(columns=['PLANT_ID'])

# Merge generation and weather data on datetime
print("\nMerging datasets...")
df_merged = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')
df_merged.to_csv(r'E:\GreenAI\Datasets\cleaned_solar_data.csv', index=False)

print("Merge completed!")
print(f"Final dataset shape: {df_merged.shape}")

# Preview and save cleaned data
print("\nFirst 5 rows of cleaned data:")
print(df_merged.head())

df_merged.to_csv('cleaned_solar_data.csv', index=False)
print("\nSaved as 'cleaned_solar_data.csv'")