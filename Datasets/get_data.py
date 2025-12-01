import kagglehub
import shutil
import os
import pandas as pd

target_dir = "./Datasets"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print("1. Downloading from Kaggle (will first save to system cache)...")
cache_path = kagglehub.dataset_download("anikannal/solar-power-generation-data")
print(f"   Cache path: {cache_path}")

print(f"2. Moving files to: {target_dir} ...")

for filename in os.listdir(cache_path):
    source_file = os.path.join(cache_path, filename)
    destination_file = os.path.join(target_dir, filename)
    shutil.copy2(source_file, destination_file)
    print(f"   Copied: {filename}")

print("Success! Files saved in Datasets folder.")

try:
    gen_df = pd.read_csv(os.path.join(target_dir, "Plant_1_Generation_Data.csv"))
    print(f"Read test successful. Data shape: {gen_df.shape}")
except Exception as e:
    print(f"Read error: {e}")