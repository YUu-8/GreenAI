import pandas as pd
import numpy as np
import os  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib  

# ==========================================
# 1. Load dataset
# ==========================================
file_path = './Datasets/cleaned_solar_data.csv'
print(f"Loading data from: {file_path} ...")

df = pd.read_csv(file_path)
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])  

# ==========================================
# 2. Filter data (use first inverter only)
# ==========================================
target_inverter = df['SOURCE_KEY_x'].unique()[0]
print(f"Current training target (Inverter ID): {target_inverter}")

df_subset = df[df['SOURCE_KEY_x'] == target_inverter].copy()

# ==========================================
# 3. Prepare features and target
# ==========================================
features = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
target = 'AC_POWER'

X = df_subset[features]  
y = df_subset[target]    

# Split into training and test sets (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. Train Random Forest model
# ==========================================
print("\nTraining Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Training completed!")

# ==========================================
# 5. Model evaluation
# ==========================================
y_pred = model.predict(X_test)  
r2 = r2_score(y_test, y_pred)   
mae = mean_absolute_error(y_test, y_pred)  

print("\n" + "="*30)
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} kW")
print("="*30)

# ==========================================
# 6. Save trained model
# ==========================================
model_dir = './Model'  
# Create directory if not exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created folder: {model_dir}")

save_path = os.path.join(model_dir, 'solar_model_rf.pkl')
joblib.dump(model, save_path)  # Save model as pickle file
print(f"Model saved successfully: {save_path}")