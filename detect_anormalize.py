import pandas as pd
import matplotlib.pyplot as plt
import joblib  
import os

# ==========================================
# 1. Preparation: Load data and trained model
# ==========================================
print("Loading data...")
# Read cleaned solar dataset
df = pd.read_csv('./Datasets/cleaned_solar_data.csv')
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])  

# Path to pre-trained Random Forest model
model_path = './Model/solar_model_rf.pkl'

# Check if model exists, load or exit with error
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded successfully: {model_path}")
else:
    print(f"Error: File {model_path} not found. Run step2_random_forest.py first.")
    exit()

# ==========================================
# 2. Target Inverter (match training inverter)
# ==========================================
target_inverter = df['SOURCE_KEY_x'].unique()[0]
print(f"Analyzing inverter: {target_inverter}")

data = df[df['SOURCE_KEY_x'] == target_inverter].copy()

print(" Simulating artificial faults...")
# Reduce AC_POWER by 90% (×0.1) for a time segment to mimic fault
power_col_idx = data.columns.get_loc('AC_POWER')
data.iloc[100:120, power_col_idx] *= 0.1

# ==========================================
# 3. AI Prediction (calculate ideal power)
# ==========================================
# Feature list (consistent with training)
features = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']

# Predict theoretical power based on weather conditions
data['Pred_Power'] = model.predict(data[features])

# ==========================================
# 4. Core Logic: Residual Calculation (Fault Detection)
# ==========================================
data['Residual'] = data['Pred_Power'] - data['AC_POWER']

# Threshold: Define anomaly (power deficit > 500 kW)
threshold = 500 
data['Anomalies'] = data['Residual'] > threshold
num_anomalies = data['Anomalies'].sum()
print(f"Analysis completed! Total anomalies detected: {num_anomalies}")

# ==========================================
# 5. Visualization (Key for presentation)
# ==========================================
plot_data = data[data['DATE_TIME'] < (data['DATE_TIME'].min() + pd.Timedelta(days=4))]

plt.figure(figsize=(15, 6))
plt.plot(plot_data['DATE_TIME'], plot_data['Pred_Power'], 
         label='AI Predicted (Ideal)', color='blue', alpha=0.6, linewidth=2)
plt.plot(plot_data['DATE_TIME'], plot_data['AC_POWER'], 
         label='Actual Power', color='orange', alpha=0.6, linewidth=2)
anomalies = plot_data[plot_data['Anomalies']]
plt.scatter(anomalies['DATE_TIME'], anomalies['AC_POWER'], 
            color='red', label='Anomaly Detected', zorder=5, s=50, edgecolors='black')

plt.title(f'Solar Fault Detection (Random Forest) - Inverter: {target_inverter}')
plt.xlabel('Time')
plt.ylabel('AC Power (kW)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot locally
plt.savefig('rf_anomaly_result.png')
print("Plot saved as 'rf_anomaly_result.png' - check it out!")

plt.show()