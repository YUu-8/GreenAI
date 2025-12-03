import pandas as pd
import numpy as np
import os  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib  
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = './Datasets/cleaned_solar_data.csv'
print(f"Loading data from: {file_path} ...")
df = pd.read_csv(file_path)
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])  # Convert to datetime

# Use only the first inverter (consistent with training)
target_inverter = df['SOURCE_KEY_x'].unique()[0]
print(f"Training target Inverter ID: {target_inverter}")
df_inverter = df[df['SOURCE_KEY_x'] == target_inverter].copy()

# Feature-target split
features = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
target = 'AC_POWER'
X = df_inverter[features]  
y = df_inverter[target]    

# Train-test split (80-20, fixed random state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Training done!")

# Model evaluation (train + test sets)
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate metrics
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print evaluation results
print("\n" + "-"*30)
print(f"Train Set:")
print(f"  R²: {r2_train:.4f} | RMSE: {rmse_train:.2f} kW | MAE: {mae_train:.2f} kW")
print(f"Test Set:")
print(f"  R²: {r2_test:.4f} | RMSE: {rmse_test:.2f} kW | MAE: {mae_test:.2f} kW")
print("-"*30)

# Save trained model
model_dir = './Model'  
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created model folder: {model_dir}")

save_path = os.path.join(model_dir, 'solar_model_rf.pkl')
joblib.dump(rf_model, save_path)
print(f"Model saved to: {save_path}")

# Anomaly detection (same logic as linear regression)
residuals = np.abs(y_test - y_test_pred)  # Absolute residuals
threshold = residuals.mean() + 2 * residuals.std()  # Dynamic threshold
anomalies = residuals > threshold
n_anomalies = anomalies.sum()
anomaly_pct = (n_anomalies / len(y_test)) * 100

print("\nAnomaly Detection Results:")
print(f"Threshold: {threshold:.2f} kW")
print(f"Anomalies found: {n_anomalies} ({anomaly_pct:.2f}%)")
print("\nPossible causes of anomalies:")
print("  - Hardware failures | Panel fouling")
print("  - Temporary shading | PV cell degradation")

# Plotting (match linear regression's style)
print("\nGenerating visualizations...")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Random Forest - Photovoltaic System', fontsize=16, fontweight='bold')

# 1. Train: Real vs Predicted
axes[0,0].scatter(y_train, y_train_pred, alpha=0.5, s=10)
axes[0,0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0,0].set_xlabel('AC Power Real (kW)', fontsize=10)
axes[0,0].set_ylabel('AC Power Predict (kW)', fontsize=10)
axes[0,0].set_title(f'Train Set (R²={r2_train:.4f})', fontsize=12)
axes[0,0].grid(alpha=0.3)

# 2. Test: Real vs Predicted (with anomalies)
normal_mask = ~anomalies
axes[0,1].scatter(y_test[normal_mask], y_test_pred[normal_mask], alpha=0.5, s=10, c='green', label='Normal')
axes[0,1].scatter(y_test[anomalies], y_test_pred[anomalies], alpha=0.7, s=30, c='red', label='Anomalies', marker='x')
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,1].set_xlabel('AC Power Real (kW)', fontsize=10)
axes[0,1].set_ylabel('AC Power Predict (kW)', fontsize=10)
axes[0,1].set_title(f'Test Set (R²={r2_test:.4f})', fontsize=12)
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# 3. Residuals Distribution
axes[1,0].hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[1,0].axvline(threshold, color='red', linestyle='--', lw=2, label=f'Threshold: {threshold:.2f} kW')
axes[1,0].set_xlabel('Residual |Real - Predict| (kW)', fontsize=10)
axes[1,0].set_ylabel('Frequency', fontsize=10)
axes[1,0].set_title('Residuals Distribution', fontsize=12)
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# 4. Residuals vs Irradiation
axes[1,1].scatter(X_test['IRRADIATION'], residuals, alpha=0.5, s=10, c=anomalies, cmap='RdYlGn_r')
axes[1,1].axhline(threshold, color='red', linestyle='--', lw=2, label=f'Threshold: {threshold:.2f} kW')
axes[1,1].set_xlabel('Irradiation (W/m²)', fontsize=10)
axes[1,1].set_ylabel('Residual (kW)', fontsize=10)
axes[1,1].set_title('Residuals vs Irradiation', fontsize=12)
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('solar_regression_results_rf.png', dpi=300, bbox_inches='tight')

# Save results to CSV
results_df = pd.DataFrame({
    'AC_POWER_REAL': y_test.values,
    'AC_POWER_PREDICTED': y_test_pred,
    'RESIDUAL': residuals,
    'IS_ANOMALY': anomalies,
    'IRRADIATION': X_test['IRRADIATION'].values,
    'AMBIENT_TEMPERATURE': X_test['AMBIENT_TEMPERATURE'].values,
    'MODULE_TEMPERATURE': X_test['MODULE_TEMPERATURE'].values
})
results_df.to_csv('solar_regression_predictions_rf.csv', index=False)

metrics_df = pd.DataFrame({
    'Model': ['Random Forest'],
    'Train_R2': [r2_train], 'Train_RMSE': [rmse_train], 'Train_MAE': [mae_train],
    'Test_R2': [r2_test], 'Test_RMSE': [rmse_test], 'Test_MAE': [mae_test],
    'Anomalies': [n_anomalies], 'Anomaly_Percentage': [anomaly_pct]
})
metrics_df.to_csv('solar_regression_metrics_rf.csv', index=False)

plt.show()
print("\nFiles saved:")
print("  - Plot: solar_regression_results_rf.png")
print("  - Predictions: solar_regression_predictions_rf.csv")
print("  - Metrics: solar_regression_metrics_rf.csv")