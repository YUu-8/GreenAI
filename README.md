# GreenAI
# 🌞 Green AI: Solar Anomaly Detection System

## 📖 Project Overview
This project aligns with **Green AI** principles by focusing on **Energy Production** and **Efficiency**. Instead of standard power forecasting, we implement an **Anomaly Detection** model for photovoltaic (PV) systems using the Kaggle *Solar Power Generation Dataset*.

By training a Machine Learning model to simulate the "ideal" power output based on weather conditions and comparing it to actual generation, we identify inefficiencies (faults, soiling, or shading) to reduce energy waste.

---

## 🛠️ Implementation Roadmap

### Phase 1: Data Preparation 🧹
**Goal:** Merge and clean the raw data for modeling.
1.  **Data Ingestion:** Load the `Generation_Data.csv` and `Weather_Sensor_Data.csv`.
2.  **Preprocessing:** Convert `DATE_TIME` columns to datetime objects for proper synchronization.
3.  **Data Merging:** Join the two datasets on the `DATE_TIME` key.
    * *Result:* A unified dataset containing Irradiation, Temperature, and Power Output for specific inverters.
4.  **Filtering:** Isolate data for a single inverter (using `SOURCE_KEY`) or aggregate data to simplify the initial analysis.

### Phase 2: Baseline Modeling 🤖
**Goal:** Train the AI to understand "normal" behavior.
1.  **Feature Selection:**
    * **Inputs (X):** `IRRADIATION` (Primary), `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`.
    * **Target (Y):** `AC_POWER`.
2.  **Model Selection:** Use a Regression model (e.g., **Random Forest Regressor** or **XGBoost**) capable of capturing non-linear relationships.
3.  **Training:** Split data into Train/Test sets (e.g., 80/20) and train the model to map Weather $\rightarrow$ Ideal Power.

### Phase 3: Residual Analysis (The Core Logic) 🔍
**Goal:** Detect anomalies by calculating the deviation from the ideal.
1.  **Prediction:** Generate predictions (`Pred_Power`) using the test set.
2.  **Residual Calculation:** Compute the difference between the AI's prediction and reality.
    * $$Residual = Predicted\_Power - Actual\_Power$$
3.  **Thresholding:** Define an "Anomaly":
    * If $Residual > Threshold$ (e.g., 20% deviation), flag the data point as a **Fault**.

### Phase 4: Visualization 📊
**Goal:** Visual proof of the model's performance.
1.  **Time-Series Plot:** Plot `Actual Power` (Orange) vs. `Predicted Power` (Blue) over time.
2.  **Highlight Anomalies:** Overlay Red points where the Residual exceeds the threshold to visualize detected faults.

---

## 💻 Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`

---

## 🚀 Quick Start (Logic Demo)

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1. Load & Merge Data
gen_df = pd.read_csv("Plant_1_Generation_Data.csv")
weather_df = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")

# Format Time
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'])
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])

# Merge on time (and optionally closest time match)
df = pd.merge(gen_df, weather_df, on='DATE_TIME')

# 2. Define Features
features = ['IRRADIATION', 'MODULE_TEMPERATURE']
target = 'AC_POWER'

# 3. Train Baseline Model (The "Ideal" State)
model = RandomForestRegressor()
model.fit(df[features], df[target])

# 4. Detect Anomalies
df['prediction'] = model.predict(df[features])
df['residual'] = df['prediction'] - df['AC_POWER']

# Flag if actual power is significantly lower than predicted (e.g., > 500 units difference)
anomalies = df[df['residual'] > 500]

print(f"Detected {len(anomalies)} potential faults.")
