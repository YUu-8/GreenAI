import pandas as pd
import matplotlib.pyplot as plt
import joblib  # 用来加载模型
import os

# ==========================================
# 1. 准备工作：加载数据和模型
# ==========================================
print("正在加载数据...")
# 读取清洗后的数据
df = pd.read_csv('./Datasets/cleaned_solar_data.csv')
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

# 加载你在上一步训练好的随机森林模型
# 注意：这里路径要和你保存时的一致
model_path = './Model/solar_model_rf.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"✅ 成功加载模型: {model_path}")
else:
    print(f"❌ 错误：找不到文件 {model_path}，请先运行 step2_random_forest.py")
    exit()

# ==========================================
# 2. 锁定目标 (必须和训练时用同一个逆变器)
# ==========================================
# 我们之前用第一个逆变器训练的，现在也要用它来做检测
target_inverter = df['SOURCE_KEY_x'].unique()[0]
print(f"正在分析逆变器: {target_inverter}")

# 筛选出该逆变器的数据
data = df[df['SOURCE_KEY_x'] == target_inverter].copy()


# ==========================================
# 3. 让 AI 进行预测 (计算理想发电量)
# ==========================================
features = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']

# 让模型根据天气，算出“理论上应该发多少电”
data['Pred_Power'] = model.predict(data[features])

# ==========================================
# 4. 🔥 核心逻辑：计算残差 (找故障)
# ==========================================
# 残差 = 理论值 (AI预测) - 实际值
# 如果残差是正数，说明：AI觉得应该发很多电，但实际发的很少 ——> 可能是故障/遮挡
data['Residual'] = data['Pred_Power'] - data['AC_POWER']

# 设定阈值：相差多少算异常？
# 这里我们设定：如果少发了 500 kW 以上，就算异常
threshold = 500 
data['Anomalies'] = data['Residual'] > threshold

# 统计一下发现了多少个异常点
num_anomalies = data['Anomalies'].sum()
print(f"分析完成！共发现 {num_anomalies} 个异常时刻。")

# ==========================================
# 5. 画图 (这是给老师看的重点)
# ==========================================
# 为了看清楚，我们只画前 4 天的数据
# 你可以修改 days=4 来看更多天
plot_data = data[data['DATE_TIME'] < (data['DATE_TIME'].min() + pd.Timedelta(days=4))]

plt.figure(figsize=(15, 6))

# 画蓝线：AI 算出来的完美曲线
plt.plot(plot_data['DATE_TIME'], plot_data['Pred_Power'], 
         label='AI Predicted (Ideal)', color='blue', alpha=0.6, linewidth=2)

# 画橙线：实际的发电曲线
plt.plot(plot_data['DATE_TIME'], plot_data['AC_POWER'], 
         label='Actual Power', color='orange', alpha=0.6, linewidth=2)

# 画红点：异常点！
anomalies = plot_data[plot_data['Anomalies']]
plt.scatter(anomalies['DATE_TIME'], anomalies['AC_POWER'], 
            color='red', label='Anomaly Detected', zorder=5, s=50, edgecolors='black')

plt.title(f'Solar Fault Detection (Random Forest) - Inverter: {target_inverter}')
plt.xlabel('Time')
plt.ylabel('AC Power (kW)')
plt.legend()
plt.grid(True, alpha=0.3)

# 保存图片到本地
plt.savefig('rf_anomaly_result.png')
print("图片已保存为 'rf_anomaly_result.png'，快去打开看看！")

plt.show()