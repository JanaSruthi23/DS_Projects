import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD DATA 
# -----------------------------
file_path = r"C:\Users\SRUTHI\food_delivery_eta_system\data\food_delivery_eta_5000.csv"

df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe())

# -----------------------------
# CONVERT TIME + TARGET
# -----------------------------
df['order_time'] = pd.to_datetime(df['order_time'])
df['delivery_time'] = pd.to_datetime(df['delivery_time'])

df['delivery_duration'] = (
    df['delivery_time'] - df['order_time']
).dt.total_seconds() / 60

# -----------------------------
# MISSING VALUES
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# DISTRIBUTION
# -----------------------------
plt.figure()
plt.hist(df['delivery_duration'], bins=30)
plt.title("Delivery Time Distribution")
plt.xlabel("Minutes")
plt.ylabel("Frequency")
plt.savefig("outputs/delivery_distribution.png")
plt.close()

# -----------------------------
# DISTANCE VS TIME
# -----------------------------
plt.figure()
plt.scatter(df['distance_km'], df['delivery_duration'])
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (min)")
plt.title("Distance vs Delivery Time")
plt.savefig("outputs/distance_vs_delivery.png")
plt.close()

# -----------------------------
# TRAFFIC IMPACT
# -----------------------------
print("\nTraffic Impact:")
print(df.groupby('traffic_level')['delivery_duration'].mean())

df.groupby('traffic_level')['delivery_duration'].mean().plot(kind='bar')
plt.title("Traffic vs Delivery Time")
plt.savefig("outputs/traffic_impact.png")
plt.close()

# -----------------------------
# WEATHER IMPACT
# -----------------------------
print("\nWeather Impact:")
print(df.groupby('weather')['delivery_duration'].mean())

df.groupby('weather')['delivery_duration'].mean().plot(kind='bar')
plt.title("Weather Impact")
plt.savefig("outputs/weather_impact.png")
plt.close()

# -----------------------------
# HOUR ANALYSIS
# -----------------------------
df['hour'] = df['order_time'].dt.hour

df.groupby('hour')['delivery_duration'].mean().plot()
plt.title("Delivery Time by Hour")
plt.xlabel("Hour")
plt.ylabel("Avg Delivery Time")
plt.savefig("outputs/avg_delivery_time.png")
plt.close()

# -----------------------------
# CITY ANALYSIS
# -----------------------------
df.groupby('city')['delivery_duration'].mean().plot(kind='bar')
plt.title("City-wise Delivery Time")
plt.savefig("outputs/citywise_delivery.png")
plt.close()

# -----------------------------
# CORRELATION
# -----------------------------
df_encoded = df.copy()

df_encoded['traffic_level'] = df_encoded['traffic_level'].map({'Low':1,'Medium':2,'High':3})
df_encoded['weather'] = df_encoded['weather'].map({'Clear':0,'Rainy':1,'Stormy':2})
numeric_df = df_encoded.select_dtypes(include=['number'])

plt.figure()
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("outputs/correlation.png")
plt.close()

# -----------------------------
# TRAFFIC INCREASE PERCENT
# -----------------------------

traffic_avg = df.groupby('traffic_level')['delivery_duration'].mean()
print(traffic_avg)
low = traffic_avg['Low']
high = traffic_avg['High']

increase_pct = ((high - low) / low) * 100
print(increase_pct)