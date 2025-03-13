# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 📌 Load dataset
# file_path = "Air_Traffic_Passenger_Statistics.csv"  # Update with your file path

# # Read CSV file and inspect columns
# df = pd.read_csv(file_path)
# print("✅ Dataset Loaded Successfully!\n")
# print("📌 Columns in Dataset:", df.columns)

# # 🔹 Convert 'Activity Period' to datetime (assuming format YYYYMM)
# if 'Activity Period' in df.columns:
#     df['Activity Period'] = pd.to_datetime(df['Activity Period'], format='%Y%m')
#     df.set_index('Activity Period', inplace=True)  # Set as index
#     print("✅ 'Activity Period' converted to datetime.")
# else:
#     print("❌ Error: 'Activity Period' column not found!")

# # 🔹 Handle missing values
# df.fillna(0, inplace=True)

# # 🔹 Aggregate passenger count over time
# passenger_trend = df.groupby(df.index)['Passenger Count'].sum()

# # 📊 Plot Passenger Trend
# plt.figure(figsize=(12, 6))
# sns.lineplot(x=passenger_trend.index, y=passenger_trend.values, marker='o', color='b')
# plt.title("Airline Passenger Trend Over Time", fontsize=14)
# plt.xlabel("Year", fontsize=12)
# plt.ylabel("Total Passengers", fontsize=12)
# plt.grid(True)
# plt.show()

# # 🔹 Save preprocessed data for Power BI / Tableau
# df.to_csv("processed_airline_passengers.csv", index=True)
# print("✅ Preprocessed data saved as 'processed_airline_passengers.csv'.")











import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "processed_airline_passengers.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# 🔹 Automatically detect and convert dates
df['Activity Period'] = pd.to_datetime(df['Activity Period'], errors='coerce')

# Check for missing dates
if df['Activity Period'].isna().sum() > 0:
    print("Warning: Some dates could not be parsed correctly!")

# Set index to Activity Period
df.set_index('Activity Period', inplace=True)
print("'Activity Period' converted successfully!")

# 🔹 Aggregate passenger count over time
passenger_trend = df.groupby(df.index)['Passenger Count'].sum()

# Line Chart: Passenger Trend Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x=passenger_trend.index, y=passenger_trend.values, marker='o', color='b')
plt.title("Airline Passenger Trend Over Time", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Total Passengers", fontsize=12)
plt.grid(True)
plt.show()

# Bar Chart: Passenger Count by Airline
if 'Airline' in df.columns:
    airline_passenger_count = df.groupby('Airline')['Passenger Count'].sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=airline_passenger_count.index, y=airline_passenger_count.values, palette='viridis')
    plt.title("📊 Total Passenger Count by Airline", fontsize=14)
    plt.xlabel("Airline", fontsize=12)
    plt.ylabel("Total Passengers", fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

# Heatmap: Passenger Density by Region (Fixed)
if 'Region' in df.columns:
    df['Year'] = df.index.year  # Extract year
    region_monthly = df.pivot_table(index='Year', columns='Region', values='Passenger Count', aggfunc='sum')

    plt.figure(figsize=(12, 6))
    sns.heatmap(region_monthly.fillna(0), cmap="coolwarm", linewidths=0.5, annot=True, fmt=".0f")
    plt.title(" Passenger Density by Region (Yearly)", fontsize=14)
    plt.xlabel("Region", fontsize=12)
    plt.ylabel("Year", fontsize=12)
    plt.show()

# Find optimal SARIMA parameters using Auto-ARIMA
print(" Finding optimal SARIMA parameters...")
auto_model = auto_arima(passenger_trend, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
p, d, q = auto_model.order
P, D, Q, s = auto_model.seasonal_order
print(f"Best SARIMA Order: ({p}, {d}, {q}) x ({P}, {D}, {Q}, {s})")

# Train the SARIMA model
print("Training SARIMA model...")
sarima_model = SARIMAX(passenger_trend, order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_result = sarima_model.fit(disp=False)

#  Forecast next 12 months
forecast_steps = 12
future_dates = pd.date_range(start=passenger_trend.index[-1], periods=forecast_steps + 1, freq='M')[1:]
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Accuracy Metrics
train_size = int(len(passenger_trend) * 0.8)  # 80% training, 20% test
train, test = passenger_trend.iloc[:train_size], passenger_trend.iloc[train_size:]

# Fit model on training data
sarima_train = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=False)
predictions = sarima_train.get_forecast(steps=len(test)).predicted_mean

# Calculate errors
mae = mean_absolute_error(test, predictions)
rmse = mean_squared_error(test, predictions, squared=False)
mape = np.mean(np.abs((test - predictions) / test)) * 100

print(f" Model Accuracy Metrics:\n- MAE: {mae:.2f}\n- RMSE: {rmse:.2f}\n- MAPE: {mape:.2f}%")

# 🔹 Line Chart: Forecasted Passenger Count
plt.figure(figsize=(12, 6))
plt.plot(passenger_trend.index, passenger_trend.values, label="Actual Passenger Count", color='blue')
plt.plot(future_dates, forecast.predicted_mean, label="Forecasted Passenger Count", color='red', linestyle='dashed')
plt.fill_between(future_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title("SARIMA Forecast for Airline Passengers", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Passenger Count", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 🔹 Save forecast results
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Passenger Count": forecast.predicted_mean
})
forecast_df.to_csv("sarima_forecast.csv", index=False)
print(" SARIMA Forecast saved as 'sarima_forecast.csv'.")
