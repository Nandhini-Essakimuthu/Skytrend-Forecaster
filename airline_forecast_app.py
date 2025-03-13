import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "processed_airline_passengers.csv"  # Update with actual file path
df = pd.read_csv(file_path)

# Convert date column and set index
df['Activity Period'] = pd.to_datetime(df['Activity Period'], errors='coerce')
df.set_index('Activity Period', inplace=True)

# Streamlit App
st.title("✈️ Airline Passenger Data Analysis & Forecasting")

# Sidebar Filters
st.sidebar.header("🔍 Filters & Settings")

# Year range filter
min_year, max_year = df.index.min().year, df.index.max().year
selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
filtered_df = df[(df.index.year >= selected_years[0]) & (df.index.year <= selected_years[1])]

# Airline selection
if 'Airline' in df.columns:
    selected_airlines = st.sidebar.multiselect("Select Airlines", df['Airline'].unique(), default=df['Airline'].unique())
    filtered_df = filtered_df[filtered_df['Airline'].isin(selected_airlines)]

# Q&A Section
st.subheader("💬 Ask a Question About Airline Passengers")
user_question = st.text_input("Type your question and press Enter...")

def answer_question(question, df):
    """Basic Q&A system using Pandas."""
    question = question.lower()

    if "most passengers" in question and "year" in question:
        year = ''.join(filter(str.isdigit, question))
        if year:
            year = int(year)
            year_data = df[df.index.year == year]
            if not year_data.empty and 'Airline' in df.columns:
                top_airline = year_data.groupby('Airline')['Passenger Count'].sum().idxmax()
                return f"In {year}, **{top_airline}** had the most passengers."
            else:
                return f"No data available for {year}."

    elif "total passengers" in question and "airline" in question:
        if 'Airline' in df.columns:
            airline_counts = df.groupby('Airline')['Passenger Count'].sum().sort_values(ascending=False)
            return airline_counts.to_string()

    elif "trend" in question or "increase" in question:
        trend = df['Passenger Count'].resample('Y').sum()
        trend_growth = trend.pct_change().dropna()
        if not trend_growth.empty and trend_growth.iloc[-1] > 0:
            return f"Passenger count has **increased** in the last year by **{trend_growth.iloc[-1]*100:.2f}%**."
        elif not trend_growth.empty:
            return f"Passenger count has **decreased** in the last year by **{abs(trend_growth.iloc[-1]*100):.2f}%**."
        else:
            return "Not enough data to determine a trend."

    else:
        return "❌ Sorry, I can't answer that yet. Try asking about trends, top airlines, or passenger counts."

if user_question:
    response = answer_question(user_question, df)
    st.write(response)

# Passenger Trend Over Time
st.subheader("📊 Airline Passenger Trend Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=filtered_df.index, y=filtered_df['Passenger Count'], marker='o', color='b', ax=ax)
plt.xlabel("Year")
plt.ylabel("Total Passengers")
st.pyplot(fig)

# Forecasting Section
st.subheader("🔮 Forecasting Airline Passengers")
forecast_steps = st.sidebar.slider("Select Forecast Duration (Months)", 6, 24, 12)

if st.button("Train SARIMA Model & Forecast"):
    with st.spinner("Training SARIMA model... ⏳"):
        auto_model = auto_arima(df['Passenger Count'], seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
        p, d, q = auto_model.order
        P, D, Q, s = auto_model.seasonal_order
        st.write(f"Best SARIMA Order: ({p}, {d}, {q}) x ({P}, {D}, {Q}, {s})")

        sarima_model = SARIMAX(df['Passenger Count'], order=(p, d, q), seasonal_order=(P, D, Q, s))
        sarima_result = sarima_model.fit(disp=False)

        future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='M')[1:]
        forecast = sarima_result.get_forecast(steps=forecast_steps)

        # Display Forecast
        st.subheader("🔮 SARIMA Forecast for Airline Passengers")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Passenger Count'], label="Actual Passenger Count", color='blue')
        ax.plot(future_dates, forecast.predicted_mean, label="Forecasted Passenger Count", color='red', linestyle='dashed')
        plt.xlabel("Year")
        plt.ylabel("Passenger Count")
        plt.legend()
        st.pyplot(fig)



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Streamlit App Title
st.title("📊 Airline Passenger Analysis & Forecasting")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert 'Activity Period' to datetime
    df['Activity Period'] = pd.to_datetime(df['Activity Period'], errors='coerce')
    df.set_index('Activity Period', inplace=True)
    
    # Show first few rows
    st.subheader("📂 Data Overview")
    st.write(df.head())

    # Aggregate passenger data
    passenger_trend = df.groupby(df.index)['Passenger Count'].sum()

    # Line Chart: Passenger Trend Over Time
    st.subheader("📈 Passenger Trend Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=passenger_trend.index, y=passenger_trend.values, ax=ax, marker='o', color='b')
    ax.set_title("Airline Passenger Trend Over Time", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Passengers")
    ax.grid(True)
    st.pyplot(fig)

    # Bar Chart: Passenger Count by Airline
    if 'Airline' in df.columns:
        st.subheader("📊 Total Passenger Count by Airline")
        airline_passenger_count = df.groupby('Airline')['Passenger Count'].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=airline_passenger_count.index, y=airline_passenger_count.values, ax=ax, palette='viridis')
        ax.set_title("Total Passenger Count by Airline")
        ax.set_xlabel("Airline")
        ax.set_ylabel("Total Passengers")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Heatmap: Passenger Density by Region
    if 'Region' in df.columns:
        st.subheader("🌍 Passenger Density by Region (Yearly)")
        df['Year'] = df.index.year  # Extract year
        region_monthly = df.pivot_table(index='Year', columns='Region', values='Passenger Count', aggfunc='sum')

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(region_monthly.fillna(0), cmap="coolwarm", linewidths=0.5, annot=True, fmt=".0f", ax=ax)
        ax.set_title("Passenger Density by Region (Yearly)")
        ax.set_xlabel("Region")
        ax.set_ylabel("Year")
        st.pyplot(fig)

    # SARIMA Forecasting
    st.subheader("📈 SARIMA Time Series Forecasting")
    with st.spinner("Finding optimal SARIMA parameters..."):
        auto_model = auto_arima(passenger_trend, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
        p, d, q = auto_model.order
        P, D, Q, s = auto_model.seasonal_order
        st.write(f"**Best SARIMA Order:** ({p}, {d}, {q}) x ({P}, {D}, {Q}, {s})")

    with st.spinner("Training SARIMA model..."):
        sarima_model = SARIMAX(passenger_trend, order=(p, d, q), seasonal_order=(P, D, Q, s))
        sarima_result = sarima_model.fit(disp=False)

    # Forecast next 12 months
    forecast_steps = 12
    future_dates = pd.date_range(start=passenger_trend.index[-1], periods=forecast_steps + 1, freq='M')[1:]
    forecast = sarima_result.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()

    # Display Forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(passenger_trend.index, passenger_trend.values, label="Actual Passenger Count", color='blue')
    ax.plot(future_dates, forecast.predicted_mean, label="Forecasted Passenger Count", color='red', linestyle='dashed')
    ax.fill_between(future_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    ax.set_title("SARIMA Forecast for Airline Passengers", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Passenger Count")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Model Accuracy Metrics
    train_size = int(len(passenger_trend) * 0.8)  # 80% training, 20% test
    train, test = passenger_trend.iloc[:train_size], passenger_trend.iloc[train_size:]

    # Fit model on training data
    sarima_train = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=False)
    predictions = sarima_train.get_forecast(steps=len(test)).predicted_mean

    # Calculate errors
    mae = mean_absolute_error(test, predictions)
    rmse = mean_squared_error(test, predictions, squared=False)
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    st.subheader("📊 Model Accuracy Metrics")
    st.write(f"🔹 **Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"🔹 **Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"🔹 **Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")

    # Save forecast results
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Passenger Count": forecast.predicted_mean
    })
    forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Forecast Data",
        data=forecast_csv,
        file_name="sarima_forecast.csv",
        mime="text/csv",
    )
