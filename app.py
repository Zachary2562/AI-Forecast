import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
import datetime

st.set_page_config(layout="wide")

st.title("📈 AI-Powered Stock & Fund Price Forecasting App")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Ticker (e.g., AAPL, FSELX, BTC-USD):", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

if start_date >= end_date:
    st.sidebar.error("End date must be after start date.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError("No data found for this ticker.")
    data["Date"] = data.index
    return data

try:
    df = load_data(ticker, start_date, end_date)
    st.success(f"Loaded {len(df)} rows of data for {ticker}")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Plot historical prices
st.subheader("📊 Historical Prices")
st.line_chart(df["Close"])

# Prophet Forecasting
st.subheader("🔮 Prophet Forecast")
prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

fig1 = model.plot(forecast)
st.pyplot(fig1)

# LSTM Forecasting
st.subheader("🧠 LSTM Forecast")

# Preprocess for LSTM
data_lstm = df[["Close"]].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_lstm)

train_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_len]
test_data = scaled_data[train_len - 60:]

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Forecast
X_test = []
for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predictions = model_lstm.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot
lstm_dates = df.index[train_len:]
plt.figure(figsize=(14, 5))
plt.plot(lstm_dates, df["Close"].iloc[train_len:], label="Actual")
plt.plot(lstm_dates, predictions, label="LSTM Forecast")
plt.title("LSTM Model Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(plt)
