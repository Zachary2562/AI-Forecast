import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AI Stock Forecast", layout="wide")
st.title("📈 AI-Powered Stock Price Forecast")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOGL, BTC-USD, GC=F):", "AAPL")
start_date = "2010-01-01"
end_date = "2025-05-31"

@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start=start_date, end=end_date)

def plot_forecast(df, forecast, model_name):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['Close'], label="Actual Price", color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label=f"{model_name} Forecast", color='green')
    ax.set_title(f"{model_name} Forecast vs Actual Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

data_load_state = st.text("Loading data...")
df = load_data(ticker)
data_load_state.text("Loading data... done!")

df = df.reset_index()
df = df[['Date', 'Close']]

# Facebook Prophet Forecast
df_prophet = df.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

st.subheader("Prophet Forecast")
plot_forecast(df, forecast, "Prophet")

# LSTM Model
st.subheader("LSTM Forecast")
df_lstm = df.copy()
scaler = MinMaxScaler(feature_range=(0, 1))
df_lstm['Close'] = scaler.fit_transform(df_lstm[['Close']])

lookback = 60
X, y = [], []
for i in range(lookback, len(df_lstm)):
    X.append(df_lstm['Close'].values[i-lookback:i])
    y.append(df_lstm['Close'].values[i])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=5, batch_size=64, verbose=0)

last_60 = df_lstm['Close'].values[-60:]
pred_input = last_60.reshape(1, lookback, 1)
lstm_pred_scaled = model.predict(pred_input)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

future_date = pd.to_datetime(df['Date'].iloc[-1]) + pd.Timedelta(days=1)
st.write(f"📊 LSTM predicts the price for {future_date.date()} to be: **${lstm_pred[0][0]:.2f}**")
