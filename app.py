
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Forecast", layout="centered")

st.sidebar.title("🔍 AI Stock Forecast Tool")
ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g. AAPL, TSLA, FBGRX):", "AAPL")
forecast_years = st.sidebar.slider("Years to Forecast:", 1, 5, 2)

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2010-01-01", end="2025-05-31", group_by='ticker')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    df.reset_index(inplace=True)
    return df

df = load_data(ticker)
st.write(f"### Showing data for: {ticker}")
st.line_chart(df[['Date', 'Close']].set_index('Date'))

def add_indicators(df):
    df['RSI'] = df['Close'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).sum() / abs(x.diff().clip(upper=0)).sum()))), raw=False)
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Bollinger_Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['Bollinger_Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    return df

df = add_indicators(df)

def run_prophet(df, years):
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=years * 365)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

forecast_df = run_prophet(df, forecast_years)

def run_lstm(df, years):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    forecast_days = years * 365
    lstm_input = scaled[-60:]
    lstm_forecast = []
    for _ in range(forecast_days):
        pred = model.predict(lstm_input.reshape(1, 60, 1), verbose=0)[0][0]
        lstm_forecast.append(pred)
        lstm_input = np.append(lstm_input[1:], pred)

    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_days+1)[1:]
    forecast_scaled = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1))
    lstm_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_scaled.flatten()})
    return lstm_df

lstm_df = run_lstm(df, forecast_years)

st.subheader("📈 Forecasted Price (Prophet)")
st.line_chart(forecast_df.set_index('ds')[['yhat']])

st.subheader("📉 Forecasted Price (LSTM)")
st.line_chart(lstm_df.set_index('Date'))

st.subheader("📊 Technical Indicators")
st.line_chart(df.set_index('Date')[['RSI', 'MACD', 'Signal']].dropna())

st.subheader("📎 Bollinger Bands")
st.line_chart(df.set_index('Date')[['Close', 'Bollinger_Upper', 'Bollinger_Lower']].dropna())
