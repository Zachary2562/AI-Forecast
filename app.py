# Force Python to use local imghdr patch to prevent Streamlit crash
import sys
sys.modules['imghdr'] = __import__('imghdr')

# Core Imports
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

# Streamlit App Title
st.title("📈 AI Stock Forecasting App")

# User Inputs
ticker = st.text_input("Enter stock or ETF ticker (e.g., AAPL, MSFT, SPY):", value="AAPL")
forecast_years = st.slider("Forecast how many years into the future?", 1, 5, 2)

# Download Historical Data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2010-01-01")
    df.reset_index(inplace=True)
    return df

df = load_data(ticker)

# Show historical chart
st.subheader("📊 Historical Closing Prices")
st.line_chart(df[['Date', 'Close']].set_index('Date'))

# Add Technical Indicators
def add_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

df = add_indicators(df)

# Prophet Forecasting
st.subheader("🔮 Prophet Forecast")
prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
m = Prophet()
m.fit(prophet_df)

future = m.make_future_dataframe(periods=forecast_years * 365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
st.pyplot(fig1)

# Show technical indicators
st.subheader("📉 Technical Indicators")
st.line_chart(df[['Date', 'SMA_50', 'SMA_200']].set_index('Date'))
st.line_chart(df[['Date', 'MACD', 'Signal_Line']].set_index('Date'))

# Footer
st.caption("Made with ❤️ using Streamlit, Prophet, and yfinance")
