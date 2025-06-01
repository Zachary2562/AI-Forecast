import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.title("AI Stock Forecast App")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL")

# Download historical data
df = yf.download(ticker, start="2015-01-01", end="2025-01-01")
df.reset_index(inplace=True)

# Plot historical data
st.subheader("Historical Closing Prices")
st.line_chart(df[['Date', 'Close']].set_index('Date'))

# Prophet forecast
st.subheader("Forecast with Prophet")
prophet_df = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
fig1 = model.plot(forecast)
st.pyplot(fig1)