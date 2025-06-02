import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import traceback

# App config and intro
st.set_page_config(page_title="🧠 AI Forecasting Tool", layout="wide")
st.title("📊 AI Forecasting Tool")
st.markdown("✅ App started successfully — awaiting user input.")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter a stock/fund/ETF ticker (e.g. AAPL, GOOG):", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

try:
    if ticker:
        df = load_data(ticker, start_date, end_date)

        if df.empty:
            st.error("❌ No data returned. Please check the ticker symbol and date range.")
            st.stop()

        # Show data chart
        st.subheader(f"📉 Closing Prices for {ticker}")
        st.line_chart(df["Close"])

        # Prophet forecast
        st.subheader("📅 Prophet Forecast (1 Year Ahead)")
        df_prophet = df.reset_index()[["Date", "Close"]]
        df_prophet.columns = ["ds", "y"]

        m = Prophet()
        m.fit(df_prophet)

        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)

        fig1 = m.plot(forecast)
        st.pyplot(fig1)

except Exception as e:
    st.error("❌ An error occurred during forecasting:")
    st.text(traceback.format_exc())