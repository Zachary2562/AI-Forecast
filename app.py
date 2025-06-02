import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import traceback
import time

# Page setup
st.set_page_config(page_title="AI Forecast", layout="wide")
st.title("📊 AI Stock & Fund Forecasting App")
st.markdown("✅ App loaded. Waiting for your input.")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter a stock/fund/ETF ticker (e.g. AAPL, GOOG):", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

try:
    if ticker:
        st.write(f"📥 Fetching data for `{ticker}`...")
        df = load_data(ticker, start_date, end_date)

        if df.empty:
            st.warning("⚠️ No data found. Try a different ticker.")
            st.stop()

        st.line_chart(df["Close"])
        st.write("🔮 Fitting Prophet model...")

        df_prophet = df.reset_index()[["Date", "Close"]]
        df_prophet.columns = ["ds", "y"]

        m = Prophet()
        start = time.time()
        m.fit(df_prophet)
        elapsed = round(time.time() - start, 2)
        st.success(f"✅ Prophet model trained in {elapsed} seconds")

        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)

        st.subheader("📅 Forecast")
        fig = m.plot(forecast)
        st.pyplot(fig)

except Exception as e:
    st.error("❌ An error occurred:")
    st.text(traceback.format_exc())
