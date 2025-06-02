import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import traceback
import time

st.set_page_config(page_title="AI Forecast", layout="wide")
st.title("📊 AI Stock & Fund Forecasting App")
st.markdown("✅ App loaded. Waiting for your input.")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter a stock/fund/ETF ticker (e.g. AAPL, GOOG):", value="AAPL")
start_date = pd.to_datetime("2010-01-01")
end_date = pd.to_datetime("today")

@st.cache_data
def load_data(ticker, start, end):
    st.write(f"📥 Downloading {ticker} data...")
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

try:
    if ticker:
        df = load_data(ticker, start_date, end_date)

        if df.empty:
            st.warning("⚠️ No data found. Try a different ticker.")
            st.stop()

        st.subheader("📈 Historical Data")
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