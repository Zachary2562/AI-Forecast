import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PIL import Image  # Pillow replacement for imghdr

# App title and logo
st.set_page_config(page_title="AI Forecasting Tool", layout="wide")
st.title("📈 AI Stock & Fund Forecasting App")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter a stock/fund/ETF ticker (e.g. AAPL, FSELX):", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Download and display data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

if ticker:
    df = load_data(ticker, start_date, end_date)
    st.subheader(f"Raw Data for {ticker}")
    st.line_chart(df["Close"])

    # Prophet Forecast
    st.subheader("📅 Prophet Forecast")
    df_prophet = df.reset_index()[["Date", "Close"]]
    df_prophet.columns = ["ds", "y"]
    m = Prophet()
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    # LSTM Forecast
    st.subheader("🔁 LSTM Forecast")

    data = df.filter(["Close"])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    valid = data[training_data_len:]
    valid["Predictions"] = predictions

    st.line_chart(valid[["Close", "Predictions"]])
