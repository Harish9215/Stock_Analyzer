import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Function to get stock data from Yahoo Finance
def get_stock_data(ticker, period="6mo", interval="1d"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period, interval=interval)

# Streamlit UI
st.title("📊 Real-Time Stock Market Analyzer with XGBoost")

# User input for stock ticker
ticker = st.text_input("Enter a Stock Symbol (e.g., AAPL, TSLA):", "AAPL")

if ticker:
    # Fetch stock data
    data = get_stock_data(ticker)

    if data.empty:
        st.error("⚠️ No data found! Please enter a valid stock symbol.")
    else:
        # Show stock data preview
        st.write("### Latest Stock Prices")
        st.dataframe(data.tail())

        # Plot stock price trend
        st.write("### Stock Price Over Time")
        fig, ax = plt.subplots()
        ax.plot(data.index, data["Close"], label="Closing Price", color="blue")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

        # Prepare data for Machine Learning
        data["Tomorrow"] = data["Close"].shift(-1)  # Next day's price
        data = data.dropna()  # Remove empty values

        # Features (input) and Target (output)
        X = data[["Open", "High", "Low", "Volume"]]
        y = data["Tomorrow"]

        # Split data into train & test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train XGBoost Model
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective="reg:squarederror")
        model.fit(X_train, y_train)

        # Predict next day's closing price
        predicted_price = model.predict(X.iloc[-1:].values)[0]
        st.write(f"📌 **Predicted Next Day Closing Price:** ${predicted_price:.2f}")

        # Buy/Sell/Hold Decision
        last_price = data["Close"].iloc[-1]

        if predicted_price > last_price * 1.02:  # If prediction is 2% higher
            decision = "BUY ✅"
        elif predicted_price < last_price * 0.98:  # If prediction is 2% lower
            decision = "SELL ❌"
        else:
            decision = "HOLD ⚠️"

        st.write(f"📊 **Suggested Action:** {decision}")