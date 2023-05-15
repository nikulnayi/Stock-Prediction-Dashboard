import streamlit as st
import pandas as pd
import pickle
from prophet import Prophet
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from GetData import get_data

# Load the Prophet models from pickle files
with open('prophet_model_high.pkl', 'rb') as file:
    prophet_model_high = pickle.load(file)
with open('prophet_model_low.pkl', 'rb') as file:
    prophet_model_low = pickle.load(file)
with open('prophet_model_close.pkl', 'rb') as file:
    prophet_model_close = pickle.load(file)

# Load historical data
historical_data = get_data()
historical_data['Date'] = pd.to_datetime(historical_data['Date'])  # Convert 'Date' column to DateTime type

# Get the last date in the historical data
last_date = historical_data['Date'].max()

# Predict the next weekday
next_weekday = last_date + timedelta(days=1)
while next_weekday.weekday() >= 5:  # Skip weekends
    next_weekday += timedelta(days=1)

# Prepare the data for forecasting
future_data = pd.DataFrame({'ds': [next_weekday]})

# Perform the forecast using the Prophet models
prophet_forecast_high = prophet_model_high.predict(future_data)
prophet_forecast_low = prophet_model_low.predict(future_data)
prophet_forecast_close = prophet_model_close.predict(future_data)

# Get the forecasted values
prophet_value_high = prophet_forecast_high['yhat'].values[0]
prophet_value_low = prophet_forecast_low['yhat'].values[0]
prophet_value_close = prophet_forecast_close['yhat'].values[0]

# Reverse min-max scaling on the forecasted values
scaler = MinMaxScaler()  # Use the same scaler as used during preprocessing
historical_scaled = scaler.fit_transform(historical_data[['High', 'Low', 'Close']])
forecast_scaled = scaler.transform([[prophet_value_high, prophet_value_low, prophet_value_close]])
prophet_value_high, prophet_value_low, prophet_value_close = forecast_scaled[0]

# Set up the Streamlit app
st.title("Stock Price Prediction")
st.subheader("Next Weekday Prediction")

# Display the forecasted values
st.write("Date:", next_weekday.date())
st.write("High:", prophet_value_high)
st.write("Low:", prophet_value_low)
st.write("Close:", prophet_value_close)

# Combine historical data and forecasted data
forecast_data = pd.DataFrame({
    'Date': historical_data['Date'].tolist() + [next_weekday],
    'High': historical_data['High'].tolist() + [prophet_value_high],
    'Low': historical_data['Low'].tolist() + [prophet_value_low],
    'Close': historical_data['Close'].tolist() + [prophet_value_close]
})

# Display line graph of historical data and forecasted data
last_10_days_data = forecast_data.tail(10)
st.subheader("Historical Data and Forecast (Last 10 Days)")
st.line_chart(last_10_days_data.set_index('Date')[['High', 'Low', 'Close']])

# Additional cards
st.subheader("Additional Information")
st.write("Here you can add additional information or insights.")
