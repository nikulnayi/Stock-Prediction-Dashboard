import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from prophet import Prophet
import pickle
from GetData import get_data
from Preprocessing import preprocess_data

def train_test_split(data, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data


def evaluate(y_true, y_pred):
    mse = np.mean(np.square(y_true - y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae


def train_var_model(train_data):
    model = VAR(train_data)
    result = model.fit(maxlags=4, ic='aic')
    return result


def train_lstm_model(train_data):
    X_train = np.array(train_data.drop(columns=['High', 'Low', 'Close']))
    y_train = np.array(train_data[['High', 'Low', 'Close']])
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    return model


def train_gru_model(train_data):
    X_train = np.array(train_data.drop(columns=['High', 'Low', 'Close']))
    y_train = np.array(train_data[['High', 'Low', 'Close']])
    model = Sequential()
    model.add(GRU(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    return model


def train_prophet_model(train_data):
    train_data = train_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(train_data)
    return model


def main():

    df = get_data()
    processed_df = preprocess_data(df)
    train_data, test_data = train_test_split(processed_df)

    # train VAR model
    var_model = train_var_model(train_data)

    # # train LSTM model
    lstm_model = train_lstm_model(train_data)

    # # train GRU model
    gru_model = train_gru_model(train_data)

    # train Prophet model
    prophet_model = train_prophet_model(train_data)

    # evaluate models on test data
    X_test = np.array(test_data.drop(columns=['High', 'Low', 'Close']))
    y_test = np.array(test_data[['High', 'Low', 'Close']])

    var_pred = var_model.forecast(X_test, len(test_data))
    lstm_pred = lstm_model.predict(X_test)
    gru_pred = gru_model.predict(X_test)
    prophet_pred = prophet_model.predict(test_data[['Date']])

    var_mse, var_mae = evaluate(y_test, var_pred)
    lstm_mse, lstm_mae = evaluate(y_test, lstm_pred)
    gru_mse, gru_mae = evaluate(y_test, gru_pred)
    prophet_mse, prophet_mae = evaluate(y_test[:, -1], prophet_pred['yhat'])
    print(prophet_mse,prophet_mae)

if __name__ == '__main__':
    main()