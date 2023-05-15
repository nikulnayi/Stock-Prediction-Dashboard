import requests
import pandas as pd

def extract_stock_data(url):
    r = requests.get(url)
    data = r.json()
    time_series_data = data['Time Series (Daily)']
    df = []
    for date, values in time_series_data.items():
        if date >= '2022-01-01':
            row = {
                'Date': date,
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close'])
            }
            df.append(row)
    return pd.DataFrame(df)

def extract_forex_data(url):
    r = requests.get(url)
    data = r.json()
    time_series_data = data['Time Series FX (Daily)']
    df = []
    for date, values in time_series_data.items():
        row = {
            'Date': date,
            'Dollar': float(values['4. close'])
        }
        df.append(row)
    return pd.DataFrame(df)

def get_data():
    stock_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=TATAMOTORS.BSE&outputsize=full&apikey=4B0125QNRR7Q3KL4'
    forex_url = 'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=INR&outputsize=full&apikey=4B0125QNRR7Q3KL4'

    stock_df = extract_stock_data(stock_url)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # forex_df = extract_forex_data(forex_url)
    # forex_df['Date'] = pd.to_datetime(forex_df['Date'])

    # merged_df = pd.merge(stock_df, forex_df, on='Date', how='inner')

    return stock_df
