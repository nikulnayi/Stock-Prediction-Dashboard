from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    scaler = MinMaxScaler()
    columns_to_scale = [col for col in data.columns if col != 'Date']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data