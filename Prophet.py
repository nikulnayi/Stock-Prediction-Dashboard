import pandas as pd
from prophet import Prophet
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from GetData import get_data

def train_prophet_model(data):
    # Create a new Prophet model
    model = Prophet()
    
    # Fit the model to the data
    model.fit(data)
    
    return model

def evaluate_prophet_model(model, data, target):
    # Make predictions using the Prophet model
    forecast = model.predict(data)
    y_true = data['y'].values
    y_pred = forecast['yhat'].values
    
    # Compute mean squared error (MSE) and mean absolute error (MAE)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Print the evaluation metrics
    print(f"Target variable: {target}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return mse, mae

def main():
    # Get the stock data and preprocess it
    df = get_data()
    processed_df = df[['Date', 'High', 'Low', 'Close']]
    
    # Prepare the data for Prophet
    data = pd.DataFrame({
        'ds': processed_df['Date'],
        'y_high': processed_df['High'],
        'y_low': processed_df['Low'],
        'y_close': processed_df['Close']
    })
    
    # Train the Prophet model for each target variable
    prophet_models = {}
    for target in ['high', 'low', 'close']:
        target_data = data[['ds', f'y_{target}']]
        target_data = target_data.rename(columns={f'y_{target}': 'y'})
        prophet_models[target] = train_prophet_model(target_data)
    
        # Evaluate the Prophet model using MSE and MAE metrics
        mse, mae = evaluate_prophet_model(prophet_models[target], target_data, target)
    
    # Save the trained Prophet models to pickle files
    for target, model in prophet_models.items():
        filename = f'prophet_model_{target}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Saved Prophet model for {target} as {filename}")

if __name__ == '__main__':
    main()
