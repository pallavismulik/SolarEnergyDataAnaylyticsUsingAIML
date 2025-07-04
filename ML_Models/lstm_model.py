# lstm_model.py 
# Time-series forecasting of solar energy based on historical solar irradiance data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, target_column, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_column])
    return np.array(X), np.array(y)

def main():
    # Load Cleaned Data
    df = pd.read_csv("cleaned_data.csv")

    #  Sort by time if not already sorted (ensure time series integrity)
    df = df.sort_values(by=['month', 'day', 'hour', 'minute']).reset_index(drop=True)

    #  Select Features (can add more if you want)
    features = ['temperature', 'dew_point', 'pressure', 'relative_humidity', 'wind_speed', 'ghi']
    df = df[features]

    # Normalize all features between 0-1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create sequences of 24 hours → predict next hour's GHI
    sequence_length = 24
    target_column = features.index('ghi')  # Index of GHI in scaled_data

    X, y = create_sequences(scaled_data, target_column, sequence_length)

    # Train/Test Split (80/20)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    #  Build LSTM Model
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    #  Train Model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    #  Make Predictions
    y_pred = model.predict(X_test)

    #  Inverse scale the predictions and actuals
    ghi_index = features.index('ghi')
    y_test_full = np.zeros((len(y_test), len(features)))
    y_pred_full = np.zeros((len(y_pred), len(features)))

    y_test_full[:, ghi_index] = y_test
    y_pred_full[:, ghi_index] = y_pred[:, 0]

    y_test_rescaled = scaler.inverse_transform(y_test_full)[:, ghi_index]
    y_pred_rescaled = scaler.inverse_transform(y_pred_full)[:, ghi_index]

    # Evaluation
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    print("\n LSTM Model Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")

    # Plot: Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_rescaled[:200], label='Actual GHI')
    plt.plot(y_pred_rescaled[:200], label='Predicted GHI', linestyle='--')
    plt.title("GHI Forecast (LSTM) - First 200 Predictions")
    plt.xlabel("Time Step (Hour)")
    plt.ylabel("GHI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/ghi_forecast_lstm.png")
    plt.show()
    

# Try out LSTM module for different values
'''
What to change	| How it Helps                               | How to Change in Code
sequence_length | More past hours may improve forecasting    | Can try 48, 72 instead of 24
LSTM units      | Bigger model can learn better patterns     | Try 32, 128, 256
epochs 		    | More training cycles 			             | Try 30–100
batch_size  	| Affects training stability & speed 	     | Try 16 or 64
layers 		    | Add more LSTM or Dense layers 	         | Stack layers
dropout 	    | Prevent overfitting                        | Add Dropout(0.2)


'''

if __name__ == "__main__":
    main()
