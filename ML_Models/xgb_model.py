
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from battery_planner import run_battery_planner  # ✅ Battery integration

def main():
    # Load cleaned data
    df = pd.read_csv("cleaned_data.csv")

    # Feature selection
    features = ['temperature',  'hour', 'month', 'cloud_type', 'relative_humidity']
    target = 'ghi'

    # Drop missing values
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost
    model = XGBRegressor()
    model.fit(X_train, y_train)
    print("✅ XGBoost model trained successfully!")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n Model Performance (XGBoost):")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # --- Rolling MAE Anomaly Detection ---
    results = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred
    })
    results['error'] = np.abs(results['actual'] - results['predicted'])
    results['index'] = range(len(results))
    results = results.set_index('index')

    # 7-day = 168-hour rolling window (assuming hourly data)
    results['rolling_mae'] = results['error'].rolling(window=168).mean()

    # Calculate overall MAE
    mean_mae = results['error'].mean()

    # Anomaly warning
    if results['rolling_mae'].max() > 1.5 * mean_mae:
        print(" Model performance degradation detected — potential anomaly (e.g., early monsoon behavior)")

    # Plot Rolling MAE
    plt.figure(figsize=(10, 4))
    plt.plot(results['rolling_mae'], label='7-day Rolling MAE', color='orange')
    plt.axhline(mean_mae, color='red', linestyle='--', label='Overall MAE')
    plt.title('Rolling MAE - Anomaly Detection')
    plt.xlabel('Hour Index')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/rolling_mae_plot.png")
    plt.show()

    # --- Battery Storage Planning ---
    predicted_ghi_series = pd.Series(y_pred)
    run_battery_planner(predicted_ghi_series)

if __name__ == "__main__":
    main()