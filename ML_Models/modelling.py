# It applies all analytics steps on each city data
# output stored in plots folder
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Directory setup
CLEAN_DIR = "data"
# To Save all output images in plots folder at one glance 
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Utility function to evaluate and visualize predictions
def evaluate_and_plot(y_test, y_pred, model_name, city):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 {model_name} Performance on {city}:")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.2f}")

    # Plot: Actual vs Predicted
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel("Actual GHI")
    plt.ylabel("Predicted GHI")
    plt.title(f"Actual vs Predicted GHI ({model_name} - {city})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{city}_{model_name}_prediction.png")
    plt.close()

# Function to load each cleaned CSV, train models, and compare results
def train_models():
    all_results = []  # To store metrics for all models and cities

    for file in os.listdir(CLEAN_DIR):
        if "_cleaned" in file and file.endswith(".csv"):

            city = file.replace("_cleaned.csv", "")
            df = pd.read_csv(os.path.join(CLEAN_DIR, file))

            if 'ghi' not in df.columns:
                print(f"⚠️ Skipping {file} – no GHI column.")
                continue

            X = df.drop(columns=['ghi'])
            y = df['ghi']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # -------- XGBoost --------
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror', random_state=42)
            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)
            evaluate_and_plot(y_test, y_pred_xgb, "XGBoost", city)

            all_results.append({
                'City': city, 'Model': 'XGBoost',
                'MAE': mean_absolute_error(y_test, y_pred_xgb),
                'RMSE': mean_squared_error(y_test, y_pred_xgb) ** 0.5,
                'R2': r2_score(y_test, y_pred_xgb)
            })

            # -------- Random Forest --------
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            evaluate_and_plot(y_test, y_pred_rf, "RandomForest", city)

            all_results.append({
                'City': city, 'Model': 'RandomForest',
                'MAE': mean_absolute_error(y_test, y_pred_rf),
                'RMSE': mean_squared_error(y_test, y_pred_rf) ** 0.5,
                'R2': r2_score(y_test, y_pred_rf)
            })

    # Final summary table
    results_df = pd.DataFrame(all_results)
    print("\n📋 Combined Model Performance Summary:")
    print(results_df)

    # Save summary to CSV
    results_df.to_csv(os.path.join(PLOT_DIR, "model_performance_summary.csv"), index=False)



    # --- Combined Comparison Plot ---
    # Output saved in plots folder
    metrics = ['MAE', 'RMSE', 'R2']
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=results_df, x='City', y=metric, hue='Model')
        plt.title(f'{metric} Comparison Across Cities')
        plt.ylabel(metric)
        plt.xlabel("City")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/model_comparison_{metric}.png")
        plt.close()

if __name__ == "__main__":
    train_models()
