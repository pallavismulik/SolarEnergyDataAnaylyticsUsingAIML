# model_train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    #  Load Cleaned Data
    df = pd.read_csv("cleaned_data.csv")

    # Define Features (X) and Target (y)
    X = df.drop(columns=['ghi'])  # 'ghi' is our prediction target
    y = df['ghi']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model elevation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n Model Evaluation:")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.2f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel("Actual GHI")
    plt.ylabel("Predicted GHI")
    plt.title("Actual vs Predicted GHI")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #  Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    #plt.show()

# Run this only if file is executed directly
if __name__ == "__main__":
    main()
