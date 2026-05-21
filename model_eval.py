import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_data(path):
    return pd.read_csv(path)

def prepare_data(df):
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y

def evaluate(model, X, y):
    y_pred = model.predict(X)

    # Reverse log transform
    y = np.expm1(y)
    y_pred = np.expm1(y_pred)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    return {"mse": mse, "rmse": rmse, "r2_score": r2}

def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

def main():
    model = load_model("model.pkl")
    test = load_data("data/processed/test_processed.csv")

    X, y = prepare_data(test)
    
    metrics = evaluate(model, X, y)

    os.makedirs("output_reports", exist_ok=True)
    save_metrics(metrics, "output_reports/metrics.json")

if __name__ == "__main__":
    main()