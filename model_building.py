from xgboost import XGBRegressor
import pandas as pd
import pickle
import yaml
from sklearn.linear_model import LinearRegression
def load_params(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["model_building"]["n_estimators"]

def load_data(path):
    return pd.read_csv(path)

def prepare_data(df):
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y

def train_model(X, y, n_estimators):
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lamda=1,
        random_state=42
    )
    # model = LinearRegression()
    model.fit(X, y)
    return model

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def main():
    data = load_data("data/processed/train_processed.csv")
    X, y = prepare_data(data)

    n_estimators = load_params("params.yaml")

    model = train_model(X, y, n_estimators)

    save_model(model, "model.pkl")

if __name__ == "__main__":
    main()