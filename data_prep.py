import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns
    df = df.drop(
        columns=["date", "street", "country", "waterfront", "yr_renovated", "condition","sqft_lot","view"],
        errors="ignore"
    )

    # Remove outliers (IMPORTANT)
    df = df[df["price"] < df["price"].quantile(0.95)]

    # -------------------------------
    # 🔥 FEATURE ENGINEERING (SAFE)
    # -------------------------------
    # if 'sqft_living' in df.columns:
    #     df['sqft_log'] = np.log1p(df['sqft_living'])

    # if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
    #     df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # if 'floors' in df.columns and 'bedrooms' in df.columns:
    #     df['floor_ratio'] = df['floors'] / (df['bedrooms'] + 1)
    # Area interaction
    # if 'sqft_living' in df.columns and 'floors' in df.columns:
    #     df['area_per_floor'] = df['sqft_living'] / (df['floors'] + 1)

    #  Bedroom density
    # if 'sqft_living' in df.columns and 'bedrooms' in df.columns:
    #     df['bedroom_density'] = df['bedrooms'] / (df['sqft_living'] + 1)
    # -------------------------------

    # Log transform target (VERY IMPORTANT)
    df["price"] = np.log1p(df["price"])

    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Impute numeric values
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def save_data(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)


def main():
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"

    train = load_data(os.path.join(raw_data_path, "train.csv"))
    test = load_data(os.path.join(raw_data_path, "test.csv"))

    train = preprocess(train)
    test = preprocess(test)

    # Align columns (VERY IMPORTANT)
    train, test = train.align(test, join="left", axis=1, fill_value=0)

    os.makedirs(processed_data_path, exist_ok=True)

    save_data(train, os.path.join(processed_data_path, "train_processed.csv"))
    save_data(test, os.path.join(processed_data_path, "test_processed.csv"))


if __name__ == "__main__":
    main()