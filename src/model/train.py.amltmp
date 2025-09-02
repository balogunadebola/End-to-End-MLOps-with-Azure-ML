# Import libraries
import argparse
import glob
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow


# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path: str) -> pd.DataFrame:
    """Reads all CSV files from a directory into a single DataFrame."""
    if not os.path.exists(path):
        raise RuntimeError(
            "Cannot use non-existent path provided: "
            f"{path}"
        )

    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not files:
        raise RuntimeError(
            f"No CSV files found in directory: {path}"
        )

    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def split_data(df):
    """
    Split the data into training and testing sets

    Parameters:
    df (DataFrame): The input dataframe containing the data

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    # Extract features and target
    X = df[
        [
            "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
            "TricepsThickness", "SerumInsulin", "BMI",
            "DiabetesPedigree", "Age"
        ]
    ].values
    y = df["Diabetic"].values

    # Log class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )

    # Log split information
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_model(data_path: str, reg_rate: float):
    """Trains a Ridge regression model with the given data and regularization."""
    df = get_csvs_df(data_path)

    if "target" not in df.columns:
        raise RuntimeError("Expected a 'target' column in the dataset")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Ridge(alpha=reg_rate)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Model trained with regularization rate {reg_rate}")
    print(f"Mean Squared Error on test set: {mse:.4f}")

    return model, mse


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Ridge regression model."
    )
    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=str,
        required=True,
        help="Path to the folder containing CSV files.",
    )
    parser.add_argument(
        "--reg_rate",
        dest="reg_rate",
        type=float,
        default=0.11,
        help="Regularization rate for Ridge regression.",
    )
    return parser.parse_args()


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    train_model(args.data_path, args.reg_rate)

    # run main function
    main(args)

    # add space in logs
    print("*" * 10)
    print("\n\n")
