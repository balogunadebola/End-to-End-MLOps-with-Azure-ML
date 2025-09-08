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


def get_csvs_df(path):
    """Read data from a CSV file or from a folder with CSVs."""
    print(f"[DEBUG] Looking for data in path: {path}")

    if not os.path.exists(path):
        error_msg = f"Cannot use non-existent path provided: {path}"
        raise RuntimeError(error_msg)

    # Debug: List contents of the mounted directory
    print(f"[DEBUG] Directory contents: {os.listdir(path)}")

    # Azure ML mounts data assets in a specific structure
    # Check if this is an Azure ML mounted path with data subdirectory
    data_subdir = os.path.join(path, "data")
    if os.path.exists(data_subdir):
        print(f"[DEBUG] Found 'data' subdirectory, using: {data_subdir}")
        path = data_subdir
        print(f"[DEBUG] Data subdirectory contents: {os.listdir(path)}")

    # Assume CSV(s) - first check if it's a single CSV file
    if os.path.isfile(path):
        if path.endswith("csv"):
            print(f"[DEBUG] Reading single CSV file: {path}")
            return pd.read_csv(path)
        error_msg = f"Provided file is not a CSV: {path}"
        raise RuntimeError(error_msg)

    # If path is a directory, look for CSV files recursively
    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
    print(f"[DEBUG] Found CSV files: {csv_files}")

    if not csv_files:
        # Also check for CSV files without recursive search
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        print(f"[DEBUG] Found CSV files (non-recursive): {csv_files}")

    if not csv_files:
        error_msg = (
            f"No CSV files found in provided data path: {path}. "
            f"Contents: {os.listdir(path)}"
        )
        raise RuntimeError(error_msg)

    print(f"[DEBUG] Loading {len(csv_files)} CSV files")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    """
    Split the data into training and testing sets.

    Parameters:
    df (DataFrame): The input dataframe containing the data

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    # Extract features and target
    X = df[
        [
            "Pregnancies",
            "PlasmaGlucose",
            "DiastolicBloodPressure",
            "TricepsThickness",
            "SerumInsulin",
            "BMI",
            "DiabetesPedigree",
            "Age"
        ]
    ].values
    y = df["Diabetic"].values

    # Log class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    print(f"Class distribution: {class_distribution}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )

    # Log split information
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(
        C=1 / reg_rate,
        solver="liblinear"
    ).fit(X_train, y_train)
    return model


def parse_args():
    # setup arg parsers
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--training_data",
        dest="training_data",
        type=str
    )
    parser.add_argument(
        "--reg_rate",
        dest="reg_rate",
        type=float,
        default=0.05
    )

    # parse args
    args = parser.parse_args()

    # return all the args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 25)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 10)
    print("\n\n")