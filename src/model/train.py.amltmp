# Import libraries
import argparse
import glob
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow


def resolve_path(path: str) -> str:
    if path.startswith("azureml:"):
        # In local runs, replace with a test folder
        return "./data"   # <-- point to your local folder of CSVs
    return path


# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog()

    training_path = resolve_path(args.training_data)
    print(f"[DEBUG] Using training path: {training_path}")
    print(f"[DEBUG] os.path.exists? {os.path.exists(training_path)}")
    if os.path.exists(training_path):
        print(f"[DEBUG] Files in training path: {os.listdir(training_path)}")

    # read data
    df = get_csvs_df(training_path)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError("Cannot use non-existent path provided")

    if os.path.isfile(path):
        # If it's a single file
        if path.endswith(".csv"):
            print(f"[DEBUG] Found single CSV file: {path}")
            return pd.read_csv(path)
        else:
            raise RuntimeError("Provided file is not a CSV")

    if os.path.isdir(path):
        # If it's a folder, grab all CSV files inside
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            raise RuntimeError("No CSV files found in provided data")

        print(f"[DEBUG] Found {len(csv_files)} CSVs in folder {path}")
        df_list = [pd.read_csv(f) for f in csv_files]
        return pd.concat(df_list, ignore_index=True)

    raise RuntimeError(f"[ERROR] Path is neither a file nor folder: {path}")


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
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--training_data",
        dest="training_data",
        type=str,
        required=True,
        help="Path to training data(uri_folder or local path)"
    )
    parser.add_argument(
        "--reg_rate",
        dest="reg_rate",
        type=float,
        default=0.01
    )

    # parse args
    args = parser.parse_args()

    # return all the argees
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
