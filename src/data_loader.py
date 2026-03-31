"""
Data Loading & Preprocessing Module

This module handles loading the California Housing dataset (used as the primary
dataset for this project) and provides preprocessing utilities.

The California Housing dataset is built into scikit-learn and contains:
- 20,640 samples with 8 features
- Target: Median house value (in $100,000s)
- Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Readme asks us to download the data.  The function asks us to store as memory.  Therefore, download_data will
# download and save the data to the data/raw folder and can be run from the cli.

BASE_DIR = Path(__file__).resolve().parent  # sets current path (03_real_estate_prediction/src/)
ROOT_DIR = BASE_DIR.parent  # sets the project root directory (03_real_estate_prediction/)

def download_data():
    """Download and save data to the local drive."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    # build data directory
    DATA_DIR = ROOT_DIR / "data" / "raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file = DATA_DIR / 'housing.csv'
    # save file to data directory
    if file.is_file():
        print("File already exists")
    else:
        df.to_csv(file, index=False)
        print(f'Data has been saved to {file}')
    return None



def load_housing_data():
    """
    Load the California Housing dataset and return it as a pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame with feature columns and a 'MedHouseVal' target column.

    Example:
        >>> df = load_housing_data()
        >>> df.shape[1]  # 8 features + 1 target
        9
        >>> 'MedHouseVal' in df.columns
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Use fetch_california_housing(as_frame=True)
    #   2. The target variable should be named 'MedHouseVal'
    #   3. Return a single DataFrame with features AND target combined



def preprocess_features(df, target_col='MedHouseVal'):
    """
    Separate features and target, then apply standard scaling to features.

    Args:
        df (pd.DataFrame): Full dataset including target column.
        target_col (str): Name of the target column.

    Returns:
        tuple: (X_scaled, y, feature_names, scaler)
            - X_scaled (np.ndarray): Scaled feature matrix
            - y (np.ndarray): Target values
            - feature_names (list): List of feature column names
            - scaler (StandardScaler): Fitted scaler object

    Example:
        >>> df = load_housing_data()
        >>> X_scaled, y, names, scaler = preprocess_features(df)
        >>> X_scaled.shape[1] == len(names)
        True
        >>> np.abs(X_scaled.mean(axis=0)).max() < 1e-10  # means ≈ 0
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Separate X (features) and y (target)
    #   2. Fit a StandardScaler on X
    #   3. Return the scaled X, y, feature names, and the scaler
    raise NotImplementedError("Implement preprocess_features()")


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target values.
        test_size (float): Proportion of data for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    Example:
        >>> X = np.random.rand(100, 5)
        >>> y = np.random.rand(100)
        >>> X_train, X_test, y_train, y_test = split_data(X, y)
        >>> len(X_train) == 80
        True
        >>> len(X_test) == 20
        True
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement split_data()")


def create_feature_engineering(df):
    """
    Create new engineered features from the existing dataset.

    Required new features:
        - 'rooms_per_household': AveRooms * AveOccup (total rooms proxy)
        - 'bedrooms_ratio': AveBedrms / AveRooms
        - 'population_density': Population / AveOccup (households proxy)

    Args:
        df (pd.DataFrame): Original DataFrame with housing features.

    Returns:
        pd.DataFrame: DataFrame with original + new engineered features.

    Example:
        >>> df = load_housing_data()
        >>> df_eng = create_feature_engineering(df)
        >>> 'rooms_per_household' in df_eng.columns
        True
        >>> 'bedrooms_ratio' in df_eng.columns
        True
        >>> 'population_density' in df_eng.columns
        True
        >>> df_eng.shape[1] > df.shape[1]
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Make a copy of df to avoid modifying the original
    #   2. Create the three new features described above
    #   3. Handle potential division by zero cases
    raise NotImplementedError("Implement create_feature_engineering()")


if __name__ == "__main__":
    # Quick test: load and display dataset info
    print("Loading California Housing dataset...")
    df = load_housing_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df.describe())
