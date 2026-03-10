import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """
    Function to load data from a specified filepath.
    """
    return pd.read_csv(filepath)


def clean_data(df):
    """
    Function to clean the DataFrame by handling missing values.
    """
    # Example: Dropping rows with missing values
    return df.dropna()


def normalize_data(df):
    """
    Function to normalize sensor data.
    """
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


def split_data(df, target_column, test_size=0.2):
    """
    Function to split the DataFrame into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)
