import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file at {file_path} does not exist.")
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Example preprocessing steps
    data = data.dropna()  # Remove missing values
    # Add more preprocessing steps as needed
    return data

def split_data(data, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def save_preprocessed_data(data, output_path):
    data.to_csv(output_path, index=False)