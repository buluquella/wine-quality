import pandas as pd

def load_and_clean_data(file_path):
    try:
        data = pd.read_csv(file_path, delimiter=';')
        
        null_counts = data.isnull().sum()
        print("Null counts in each column:\n", null_counts)

        cleaned_data = data.dropna()
        
        return cleaned_data

    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

dataset_path = 'data/winequality-white.csv'

cleaned_data = load_and_clean_data(dataset_path)

if cleaned_data is not None:
    print("First five rows of the cleaned data:\n", cleaned_data.head())
