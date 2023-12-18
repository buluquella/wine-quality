import pandas as pd
import numpy as np

# Step 1: Load the Data
# Replace 'path_to_dataset.csv' with the actual file path
dataset_path = 'winequality-white.csv'
data = pd.read_csv(dataset_path, delimiter=';')

# Step 2: Check for Missing Values
print(data.isnull().sum())  # Prints the count of missing values per column

# Handling missing values (if any)
# For simplicity, we will drop rows with missing values. 
# In a real project, consider other methods like imputation.
data = data.dropna()

# Step 3: Data Transformation (if needed)
# For example, converting a categorical variable using one-hot encoding
# data = pd.get_dummies(data)

# Step 4: Feature Scaling (optional here, but often necessary before modeling)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# select columns to scale
# scale_columns = ['column1', 'column2', 'column3']
# data[s
