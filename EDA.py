import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dataset_path = 'winequality-white.csv'
data = pd.read_csv(dataset_path, delimiter=';')

print(data.dtypes)

# Display basic statistical details like percentile, mean, std etc.
print(data.describe())

# Exploratory Data Analysis (EDA)
# Histograms for each feature
data.hist(bins=10, figsize=(20,15))
plt.show()

# Box plots for understanding distributions and spotting outliers
for col in data.columns:
    data.boxplot(column=col)
    plt.title(col)
    plt.show()

# Correlation Matrix to understand the relationship between different features
plt.figure(figsize=(12,10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
plt.show()
