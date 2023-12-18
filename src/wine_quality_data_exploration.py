import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'data/winequality-white.csv'
data = pd.read_csv(file_path, delimiter=';')

print("Data Types:\n", data.dtypes)

print("\nStatistical Summary:\n", data.describe())

data.hist(bins=10, figsize=(20, 15))
plt.suptitle('Histograms of Various Features')
plt.show()

for col in data.columns:
    data.boxplot(column=col)
    plt.title(f'Box Plot of {col}')
    plt.show()

plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
plt.title('Correlation Heatmap')
plt.show()
