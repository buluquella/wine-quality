import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Set the file path for the dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, '..', '..', 'data', 'winequality-white.csv')

# Load the dataset
data = pd.read_csv(file_path, delimiter=';')

# Select features and target
selected_features = ['volatile acidity', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']
X = data[selected_features]
y = data['quality']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Finding the best k value
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

# Plotting the results to find the best k value
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding Optimal K Value')
plt.show()

# Choose the best k and retrain the model
best_k = k_scores.index(max(k_scores)) + 1  # Adding 1 because index starts at 0
knn_optimized = KNeighborsClassifier(n_neighbors=best_k)
knn_optimized.fit(X_train_scaled, y_train)

# Predict on the test set with optimized KNN
y_pred_optimized = knn_optimized.predict(X_test_scaled)

# Output results
print(f"Optimized K-Nearest Neighbors Model Results (k={best_k}):")
print(classification_report(y_test, y_pred_optimized))
print(confusion_matrix(y_test, y_pred_optimized))
print("Accuracy Score:", accuracy_score(y_test, y_pred_optimized))
