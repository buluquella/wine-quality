import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

file_path = os.path.join(current_dir, '..', '..', 'data', 'winequality-white.csv')
data = pd.read_csv(file_path, delimiter=';')

selected_features = ['volatile acidity', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']
X = data[selected_features]
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print("Decision Tree Model Results:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
