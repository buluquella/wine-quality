import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

file_path = 'winequality-white.csv'
data = pd.read_csv(file_path, delimiter=';')

###############
# Decision Tree
###############

selected_features = ['volatile acidity', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']

X = data[selected_features]

y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

classification_report_output = classification_report(y_test, y_pred)
confusion_matrix_output = confusion_matrix(y_test, y_pred)
accuracy_score_output = accuracy_score(y_test, y_pred)

print(classification_report_output, confusion_matrix_output, accuracy_score_output)

###############
# Random Forest
###############

X_white = data[selected_features]
y_white = data['quality']

X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_white, y_white, test_size=0.01, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest.fit(X_train_white, y_train_white)

y_pred_rf_white = random_forest.predict(X_test_white)

rf_classification_report_white = classification_report(y_test_white, y_pred_rf_white)
rf_confusion_matrix_white = confusion_matrix(y_test_white, y_pred_rf_white)
rf_accuracy_score_white = accuracy_score(y_test_white, y_pred_rf_white)

print(rf_classification_report_white, rf_confusion_matrix_white, rf_accuracy_score_white)