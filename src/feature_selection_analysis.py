from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np

file_path = 'data/winequality-white.csv'
data = pd.read_csv(file_path, delimiter=';')

X = data.drop('quality', axis=1)
y = data['quality']

select_k_best = SelectKBest(score_func=f_classif, k='all')
select_k_best_scores = select_k_best.fit(X, y).scores_
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': select_k_best_scores})
print(feature_scores.nlargest(10, 'Score'))

random_forest = RandomForestClassifier()
random_forest.fit(X, y)
importances = random_forest.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
print("\nFeature ranking:")
for index in sorted_indices:
    print(f"{index + 1}. feature {X.columns[index]} ({importances[index]})")

rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
print('\nSelected features by RFE:', selected_features)
