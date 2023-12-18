from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np

# Step 1: Load the Data
# Replace 'path_to_dataset.csv' with the actual file path
dataset_path = 'winequality-white.csv'
data = pd.read_csv(dataset_path, delimiter=';')

# Assuming `data` is your DataFrame and 'quality' is the target variable
X = data.drop('quality', axis=1)  # Independent variables
y = data['quality']  # Dependent variable (target)

# Using SelectKBest to select top features based on f-test
bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X, y)

# Get scores for each feature
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# Concatenate two dataframes for better visualization and print the scores
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'Score']
print(featureScores.nlargest(10, 'Score'))  # print 10 best features

# Using RandomForest to get feature importances
rf = RandomForestClassifier()
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

# Using RFE to select top n features
rfe = RFE(estimator=rf, n_features_to_select=5)
rfe.fit(X, y)
print('Selected features by RFE:', X.columns[rfe.support_])
