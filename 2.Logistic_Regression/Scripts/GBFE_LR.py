#Conducting Greedy Backward Feature Elimination using Logistic Regression model
#on BPAD200 dataset with all 525 features

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

#Reading scaled data with all features
url = "https://raw.githubusercontent.com/ShrutiBaikerikar/machine-learning-bioinformatics-paper-implementations/main/2.Logistic_Regression/Datasets/BPAD200.csv"
rv_data = pd.read_csv(url)
rv_data.rename(columns=lambda x:x.replace(" ","_"),inplace=True)
print(rv_data.head(3))

#Converting the label of -1 for non-BPAs into 0 for 0-1 classification
for index in range(rv_data.shape[0]):
    if rv_data.iloc[index,-1] == -1:
        rv_data.iloc[index,-1] = 0

#Defining X and Y
X = rv_data.drop(['Id','Label'],axis=1)
y = np.ravel(rv_data['Label']) #Flatten to 1D array

#Getting feature names
col_names = list(X.columns.values)

#Implementing Greedy Backward Feature Elimination - RFECV
model = LogisticRegression(solver='lbfgs',max_iter=1000)
min_features_to_select=1
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
rfecv = RFECV(model,min_features_to_select=10,cv= cv,step=1,scoring='accuracy')
rfecv.fit(X, y)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Classification accuracy(%)")
plt.plot(range(min_features_to_select,len(list(rfecv.cv_results_["mean_test_score"])) + min_features_to_select),
         (rfecv.cv_results_["mean_test_score"]*100))
plt.show()
