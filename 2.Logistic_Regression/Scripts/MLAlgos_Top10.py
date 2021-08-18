# Comparing different classification machine learning algorithms on
# BPAD200 dataset containing only Top 10 features as given in the research paper

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt


# Reading the dataset
rv_data = pd.read_csv("C:/Users/Shruti SB/PycharmProjects/ML_BI_papers/Reverse_vaccine/BPAD200_TOP10.csv")
rv_data.rename(columns=lambda x:x.replace(" ","_"),inplace=True)
print(rv_data.head(3))

# Converting the label of -1 for non-BPAs into 0 for 0-1 classification
#print(rv_data.iloc[216,-1])
for index in range(rv_data.shape[0]):
    if rv_data.iloc[index,-1] == -1:
        rv_data.iloc[index,-1] = 0

#print(rv_data.iloc[216,-1])

# Defining X and Y
X = rv_data.drop(['Id','Label'],axis=1)
y = np.ravel(rv_data['Label']) #Flatten to 1D array

####################################################################################################################
####################################################################################################################
###################################################################################################################
# Logistic Regression Model
lr_model = LogisticRegression(solver='lbfgs')
# define cross validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
lr_scores = cross_val_score(lr_model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:' ,np.mean(lr_scores))

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# Decision Tree Classifier
dt_model = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=3,min_samples_leaf=2,random_state=1)
# evaluate model
dt_scores = cross_val_score(dt_model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:' ,np.mean(dt_scores))

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# Random Forest Classifier

# RF-10 trees
rf_10_model = RandomForestClassifier(n_estimators=10,criterion="gini",max_depth=30,min_samples_split=3,
                                bootstrap=True,max_features='auto',random_state=1,min_samples_leaf=1)
# evaluate model
rf_10_scores = cross_val_score(rf_10_model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:' ,np.mean(rf_10_scores))

# RF-100 trees
rf_100_model = RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=5,min_samples_split=2,
                                bootstrap=True,max_features='auto',random_state=1,min_samples_leaf=1)
# evaluate model
rf_100_scores = cross_val_score(rf_100_model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:' ,np.mean(rf_100_scores))

# RF-1000 trees
rf_1000_model = RandomForestClassifier(n_estimators=1000,criterion="gini",max_depth=5,min_samples_split=3,
                                bootstrap=True,max_features='auto',random_state=1,min_samples_leaf=1)
# evaluate model
rf_1000_scores = cross_val_score(rf_1000_model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:' ,np.mean(rf_1000_scores))

###################################################################################################################
##################################################################################################################
###################################################################################################################
# Support Vector Machines

# SVM_Linear
svm_linear_model = SVC(kernel='linear',C=3.0,random_state=1)
# evaluate model
svm_linear_scores = cross_val_score(svm_linear_model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:',np.mean(svm_linear_scores))

# SVM_Polynomial
svm_poly_model = SVC(kernel='poly',C=1.0,degree=2,random_state=1)
# evaluate model
svm_poly_scores = cross_val_score(svm_poly_model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:',np.mean(svm_poly_scores))

# SVM_RBF
svm_rbf_model = SVC(kernel='rbf',C=10,gamma=0.1,random_state=1)
# evaluate model
svm_rbf_scores = cross_val_score(svm_rbf_model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:',np.mean(svm_rbf_scores))

#######################################################################################################
#######################################################################################################
######################################################################################################
# Plotting results
# boxplot of all scores of all ML models
total_scores = [svm_linear_scores, svm_poly_scores, svm_rbf_scores, lr_scores, dt_scores, rf_10_scores
                ,rf_100_scores, rf_1000_scores]
labels_scores = ['SVM_linear','SVM_poly','SVM_rbf','Logistic_regression','Decision_Tree','RF_10trees',
                 'RF_100trees','RF_1000trees']

fig1, ax1 = plt.subplots()
plt.title('Comparison of AUC with different algorithms to predict BPA')
plt.xlabel("Algorithms to predict BPA")
plt.ylabel("Area under the Curve")
colors = ['darkblue', 'darkblue','darkblue','blue','lightsteelblue', 'lightsteelblue','lightsteelblue','lightsteelblue']

bp= ax1.boxplot(total_scores,labels=labels_scores,patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.show()