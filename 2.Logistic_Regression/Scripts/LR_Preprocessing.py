# Logistic Regression comparison on preprocessed/scaled data including all features
# The original research paper also compared the effect of Logistic Regression
# on the raw data vs preprocessed data.
# Since we do not have the raw data, we are excluding that from the analysis.

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt

#Reading scaled/preprocessed data with all features
url_1 = "https://raw.githubusercontent.com/ShrutiBaikerikar/machine-learning-bioinformatics-paper-implementations/main/2.Logistic_Regression/Datasets/BPAD200.csv"
rv_data_scaled = pd.read_csv(url_1)
rv_data_scaled.rename(columns=lambda x:x.replace(" ","_"),inplace=True)
print(rv_data_scaled.head(3))

#Converting the label of -1 for non-BPAs into 0 for 0-1 classification
for index in range(rv_data_scaled.shape[0]):
    if rv_data_scaled.iloc[index,-1] == -1:
        rv_data_scaled.iloc[index,-1] = 0

#Defining X and Y
X = rv_data_scaled.drop(['Id','Label'],axis=1)
y = np.ravel(rv_data_scaled['Label']) #Flatten to 1D array

#######################################################################################################
#######################################################################################################
#######################################################################################################
#Reading scaled data with top10 features from BPAD200 dataset
url_2 = "https://raw.githubusercontent.com/ShrutiBaikerikar/machine-learning-bioinformatics-paper-implementations/main/2.Logistic_Regression/Datasets/BPAD200_Top10.csv"
rv_data_top10 = pd.read_csv(url_2)
rv_data_top10.rename(columns=lambda x:x.replace(" ","_"),inplace=True)
print(rv_data_top10.head(3))

#Converting the label of -1 for non-BPAs into 0 for 0-1 classification
for index in range(rv_data_top10.shape[0]):
    if rv_data_top10.iloc[index,-1] == -1:
        rv_data_top10.iloc[index,-1] = 0

#Defining X and Y
X_top10 = rv_data_top10.drop(['Id','Label'],axis=1)
y_top10 = np.ravel(rv_data_top10['Label']) #Flatten to 1D array


######################################################################################################################
######################################################################################################################
#Logistic Regression on Preprocessed data (pdata)
model_pdata = LogisticRegression(solver='lbfgs',max_iter=1000)
# define cross-validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores_pdata = cross_val_score(model_pdata, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:' ,np.mean(scores_pdata))

###############################################################################################################
#Logistic Regression on Preprocessed data including only Top 10 features identified by GBFE using SVM (SVMTop10)
model_SVMtop10 = LogisticRegression(solver='lbfgs')
# evaluate model
scores_SVMtop10 = cross_val_score(model_SVMtop10, X_top10, y_top10, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:' ,np.mean(scores_SVMtop10))

##############################################################################################################
# Logistic Regression on Preprocessed data including only Top 10 features identified by GBFE using LR (LRTop10)
# These features were obtained by performing Greedy Backward Feature Elimination using Logistic Regression

X_LR_top10 = X[['DictOGlyc_Average_Threshold_Length', 'YinOYang-AvgDiff1', 'LipoP_CleavII_Avg_Length',
                'NetOGlyc_Count_Length', 'NetOGlyc-T_Count_Score_Length', 'GPS-ARM_Dbox_AverageDiff',
                'CCD_av_diff', 'MBAAgl7_CorCount', 'SNO_Length_Count', 'NetsurfP_RSA_Buried_MaxDiff']]
print(X_LR_top10.shape)

model_LRtop10 = LogisticRegression(solver='lbfgs',max_iter=1000)
# evaluate model
scores_LRtop10 = cross_val_score(model_LRtop10, X_LR_top10, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# calculate mean ROC AUC
print('Mean ROC AUC:' ,np.mean(scores_LRtop10))

#################################################################################################################
#################################################################################################################
#################################################################################################################
#boxplot of all scores on preprocessed data
total_scores = [scores_pdata, scores_SVMtop10, scores_LRtop10]
labels_scores = ['Preprocessed data','Preprocessed_Top10_GBFE_SVM','Preprocessed_Top10_GBFE_LR']

fig1, ax1 = plt.subplots()
plt.title('Comparison of AUC with preprocessing of data')
plt.xlabel("Data")
plt.ylabel("Area under the Curve")
colors = ['navy','mediumblue','lightsteelblue']

bp= ax1.boxplot(total_scores,labels=labels_scores,patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.show()
