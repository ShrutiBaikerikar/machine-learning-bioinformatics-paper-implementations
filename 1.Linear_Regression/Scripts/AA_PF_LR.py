#Performing Feature Selection and applying Linear Regression on entire dataset

import numpy as np
import pandas as pd
import operator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt

#reading the data
url = "https://raw.githubusercontent.com/ShrutiBaikerikar/machine-learning-bioinformatics-paper-implementations/main/1.Linear_Regression/Datasets/AA_train.csv"
data = pd.read_csv(url)
#print(data.head())

#Converting feature 'structure' to numeric labels
labelEncoder = LabelEncoder()
labelEncoder.fit(data['Structure'])
data['Structure'] = labelEncoder.transform(data['Structure'])
#print(data.head(20))

#Creating set of explanatory and target variables i.e. X and y variables
y = data['lnKf']
X = data.drop(['PDB_Code','lnKf'],axis=1)
cols = list(X.columns)

#Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)

##################################################################################################
##################################################################################################
##################################################################################################
#Linear Regression with Stochastic Gradient Descent on Entire Dataset

model1 = SGDRegressor(fit_intercept=True,random_state=1)
model1.fit(X_train,y_train)
print("Linear Regression Results" )
print ("Intercept",model1.intercept_)

#printing coefficients
coeff_df = pd.DataFrame(model1.coef_, cols, columns=['Coefficient'])
print("Values of Coefficients In Linear Model")
print(coeff_df)

#predicting on test data
y_pred = model1.predict(X_test)

#compare the actual output values for X_test with the predicted values
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(df2)

#plot differences
df2.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print("----------")
print("Final Evaluation Metrics on Test dataset")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Coefficient of determination:',metrics.r2_score(y_test, y_pred))

#####################################################################################################
#####################################################################################################
#####################################################################################################
#Linear Regression with Stochastic Gradient Descent on Informative Features according to the Author

#Selecting only specific features and performing train-test split
sel_cols = ['Structure','K0','Pb','aC','Ra','dASA','GhD']
X_sel = X[sel_cols]

X_sel_train,X_sel_test,y_sel_train,y_sel_test = train_test_split(X_sel,y,train_size=0.8,random_state=0)

#Linear Regression with Stochastic Gradient Descent
model2 = SGDRegressor(fit_intercept=True,random_state=1)
model2.fit(X_sel_train,y_sel_train)
print()
print("Linear Regression Results on Author Selected Informative Features" )
print ("Intercept",model2.intercept_)

#printing coefficients
coeff_df = pd.DataFrame(model2.coef_, sel_cols, columns=['Coefficient'])
print("Values of Coefficients In Linear Model")
print(coeff_df)

#predicting on test data
y_pred = model2.predict(X_sel_test)

#compare the actual output values for X_test with the predicted values
df3 = pd.DataFrame({'Actual': y_sel_test, 'Predicted': y_pred})
#print(df3)

#plot differences
df3.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print("----------")
print("Final Evaluation Metrics on Test dataset with Author Selected Informative Features")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_sel_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_sel_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_sel_test, y_pred)))
print('Coefficient of determination:',metrics.r2_score(y_sel_test, y_pred))

################################################################################################
################################################################################################
################################################################################################
#Linear Regression with Feature Importance based on Correlation

# Top 10 Feature selection based on correlation
feat_sel = SelectKBest(score_func=f_regression, k=10)
feat_sel.fit(X_train,y_train)
# transform train input data
X_train_fs = feat_sel.transform(X_train)
# transform test input data
X_test_fs = feat_sel.transform(X_test)

#print(feat_sel.scores_)
#print(feat_sel.get_feature_names_out())
#print(X_train.shape)
#print(X_train_fs.shape)

#Getting the list of the top10 selected features
top10_feats = feat_sel.get_feature_names_out()

model3 = SGDRegressor(fit_intercept=True,random_state=1)
model3.fit(X_train_fs,y_train)
print()
print("Linear Regression Results for Top 10 Features" )
print ("Intercept",model3.intercept_)

#printing coefficients
coeff_df = pd.DataFrame(model3.coef_, top10_feats, columns=['Coefficient'])
print("Values of Coefficients In Linear Model")
print(coeff_df)

#predicting on test data
y_pred = model3.predict(X_test_fs)

#compare the actual output values for X_test with the predicted values
df3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(df3)

#plot differences
df3.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print("----------")
print("Final Evaluation Metrics on Test dataset with Top 10 Features")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Coefficient of determination:',metrics.r2_score(y_test, y_pred))
