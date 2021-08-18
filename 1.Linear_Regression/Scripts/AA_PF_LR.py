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
data = pd.read_csv("C:/Users/Shruti SB/PycharmProjects/ML_BI_papers/AA_PF/AA_train.csv")
#print(data.head())

#Converting feature 'structure' to numeric labels
labelEncoder = LabelEncoder()
labelEncoder.fit(data['Structure'])
data['Structure'] = labelEncoder.transform(data['Structure'])
#print(data.head(20))

#Creating X and y variables
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
feat_sel = SelectKBest(score_func=f_regression, k="all")
feat_sel.fit(X_train,y_train)

#creating a dictionary of features with their scores
sel_feat_all ={}
for i in range(0,len(cols)):
    sel_feat_all[cols[i]] = feat_sel.scores_[i]
print()
print("Feature Scoring according to Correlation")
print(sel_feat_all)

#sorting the dictionary in descending order to get highest scored features
sorted_features = dict( sorted(sel_feat_all.items(), key=operator.itemgetter(1),reverse=True))
print(sorted_features)

#Extracting top 10 features
top10_feats = list(sorted_features.keys())[0:10]
print(top10_feats)

#Applying Linear Regression on Top 10 features
X_train_top10 = X_train[top10_feats]
X_test_top10 = X_test[top10_feats]

model3 = SGDRegressor(fit_intercept=True,random_state=1)
model3.fit(X_train_top10,y_train)
print()
print("Linear Regression Results for Top 10 Features" )
print ("Intercept",model3.intercept_)

#printing coefficients
coeff_df = pd.DataFrame(model3.coef_, top10_feats, columns=['Coefficient'])
print("Values of Coefficients In Linear Model")
print(coeff_df)

#predicting on test data
y_pred = model3.predict(X_test_top10)

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