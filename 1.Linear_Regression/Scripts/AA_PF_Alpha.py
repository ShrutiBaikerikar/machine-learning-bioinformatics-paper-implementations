# Linear Regression on Alpha proteins dataset
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#reading data
url = "https://github.com/ShrutiBaikerikar/machine-learning-bioinformatics-paper-implementations/blob/main/1.Linear Regression/Datasets/alphaproteins.csv"
data = pd.read_csv(url)
#print(data.head())

#Splitting the data into test and train groups as per the author's choice of proteins and features
X_train = data['aC'][0:6]
#print(X_train)
X_train = X_train.values.reshape(-1,1)
X_test = data['aC'][6:9]
X_test = X_test.values.reshape(-1,1)
#print(X_test)

y_train = data['lnKf'][0:6]
y_train = y_train.values.reshape(-1,1)
#print(y_train)
y_test = data['lnKf'][6:9]
y_test = y_test.values.reshape(-1,1)
#print(y_test)

#selecting features that were used by the author
cols = ['aC']

#performing linear regression on all features
model = LinearRegression(fit_intercept=True)
model.fit(X_train,y_train)
print("Linear Regression Results" )
print ("Intercept",model.intercept_[0],"Coefficient", model.coef_[0])

#printing coefficients of selected features
coeff_df = pd.DataFrame(model.coef_, cols, columns=['Coefficient'])
print("Value of Coefficients In Linear Model")
print(coeff_df)

#predicting on train data
y_train_pred = model.predict(X_train)

#compare the actual output values for X_train with the predicted values
df_pred_train = pd.DataFrame({'Actual': y_train.flatten(), 'Predicted': y_train_pred.flatten()})
print("----------")
print("Predictions on Training data")
print(df_pred_train)

#predicting on test data
y_pred = model.predict(X_test)

#compare the actual output values for X_test with the predicted values
df_pred_test = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print("----------")
print("Predictions on Test data")
print(df_pred_test)

#plot differences
df_pred_test.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print("----------")
print("Evaluation on Test data")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Coefficient of determination:',metrics.r2_score(y_test, y_pred))
