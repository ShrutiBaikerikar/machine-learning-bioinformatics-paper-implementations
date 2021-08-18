#Linear Regression on Mixed Proteins

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#################################################################################################################
#Mixed Proteins
#Mean Absolute Error: 1.923052400431655
#Mean Squared Error: 5.406618024856456
#Root Mean Squared Error: 2.3252135439259027

data = pd.read_csv("C:/Users/Shruti SB/PycharmProjects/ML_BI_papers/AA_PF/mixedproteins.csv")
#print(data.head())

#selecting features as specified by the author
cols = ['K0','Ra','dASA','GhD']

#Splitting data into train and test sets based on author's choice of proteins
X_train = data.loc[0:12]
#print(X_train)
X_train = X_train[cols]
#print(X_train)

X_test = data.loc[13:22]
#print(X_test)
X_test = X_test[cols]
#print(X_test)

y_train = data['lnKf'][0:13]
#print(y_train)
y_test = data['lnKf'][13:25]
#print(y_test)

#Linear regression on selected features
model = LinearRegression()
model.fit(X_train, y_train)
print ("Intercept",model.intercept_)

#printing coefficients
coeff_df = pd.DataFrame(model.coef_, cols, columns=['Coefficient'])
print("Values of Coefficients In Linear Model")
print(coeff_df)

#prediction on train dataset
y_train_pred = model.predict(X_train)

#compare actual and predicted on train dataset
df_train_pred = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
print("-----------")
print("Predictions on Training data")
print(df_train_pred)

#prediction on test
y_test_pred = model.predict(X_test)

#compare actual and predicted on test dataset
df_test_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
print("-----------")
print("Predictions on Test data")
print(df_test_pred)

#plot the differences
df_test_pred.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#evaluate algorithm
print("-----------")
print("Evaluation on Test data")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('Coefficient of determination:',metrics.r2_score(y_test, y_test_pred))