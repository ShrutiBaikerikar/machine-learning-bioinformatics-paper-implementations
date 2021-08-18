#Implementing Leave One Bacteria Out (LOBOV) strategy to predict BPAs

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt

#Reading data with all features
url = "https://raw.githubusercontent.com/ShrutiBaikerikar/machine-learning-bioinformatics-paper-implementations/main/2.Logistic_Regression/Datasets/BPAD200_Species.csv"
rv_data = pd.read_csv(url)
rv_data.rename(columns=lambda x:x.replace(" ","_"),inplace=True)
print(rv_data.head(3))

#Converting the label of -1 for non-BPAs into 0 for 0-1 classification

for index in range(rv_data.shape[0]):
    if rv_data.iloc[index,-1] == -1:
        rv_data.iloc[index,-1] = 0

#Defining a list of bacterial species in the dataset
bacteria_list=['Bacillus anthracis','Borrelia burgdorferi','Campylobacter jejuni',
               'Haemophilus influenzae','Helicobacter pylori','Pseudomonas aeruginosa',
               'Staphylococcus aureus','Streptococcus agalactiae','Streptococcus pneumoniae',
               'Streptococcus pyogenes','Streptococcus suis','Treponema pallidum','Escherichia coli',
               'Bordetella pertussis','Brucella abortus','Brucella melitensis','Chlamydia trachomatis',
               'Chlamydophila pneumoniae','Leptospira interrogans','Mycobacterium bovis','Mycobacterium tuberculosis',
               'Neisseria meningitidis','Salmonella enterica','Yersinia pestis']

#define model
model = LogisticRegression(max_iter=1000)
# define cross-validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#AUC lists
train_auc = []
test_auc = []


for bacteria in bacteria_list:
    print(bacteria)
    new = rv_data.copy()
    #Including bacterial species in Test data
    data_test = new[new['Species'].isin([bacteria])]
    print("Data_test:",data_test.shape)
    #Exluding the same bacterial species from Training data
    data_train = new[new.Species != bacteria]
    print("Data_train:",data_train.shape)

    #training data
    X_train = data_train.drop(['Id','Species','Label'],axis=1)
    y_train = np.ravel(data_train['Label'])

    #test data
    X_test = data_test.drop(['Id','Species','Label'],axis=1)
    y_test = np.ravel(data_test['Label'])

    #evaluating model via cross validation
    scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    train_auc.append(np.mean(scores))
    model.fit(X_train,y_train)

    #making predictions
    y_pred = model.predict(X_test)
    #calculating AUC values for prediction
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    test_auc.append(metrics.auc(fpr, tpr))


# creating the bar plot
fig = plt.figure()
plt.bar(bacteria_list, test_auc, color='blue',tick_label=bacteria_list )
plt.xticks(rotation='vertical')
plt.xlabel("Bacteria species")
plt.ylabel("AUC")
plt.title("AUC using LOBOV")
plt.show()
