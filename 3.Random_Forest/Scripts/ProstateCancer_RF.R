#Random Forest on Prostate Cancer Dataset (GSE71783)
library(curl)
library(randomForest)
library(varSelRF)
library(caret)

#reading data
url = "https://raw.githubusercontent.com/ShrutiBaikerikar/machine-learning-bioinformatics-paper-implementations/main/3.Random_Forest/Datasets/ProstateCancer_RFData.csv"
curl_download(url,'data.csv')
data = read.csv('data.csv')
head(data)[1:5]
rownames(data)[1:5]
colnames(data)[1:5]

#processing data for training
training_data = data[,-1]
head(training_data)[1:5]
training_data$Disease_state = as.factor(training_data$Disease_state)
training_data$Disease_state

#Applying Random Forest classifier to the data
set.seed(19)
rf_classifier = randomForest(x=training_data[,1:2000], y = training_data$Disease_state, importance=TRUE)
rf_classifier

#Extracting the observed and predicted classes from training data
observed.classes = training_data$Disease_state
str(observed.classes)

predicted.classes = rf_classifier$predicted
str(predicted.classes)
predicted.classes = unname(predicted.classes)

#Creating confusion matrix for evaluation metrics
confusionMatrix(predicted.classes,observed.classes,positive ="Tumor" )
confusionMatrix(predicted.classes,observed.classes,positive="Tumor",mode = "prec_recall")

#Calculating Variable Importance using varselRF
set.seed(4)
k= varSelRF(xdata=training_data[1:2000],Class=training_data$Disease_state,c.sd=0)
k$selected.vars
