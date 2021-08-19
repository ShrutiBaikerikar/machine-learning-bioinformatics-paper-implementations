library(randomForest)
library(varSelRF)
library(caret)

#reading data
data = read.csv(file="C:/Users/Shruti/Documents/RF_CancerBiomarkers/for github/GSE9476/GSE9476_RFData.csv",header=TRUE)
head(data)[1:5]
rownames(data)[1:5]
colnames(data)[1:5]

#processing data for training
training_data = data[,-1]
head(training_data)[1:5]
training_data$Disease_state = as.factor(training_data$Disease_state)
training_data$Disease_state

#Applying Random Forest classifier to the data
set.seed(1)
rf_classifier = randomForest(x=training_data[,1:2000], y = training_data$Disease_state, mtry=14, importance=TRUE)
rf_classifier

#Extracting the observed and predicted classes from training data
observed.classes = training_data$Disease_state
str(observed.classes)

predicted.classes = rf_classifier$predicted
str(predicted.classes)
predicted.classes = unname(predicted.classes)

#Creating confusion matrix for evaluation metrics
confusionMatrix(predicted.classes,observed.classes)
confusionMatrix(predicted.classes,observed.classes,mode = "prec_recall")

#Calculating Variable Importance using varselRF
set.seed(20)
k= varSelRF(xdata=training_data[1:2000],Class=training_data$Disease_state,c.sd=0)
k$selected.vars
