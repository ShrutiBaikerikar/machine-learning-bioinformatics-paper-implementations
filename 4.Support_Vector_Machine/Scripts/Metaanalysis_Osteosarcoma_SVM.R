#Support Vector Machine Classifier on Merged Dataset obtained from meta-analysis of Osteosarcoma Gene Expression Datasets
library(curl)
library(caret)
library(e1071)

#reading data
url = "https://raw.githubusercontent.com/ShrutiBaikerikar/machine-learning-bioinformatics-paper-implementations/main/4.Support_Vector_Machine/Datasets/Metaanalysis_dataset.csv"
curl_download(url,'data.csv')
data = read.csv('data.csv')
head(data)[1:5]
rownames(data)[1:5]
colnames(data)[1:5]

#processing data for training
data = data[,-1]
head(data)[1:5]
data$Disease_State = as.factor(data$Disease_State)
data$Disease_State


#splitting data into train and test set
set.seed(1000)  
trainIndex = createDataPartition(y=data$Disease_State,p=0.70,list=FALSE)
train_data = data[trainIndex,]
test_data = data[-trainIndex,]

#scaling and centering
train_data = scale_data_frame(train_data)
test_data = scale_data_frame(test_data)

#Training SVM classifier model
set.seed(21)
svm_model = svm(Disease_State~ ., data=train_data,type='C',kernel='radial',cost=3,gamma=0.05)
summary(svm_model)

#Predicting on Train data
pred_train <- predict(svm_model,x = train_data[,1:100])

#getting observed and predicted classes
observed_classes_train = train_data$Disease_State
str(observed_classes_train)
predicted_classes_train = pred_train
str(predicted_classes_train)
predicted_classes_train = unname(predicted_classes_train)

#Creating confusion matrix for evaluation metrics for train data
confusionMatrix(predicted_classes_train,observed_classes_train,positive ="Metastasis" )


#Predicting on Test data
pred_test <- predict(svm_model,newdata = test_data[,1:100])

#getting observed and predicted classes
observed_classes_test = test_data$Disease_State
str(observed_classes_test)
predicted_classes_test = pred_test
str(predicted_classes_test)
predicted_classes_test = unname(predicted_classes_test)

#Creating confusion matrix for evaluation metrics for test data
confusionMatrix(predicted_classes_test,observed_classes_test,positive ="Metastasis" )



