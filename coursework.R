require(caret)
#load the training data and testing data
training_orig=read.csv("pml-training.csv", na.strings= c("NA","","#DIV/0!"))
testing_orig=read.csv("pml-testing.csv", na.strings= c("NA","","#DIV/0!"))
#data cleaning
#remove columns with almost all NAs
training_mod=training_orig[,!colSums(is.na(training_orig))>dim(training_orig)[1]*.95]
#leave only quantative variables
training_mod=training_mod[, !(names(training_mod) %in% c("X","user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window"))]
#predict using random forrest
require(randomForest)
require(doParallel)
require(parallel)
require(foreach)

set.seed(3234)
#use 75% of the spam data as training set
training_mod$classe = as.factor(training_mod$classe)
inTrain=createDataPartition(training_mod$classe, p=0.75, list=FALSE)
training=training_mod[inTrain,]
testing=training_mod[-inTrain,]
dim(training)
# cl <- makePSOCKcluster(3)
# clusterEvalQ(cl, library(foreach))
# registerDoParallel(cl)
#ctrl=trainControl(method = "cv",number=5,repeats=5)
model <- train(classe~., data=training, method="parRF", trControl=trainControl(method = "cv", number = 5),ntree=100, do.trace=TRUE)
predictions <- predict(model,newdata=testing)
confusionMatrix(predictions, testing$classe)
predictions <- predict(model, newdata=testing_orig)
print(predictions)
model$finalModel
varImpObj <- varImp(model)
# Top 40 plot
plot(varImpObj, main = "Importance of Top 40 Variables", top = 40)

install.packages("ROCR")
library(randomForest)
library(ROCR)

