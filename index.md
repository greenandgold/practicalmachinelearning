# Classifying Physical Exercise with Machine Learning
Katherine Crowley  

## Executive Summary
We use random forests to build a model on exercise data with an out-of-sample estimated error of 0.1%. We use this model to classify the manner in which the participants in the test set performed the exercises. 

## The Data
The data for this report comes from the webiste http://groupware.les.inf.puc-rio.br/har. There are two datasets provided, a testing and a training set, each of which includes data from accelerometers on the belt, forearm, arm, and dumbbell of six participants. Some of the data is the result of exercises done correctly; other data is the result of exercises done incorrectly. The goal is to use machine learning on the training data to determine how the exercises were done in the testing set. 

We begin by reading the training and testing files provided.


```r
training <- read.csv("pml-training.csv", na.strings = c("#DIV/0!", "NA", ""))
testing <- read.csv("pml-testing.csv", na.strings = c("#DIV/0!", "NA", ""))
```

The training and testing files have the same column names, but in a few cases, the classes of corresponding columns do not match. We correct this so that it doesn't cause problems when applying a model trained on the training set to data in the testing set.


```r
testing$magnet_dumbbell_z <- as.numeric(testing$magnet_dumbbell_z) 
testing$magnet_forearm_y <- as.numeric(testing$magnet_forearm_y)
testing$magnet_forearm_z <- as.numeric(testing$magnet_forearm_z)
```

## Clean the Data

We clean the data by eliminating unhelpful columns of the training and testing sets.

In the training set, the number of NAs in each column is either zero or over 19,000 (out of 19,622 rows). In the testing set, the number of NAs in each column is either zero or 20 (out of 20 rows), as shown here. 


```r
nrow(training)
```

```
## [1] 19622
```

```r
unique(sapply(training, function(x) sum(is.na(x))))
```

```
##  [1]     0 19226 19248 19622 19225 19216 19294 19296 19227 19293 19221
## [12] 19218 19220 19217 19300 19301 19299
```

```r
nrow(testing)
```

```
## [1] 20
```

```r
unique(sapply(testing, function(x) sum(is.na(x))))
```

```
## [1]  0 20
```

Since columns that have NAs are entirely or almost entirely NAs, they will not be useful for building a model, and we'd like to remove them. To maintain consistency between our training and testing sets, we first verify that the training columns with zero NAs match the testing columns with zero NAs. 


```r
training_col_NAs <- as.vector(colSums(is.na(training)))
testing_col_NAs <- as.vector(colSums(is.na(testing)))
identical(testing_col_NAs[testing_col_NAs == 0], training_col_NAs[training_col_NAs == 0])
```

```
## [1] TRUE
```

We remove the columns with NAs. 


```r
good_columns <- training_col_NAs == 0
training <- training[, good_columns]
testing <- testing[, good_columns]
```

Of the remaining 60 columns, 1 through 7 provide user and time information, rather than exercise data. So we remove these as well. 



```r
training <- training[,8:60]
testing <- testing[,8:60]
```

Now we have training and testing sets that contain only exercise data and no NAs.

## Cross-Validation

After some preliminary experimentation, 5-fold cross-validation will be sufficient to achieve a favorable out-of-sample error estimate. We load required libraries, set the seed so that results are reproducible, and create 5 folds on the training set.


```r
library(caret); library(randomForest)
set.seed(386)
folds <- createFolds(training$classe, k = 5, list = TRUE, returnTrain = TRUE)
```

Each of the five folds is approximately 4/5 of the original training set, and will serve as a smaller training set for cross-validation. The remaining 1/5 of the original training set in each case will serve as the corresponding testing set. We name each of these smaller training sets


```r
fold1_train <- training[folds[[1]],]
fold2_train <- training[folds[[2]],]
fold3_train <- training[folds[[3]],]
fold4_train <- training[folds[[4]],]
fold5_train <- training[folds[[5]],]
```

as well as the corresponding testing sets.


```r
fold1_test <- training[-folds[[1]],]
fold2_test <- training[-folds[[2]],]
fold3_test <- training[-folds[[3]],]
fold4_test <- training[-folds[[4]],]
fold5_test <- training[-folds[[5]],]
```

## Build the Model

We have seen in lectures and homework that random forests are one of the most accurate machine learning models. So we will start with random forests and see whether they suffice here. In order to correctly classify the manner in which subjects did their exercises in all 20 cases in the testing set, we are looking for accuracy of greater than 99%.

We build random forest models on the new training sets in the five folds.


```r
rf1 <- randomForest(classe ~ ., data = fold1_train)
rf2 <- randomForest(classe ~ ., data = fold2_train)
rf3 <- randomForest(classe ~ ., data = fold3_train)
rf4 <- randomForest(classe ~ ., data = fold4_train)
rf5 <- randomForest(classe ~ ., data = fold5_train)
```

Then we use those models to predict the classe on the corresponding testing sets in the five folds.


```r
prediction1 <- predict(rf1, fold1_test)
prediction2 <- predict(rf1, fold2_test)
prediction3 <- predict(rf1, fold3_test)
prediction4 <- predict(rf1, fold4_test)
prediction5 <- predict(rf1, fold5_test)
```

## Out-of-Sample Error

To see how well the models did, we find the confusion matrices


```r
cM1 <- confusionMatrix(prediction1, fold1_test$classe)
cM2 <- confusionMatrix(prediction2, fold2_test$classe)
cM3 <- confusionMatrix(prediction3, fold3_test$classe)
cM4 <- confusionMatrix(prediction4, fold4_test$classe)
cM5 <- confusionMatrix(prediction5, fold5_test$classe)
```

and calculate the accuracy in each case.


```r
accuracy1 <- cM1$overall["Accuracy"]
accuracy2 <- cM2$overall["Accuracy"]
accuracy3 <- cM3$overall["Accuracy"]
accuracy4 <- cM4$overall["Accuracy"]
accuracy5 <- cM5$overall["Accuracy"]
c(accuracy1, accuracy2, accuracy3, accuracy4, accuracy5)
```

```
##  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
## 0.9936273 1.0000000 1.0000000 1.0000000 1.0000000
```

We subtract accuracy from 1 to get the error rate. Our estimated out-of-sample error is the average of the five errors.


```r
mean(c(1-accuracy1, 1-accuracy2, 1-accuracy3, 1-accuracy4, 1-accuracy5))
```

```
## [1] 0.001274535
```

The out-of-sample error rate is estimated at 0.001, or 0.1%. Equivalently, the accuracy is estimated at 99.9%, which is excellent. Satisfied with the results of our cross-validation, we conclude that random forests are sufficient here and forego other models.


## The Final Model

We now make a single random forest on the entire training set.


```r
rf_model <- randomForest(classe ~ ., data = training)
```

## Predictions

Finally, we use our single random forest model to predict the manner in which exercises were performed in the testing set.


```r
predict(rf_model, testing)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

## Summary
Cross-validation with a random forest model suggests a low out-of-sample error rate, so we can be reasonably confident that we successfully classifed the exercises on the original testing set.  
