---
title: "Coursera_PML_Assessment"
author: "Himanshu Raj"
date: "4 April 2018"
output: 
  html_document:
    keep_md: true
---



## Practical Machine Learning Course Project Report

### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Data

The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

### Expected Results

The goal of your project is to predict the manner in which they did the exercise


### Installing Packages and Loading Necessary Libraries

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.4.4
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.4.4
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.4.4
```

```r
library(rpart)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.4.4
```

```r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 3.4.2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```r
library(RColorBrewer)
```


### Getting Data

```r
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "pml-training.csv"
testFile  <- "pml-testing.csv"
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile = trainFile)
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile = testFile)
}
rm(trainUrl)
rm(testUrl)
```

### Reading the Data


```r
train <- read.csv(trainFile)
test <- read.csv(testFile)
dim(train)
```

```
## [1] 19622   160
```

```r
summary(train$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
rm(trainFile)
rm(testFile)
```

### Cleaning the Data

Step1:  clean the Near Zero Variance Variables from the train data:
For more info on the package and process please visit below link: https://www.rdocumentation.org/packages/caret/versions/6.0-79/topics/nearZeroVar


```r
nearZero<- nearZeroVar(train, saveMetrics = TRUE)
head(nearZero, 20)
```

```
##                        freqRatio percentUnique zeroVar   nzv
## X                       1.000000  100.00000000   FALSE FALSE
## user_name               1.100679    0.03057792   FALSE FALSE
## raw_timestamp_part_1    1.000000    4.26562022   FALSE FALSE
## raw_timestamp_part_2    1.000000   85.53154622   FALSE FALSE
## cvtd_timestamp          1.000668    0.10192641   FALSE FALSE
## new_window             47.330049    0.01019264   FALSE  TRUE
## num_window              1.000000    4.37264295   FALSE FALSE
## roll_belt               1.101904    6.77810621   FALSE FALSE
## pitch_belt              1.036082    9.37722964   FALSE FALSE
## yaw_belt                1.058480    9.97349913   FALSE FALSE
## total_accel_belt        1.063160    0.14779329   FALSE FALSE
## kurtosis_roll_belt   1921.600000    2.02323922   FALSE  TRUE
## kurtosis_picth_belt   600.500000    1.61553358   FALSE  TRUE
## kurtosis_yaw_belt      47.330049    0.01019264   FALSE  TRUE
## skewness_roll_belt   2135.111111    2.01304658   FALSE  TRUE
## skewness_roll_belt.1  600.500000    1.72255631   FALSE  TRUE
## skewness_yaw_belt      47.330049    0.01019264   FALSE  TRUE
## max_roll_belt           1.000000    0.99378249   FALSE FALSE
## max_picth_belt          1.538462    0.11211905   FALSE FALSE
## max_yaw_belt          640.533333    0.34654979   FALSE  TRUE
```

```r
train_new <- train[, !nearZero$nzv]
test_new <- test[, !nearZero$nzv]
dim(train_new)
```

```
## [1] 19622   100
```

```r
dim(test_new)
```

```
## [1]  20 100
```

```r
rm(train)
rm(test)
rm(nearZero)
```

### Cleaning the unwated predictors from the remaining set of predictors like timestamp and username


```r
regex <- grepl("^X|timestamp|user_name", names(train_new))
training <- train_new[, !regex]
testing <- test_new[, !regex]
rm(regex)
rm(train_new)
rm(test_new)
dim(training)
```

```
## [1] 19622    95
```

```r
dim(testing)
```

```
## [1] 20 95
```

### Removing the NA's Column


```r
cond <- (colSums(is.na(training)) == 0)
training <- training[, cond]
testing <- testing[, cond]
rm(cond)
```

### Finding the correlation matrix


```r
corrplot(cor(training[, -length(names(training))]), method = "color", tl.cex = 0.5)
```

![](PML_Asses_files/figure-html/correlationData-1.png)<!-- -->

### Dividing the Data into training and test sets
Spliting the cleaned training set into a pure training data set (80%) to train the machine and a validation data set (20%) to conduct cross validation.


```r
set.seed(12345) # For reproducibile purpose
inTrain <- createDataPartition(training$classe, p = 0.80, list = FALSE)
validation <- training[-inTrain, ]
training <- training[inTrain, ]
rm(inTrain)
```

## Data Modelling 

### Decision Tree
Using Decision Tree algorithm to fit a model.


```r
modelTree <- rpart(classe ~ ., data = training, method = "class")
prp(modelTree)
```

![](PML_Asses_files/figure-html/desicionModel-1.png)<!-- -->

Now, we estimate the performance of the model on the validation data set.


```r
predictTree <- predict(modelTree, validation, type = "class")
confusionMatrix(validation$classe, predictTree)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1012   26   14   45   19
##          B  173  421   41   97   27
##          C   24   60  544   40   16
##          D   79   27   99  419   19
##          E   44   70   60   91  456
## 
## Overall Statistics
##                                           
##                Accuracy : 0.727           
##                  95% CI : (0.7128, 0.7409)
##     No Information Rate : 0.3395          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6526          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7598   0.6970   0.7177   0.6055   0.8492
## Specificity            0.9599   0.8982   0.9558   0.9307   0.9217
## Pos Pred Value         0.9068   0.5547   0.7953   0.6516   0.6325
## Neg Pred Value         0.8860   0.9422   0.9339   0.9168   0.9747
## Prevalence             0.3395   0.1540   0.1932   0.1764   0.1369
## Detection Rate         0.2580   0.1073   0.1387   0.1068   0.1162
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8598   0.7976   0.8367   0.7681   0.8854
```

```r
accuracy <- postResample(predictTree, validation$classe)
ose <- 1 - as.numeric(confusionMatrix(validation$classe, predictTree)$overall[1])
rm(predictTree)
rm(modelTree)
```

The Estimated Accuracy of the rPart Model is 72.6994647% and the Estimated Out-of-Sample Error is 72.6994647%.

### Random Forest


```r
modelRF <- train(classe ~ ., data = training, method = "rf", trControl = trainControl(method = "cv", 5), ntree = 250)
modelRF
```

```
## Random Forest 
## 
## 15699 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 12560, 12559, 12559, 12558, 12560 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9945218  0.9930704
##   27    0.9968787  0.9960517
##   53    0.9951588  0.9938759
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

Now, estimating the performance of the model on the validation data set.


```r
predictRF <- predict(modelRF, validation)
confusionMatrix(validation$classe, predictRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    2  757    0    0    0
##          C    0    4  680    0    0
##          D    0    0    2  641    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                          
##                Accuracy : 0.998          
##                  95% CI : (0.996, 0.9991)
##     No Information Rate : 0.285          
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9974         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9947   0.9971   1.0000   1.0000
## Specificity            1.0000   0.9994   0.9988   0.9994   1.0000
## Pos Pred Value         1.0000   0.9974   0.9942   0.9969   1.0000
## Neg Pred Value         0.9993   0.9987   0.9994   1.0000   1.0000
## Prevalence             0.2850   0.1940   0.1738   0.1634   0.1838
## Detection Rate         0.2845   0.1930   0.1733   0.1634   0.1838
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9991   0.9971   0.9979   0.9997   1.0000
```

```r
accuracy <- postResample(predictRF, validation$classe)
ose <- 1 - as.numeric(confusionMatrix(validation$classe, predictRF)$overall[1])
rm(predictRF)
```

The Estimated Accuracy of the Random Forest Model is 99.7960744% and the Estimated Out-of-Sample Error is 0.2039256%.
Random Forests yielded better Results

### Predicting The Manner of Exercise for Test Data Set


```r
rm(accuracy)
rm(ose)
predict(modelRF, testing[, -length(names(testing))])
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Remove the objects


```r
rm(modelRF)
rm(training)
rm(testing)
rm(validation)
```
