---
title: "Course Project Prediction"
author: "Obed Garcia"
date: "16/7/2020"
output: 
  html_document: 
    keep_md: yes
---
# Project goal

In this project, the data from the accelerometer measurements of people who participated in different kinds of physical activity are used, with the aim of predicting the way in which 6 participants performed an exercise in an adequate or inadequate way, which is described by the variable "classe" in the training set, which contains 5 levels, namely A, B, C, D and E which represent the way the participant performs physical activity, and are the result of the instructions he received each participant to perform the activity exactly according to Class A specifications (properly), or in a way that replicates 4 common weight-lifting mistakes (improperly), including: pulling the elbows forward (Class B), raising the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and pulling the hips forward (Class E).

# General Summary

To predict the way in which the 6 participants carried out the exercise, the first thing to do is load and subsequently clean the data, removing unnecessary variables for prediction and variables that have observations with invalid values, after the valid data set, with the necessary variables, we proceed to create the training and test set, and then create 3 prediction models, with the training data set and for each model, the prediction is made with the set of test data, in order to later, be able to choose the model that has the highest accuracy, and thus finally be able to test the machine learning algorithm, corresponding to the chosen model, in 20 test cases different from the data used for the creation of the models.

# Loading and Cleaning the data

In this part of the project, the data is loaded into a Data frame, whose name is data_train, and the necessary libraries are also loaded to carry out the project, subsequently the data is cleaned by removing the variables that have observations with invalid values ​​( Variables close to zero (NZV) and variables that contain NA values) and unnecessary variables for prediction (such as columns 1 to 5, which are identification variables).


```r
## LOADING LIBRARIES
library(caret);library(rpart)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
## LOADING THE DATA
url_train <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
data_train <- read.csv(url_train, header = T, na.strings = c("NA",""))

## CLEANING THE DATA
## Checking for values close to zero
nzv_var <- nearZeroVar(data_train)
data_train <- data_train[ , -nzv_var]
## Deleting NA values
data_train <- data_train[, colSums(is.na(data_train)) == 0]
## Removing ID variables
data_train <- data_train[ , -(1:5)]
```

Next, I show the dimension and variables of the data frame (which was left after cleaning the abovementioned data) of the training set, with which we will work.


```r
dim(data_train)
```

```
## [1] 19622    54
```

```r
names(data_train)
```

```
##  [1] "num_window"           "roll_belt"            "pitch_belt"          
##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [52] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

# Model building

In this part of the project, first, I build the training and test data set, then cross-validate, then build 3 different models (Random Forest Model, Generalized Enhanced Model (GBM), and Tree Model decision) and with each model, made the prediction with the test data set, which will allow us to see the accuracy of each model, and thus later be able to choose the best model, which is precisely the one with the highest accuracy.

### Creation of training and test sets


```r
in_train  <- createDataPartition(data_train$classe, p=0.75, list=FALSE)
train_set <- data_train[ in_train, ]
test_set  <- data_train[-in_train, ]
```

### Cross validation

Cross validation is performed, for each model, through the trainControl function using the CV method with K = 3, and is set in the control variable, as can be seen below.


```r
control <- trainControl(method = "cv", number = 3)
```

### Model creation


```r
#Random forest model
fit_RF <- train(classe ~ ., data = train_set, trControl = control, method = 'rf', ntree=100)
predict_RF <- predict(fit_RF, newdata = test_set)
cm_RF <- confusionMatrix(predict_RF, as.factor(test_set$classe))

#Generalized Enhanced Model (GBM)
fit_GBM  <- train(classe ~ ., data = train_set, trControl = control,  method = "gbm")
predict_GBM <- predict(fit_GBM, newdata = test_set)
cm_GBM <- confusionMatrix(predict_GBM, as.factor(test_set$classe))

#Decision tree model
fit_decision_tree <- rpart(classe ~ ., data = train_set, method="class")
predict_decision_tree <- predict(fit_decision_tree, newdata = test_set, type="class")
cm_DT <- confusionMatrix(predict_decision_tree, as.factor(test_set$classe))
```

# Choice of the best model

Below I show a table where the accuracy of each model appears, obtained from the different confusion matrices created, previously for each model.


```r
Model <- c("Random Forest", "Decision tree", "Generalized Boosted")
Accuracy <- rbind(round(cm_RF$overall[1],5), round(cm_DT$overall[1],5), round(cm_GBM$overall[1],5))
Accuracy_Results <- cbind(Model, Accuracy)
Accuracy_Results
```

```
##      Model                 Accuracy 
## [1,] "Random Forest"       "0.99816"
## [2,] "Decision tree"       "0.71941"
## [3,] "Generalized Boosted" "0.98878"
```

In the previous table it can be seen that the highest accuracy obtained is 0.99898 and is the one corresponding to the random forest model, and therefore this is the model chosen to make the predictions. Next, I show the confusion matrix corresponding to this model.


```r
cm_RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    1    0    0    0
##          B    0  946    1    0    0
##          C    0    2  854    2    0
##          D    0    0    0  802    2
##          E    1    0    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9982          
##                  95% CI : (0.9965, 0.9992)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9977          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9968   0.9988   0.9975   0.9978
## Specificity            0.9997   0.9997   0.9990   0.9995   0.9998
## Pos Pred Value         0.9993   0.9989   0.9953   0.9975   0.9989
## Neg Pred Value         0.9997   0.9992   0.9998   0.9995   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1929   0.1741   0.1635   0.1833
## Detection Prevalence   0.2845   0.1931   0.1750   0.1639   0.1835
## Balanced Accuracy      0.9995   0.9983   0.9989   0.9985   0.9988
```

**Note:** Since the accuracy of the random forest model is quite high, I don't build a model that is the union of the three models created above.

### Expected sample error

Since the accuracy corresponding to the previously chosen model is 0.99898, I think the expected error of the sample is 0.00102 or equivalently 0.102%.

# Prediction of the 20 different test cases

In this last part of the project, with the 20 test cases different from the data used to create the models, I test the previously chosen model and show the prediction results, as shown below.


```r
url_quiz  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
data_quiz <- read.csv(url_quiz,  header = T, na.strings = c("NA",""))
prediction <- predict(fit_RF, newdata = data_quiz)
predict_quiz <- data.frame(Case_number = data_quiz$problem_id, Prediction = prediction)
predict_quiz
```

```
##    Case_number Prediction
## 1            1          B
## 2            2          A
## 3            3          B
## 4            4          A
## 5            5          A
## 6            6          E
## 7            7          D
## 8            8          B
## 9            9          A
## 10          10          A
## 11          11          B
## 12          12          C
## 13          13          B
## 14          14          A
## 15          15          E
## 16          16          E
## 17          17          A
## 18          18          B
## 19          19          B
## 20          20          B
```




