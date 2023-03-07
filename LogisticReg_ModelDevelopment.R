# ==============================================================================================================
#Required Libraries
#install.packages("caret")
#install.packages("rstudioapi")
library(rstudioapi)
library(remotes)
library(data.table)
library(caTools)
library(car)
library(caret)
#remotes::install_version("DMwR", version="0.4.1") #Run this For the First time Keep this here as it has to be behind remotes
library(DMwR)
library(boot)

# ==============================================================================================================
#Set Working Directory to Source File Location Automatically
#setwd(dirname(getActiveDocumentContext()$path))
setwd("~/Desktop/Bc2407/Project")

#Importing datasets
fraud <- fread("fraud_clean.csv", stringsAsFactors = TRUE)
View(fraud)


#Convert categorical variables to factor type
fraud$WeekOfMonth <- factor(fraud$WeekOfMonth)
fraud$WeekOfMonthClaimed <- factor(fraud$WeekOfMonthClaimed)
fraud$FraudFound_P <- factor(fraud$FraudFound_P)
fraud$DriverRating <- factor(fraud$DriverRating, levels = c("1", "2", "3", "4"))

levels(fraud$WeekOfMonth)
levels(fraud$WeekOfMonthClaimed)
levels(fraud$FraudFound_P) # 0 for not found and 1 for found
levels(fraud$DriverRating) # 1 to 4

# ============================================================================================================
# ============================================================================================================
# 
# Logistic Regression for Vehicle Fraud Identification
#
# ============================================================================================================
# ============================================================================================================

#Train-Test Split ==========================
set.seed(2022)
train = sample.split(Y = fraud$FraudFound_P, SplitRatio=0.7)
trainset = subset(fraud,train == T)
testset = subset(fraud, train == F)


#Check Data Skew ==========================
table(trainset$FraudFound_P)
prop.table(table(trainset$FraudFound_P))
# Highly Skewed


#=====================================================

# Data Balancing ==========================

#Different Imbalance Data Handling Methods to Ensure Highest Accuracy
#Because there is a trade-off between losing data with undersampling, and
#over emphasising meaningless data with oversampling


#Method 1: Manual Undersampling Data
majority <- trainset[FraudFound_P == 0]
minority <- trainset[FraudFound_P == 1]
chosen <- sample(seq(1:nrow(majority)), size = nrow(minority))
majority.chosen <- majority[chosen]
trainset.bal <- rbind(majority.chosen, minority)
prop.table(table(trainset.bal$FraudFound_P))


#Method 2: SMOTE Mix of Oversampling & Minor Undersampling without K-Nearest Neighbour Algorithm
trainset.bal2 <- SMOTE(FraudFound_P ~ ., trainset, perc.over=1490, perc.under=110)
prop.table(table(trainset.bal2$FraudFound_P))

#Method 3: SMOTE Mix of Oversampling & Minor Undersampling with K-Nearest Neighbour Algorithm to determine Oversample data
trainset.bal3 <- SMOTE(FraudFound_P ~ ., trainset,k=5, perc.over=1490, perc.under=110)
prop.table(table(trainset.bal3$FraudFound_P))



#================================================================================================
# LOGISTIC REGRESSION USING MANUALLY UNDERSAMPLED TRAINSET
#===================================================================================================
threshold = 0.5 #For prediction


# Model 0: Full Model ==========================
log1.full = glm(FraudFound_P~., family=binomial, data = trainset.bal)
summary(log1.full)

# Model 0 Trainset Prediction
trainprob1.full = predict(log1.full,type = 'response')
y.hat1.full = ifelse(trainprob1.full>threshold, 1, 0)
confmat1.full = confusionMatrix(data = factor(y.hat1.full), trainset.bal$FraudFound_P)
acc1.full = confmat1.full$overall[1]
sens1.full = confmat1.full$byClass[1]
vif(log1.full)

#Model 0 Testset Prediction
testprob1.full = predict(log1.full, newdata = testset,type="response")
#Error New Levels


# Model 1: Remove Significant Variables ==========================
log1.m1 = glm(FraudFound_P~ Fault +
                VehicleCategory +
                Deductible +
                AddressChange_Claim +
                BasePolicy
             , family=binomial, data = trainset.bal)
summary(log1.m1)

#Model 1 Trainset Prediction
trainprob1.m1 = predict(log1.m1,type = 'response')
y.hat1.m1 = ifelse(trainprob1.m1>threshold, 1, 0)
confmat1.m1 = confusionMatrix(data = factor(y.hat1.m1), trainset.bal$FraudFound_P)
acc1.m1 = confmat1.m1$overall[1]
sens1.m1 = confmat1.m1$byClass[1]
vif(log1.m1)

#Model 1 Testset Prediction
testprob1.m1 = predict(log1.m1, newdata = testset,type="response")
y.hat1.m1.test = ifelse(testprob1.m1>threshold, 1, 0)
confmat1.m1.test = confusionMatrix(data = factor(y.hat1.m1.test), testset$FraudFound_P)
acc1.m1.test = confmat1.m1.test$overall[1]
sens1.m1.test = confmat1.m1.test$byClass[1]


# Model 2: Remove GVIF >2:BasePolicy ==========================
log1.m2 = glm(FraudFound_P~ Fault +
                VehicleCategory +
                Deductible +
                AddressChange_Claim 
              , family=binomial, data = trainset.bal)
summary(log1.m2)

#Model 2 Trainset Prediction
trainprob1.m2 = predict(log1.m2,type = 'response')
y.hat1.m2 = ifelse(trainprob1.m2>threshold, 1, 0)
confmat1.m2 = confusionMatrix(data = factor(y.hat1.m2), trainset.bal$FraudFound_P)
acc1.m2 = confmat1.m2$overall[1]
sens1.m2 = confmat1.m2$byClass[1]
vif(log1.m2)


#Model 2 Testset Prediction
testprob1.m2 = predict(log1.m2, newdata = testset,type="response")
y.hat1.m2.test = ifelse(testprob1.m2>threshold, 1, 0)
confmat1.m2.test = confusionMatrix(data = factor(y.hat1.m2.test), testset$FraudFound_P)
acc1.m2.test = confmat1.m2.test$overall[1]
sens1.m2.test = confmat1.m2.test$byClass[1]



#================================================================================================
# LOGISTIC REGRESSION SMOTE MIX without K-nearest Algorithm
#===================================================================================================


# Model 0: Full Model ==========================
log2.full = glm(FraudFound_P~., family=binomial, data = trainset.bal2)
summary(log2.full)

#Model 0 Trainset Prediction
trainprob2.full = predict(log2.full,type = 'response')
y.hat2.full = ifelse(trainprob2.full>threshold, 1, 0)
confmat2.full = confusionMatrix(data = factor(y.hat2.full), trainset.bal2$FraudFound_P)
acc2.full = confmat2.full$overall[1]
sens2.full = confmat2.full$byClass[1]
vif(log2.full)

#Model 0 Testset Prediction
testprob2.full = predict(log2.full, newdata = testset,type="response")
y.hat2.full.test = ifelse(testprob2.full>threshold, 1, 0)
confmat2.full.test = confusionMatrix(data = factor(y.hat2.full.test), testset$FraudFound_P)
acc2.full.test = confmat2.full.test$overall[1]
sens2.full.test = confmat2.full.test$byClass[1]


# Model 1: Remove Insignificant ==========================
log2.m1 = glm(FraudFound_P~.
              -DayOfWeekClaimed
              -PastNumberOfClaims 
              , family=binomial, data = trainset.bal2)
summary(log2.m1)

#Model 1 Trainset Prediction
trainprob2.m1 = predict(log2.m1,type = 'response')
y.hat2.m1 = ifelse(trainprob2.m1>threshold, 1, 0)
confmat2.m1 = confusionMatrix(data = factor(y.hat2.m1), trainset.bal2$FraudFound_P)
acc2.m1 = confmat2.m1$overall[1]
sens2.m1 = confmat2.m1$byClass[1]
vif(log2.m1)

#Model 1 Testset Prediction
testprob2.m1 = predict(log2.m1, newdata = testset,type="response")
y.hat2.m1.test = ifelse(testprob2.m1>threshold, 1, 0)
confmat2.m1.test = confusionMatrix(data = factor(y.hat2.m1.test), testset$FraudFound_P)
acc2.m1.test = confmat2.m1.test$overall[1]
sens2.m1.test = confmat2.m1.test$byClass[1]


# Model 2: Remove Additional Insignificant ==========================
log2.m2 = glm(FraudFound_P~.
              -DayOfWeekClaimed
              -PastNumberOfClaims 
              -MonthClaimed
              , family=binomial, data = trainset.bal2)
summary(log2.m2)

#Model 2 Trainset Prediction
trainprob2.m2 = predict(log2.m2,type = 'response')
y.hat2.m2 = ifelse(trainprob2.m2>threshold, 1, 0)
confmat2.m2 = confusionMatrix(data = factor(y.hat2.m2), trainset.bal2$FraudFound_P)
acc2.m2 = confmat2.m2$overall[1]
sens2.m2 = confmat2.m2$byClass[1]
vif(log2.m2)

#Model 2 Testset Prediction
testprob2.m2 = predict(log2.m2, newdata = testset,type="response")
y.hat2.m2.test = ifelse(testprob2.m2>threshold, 1, 0)
confmat2.m2.test = confusionMatrix(data = factor(y.hat2.m2.test), testset$FraudFound_P)
acc2.m2.test = confmat2.m2.test$overall[1]
sens2.m2.test = confmat2.m2.test$byClass[1]


# Model 3: Remove Multicollinear: Age ==========================
log2.m3 = glm(FraudFound_P~.
              -DayOfWeekClaimed
              -PastNumberOfClaims 
              -MonthClaimed
              -Age
              , family=binomial, data = trainset.bal2)
summary(log2.m3)

#Model 3 Trainset Prediction
trainprob2.m3 = predict(log2.m3,type = 'response')
y.hat2.m3 = ifelse(trainprob2.m3>threshold, 1, 0)
confmat2.m3 = confusionMatrix(data = factor(y.hat2.m3), trainset.bal2$FraudFound_P)
acc2.m3 = confmat2.m3$overall[1]
sens2.m3 = confmat2.m3$byClass[1]
vif(log2.m3)

#Model 3 Testset Prediction
testprob2.m3 = predict(log2.m3, newdata = testset,type="response")
y.hat2.m3.test = ifelse(testprob2.m3>threshold, 1, 0)
confmat2.m3.test = confusionMatrix(data = factor(y.hat2.m3.test), testset$FraudFound_P)
acc2.m3.test = confmat2.m3.test$overall[1]
sens2.m3.test = confmat2.m3.test$byClass[1]

# Model 4: Remove Multicollinear: Vehicle Category ==========================
log2.m4 = glm(FraudFound_P~.
              -DayOfWeekClaimed
              -PastNumberOfClaims 
              -MonthClaimed
              -Age
              -VehicleCategory
              , family=binomial, data = trainset.bal2)
summary(log2.m4)

#Model 4 Trainset Prediction
trainprob2.m4 = predict(log2.m4,type = 'response')
y.hat2.m4 = ifelse(trainprob2.m4>threshold, 1, 0)
confmat2.m4 = confusionMatrix(data = factor(y.hat2.m4), trainset.bal2$FraudFound_P)
acc2.m4 = confmat2.m4$overall[1]
sens2.m4 = confmat2.m4$byClass[1]
vif(log2.m4)

#Model 4 Testset Prediction
testprob2.m4 = predict(log2.m4, newdata = testset,type="response")
y.hat2.m4.test = ifelse(testprob2.m4>threshold, 1, 0)
confmat2.m4.test = confusionMatrix(data = factor(y.hat2.m4.test), testset$FraudFound_P)
acc2.m4.test = confmat2.m4.test$overall[1]
sens2.m4.test = confmat2.m4.test$byClass[1]

#Overall testset accuracy higher



#================================================================================================
# LOGISTIC REGRESSION SMOTE MIX with K-nearest Algorithm = 5
#===================================================================================================


# Model 0: Full Model ==========================
log3.full = glm(FraudFound_P~., family=binomial, data = trainset.bal3)
summary(log3.full)


# Model 0 Trainset Prediction
trainprob3.full = predict(log3.full,type = 'response')
y.hat3.full = ifelse(trainprob3.full>threshold, 1, 0)
confmat3.full = confusionMatrix(data = factor(y.hat3.full), trainset.bal3$FraudFound_P)
acc3.full = confmat3.full$overall[1]
sens3.full = confmat3.full$byClass[1]
vif(log3.full)

#Model 0 Testset Prediction
testprob3.full = predict(log3.full, newdata = testset,type="response")
y.hat3.full.test = ifelse(testprob3.full>threshold, 1, 0)
confmat3.full.test = confusionMatrix(data = factor(y.hat3.full.test), testset$FraudFound_P)
acc3.full.test = confmat3.full.test$overall[1]
sens3.full.test = confmat3.full.test$byClass[1]


# Model 2: Remove Insignificant Variable ==========================
log3.m2 = glm(FraudFound_P~.
              -DayOfWeekClaimed
              -MonthClaimed
              , family=binomial, data = trainset.bal3)
summary(log3.m2)

#Model 2 Trainset Prediction 
trainprob3.m2 = predict(log3.m2,type = 'response')
y.hat3.m2 = ifelse(trainprob3.m2>threshold, 1, 0)
confmat3.m2 = confusionMatrix(data = factor(y.hat3.m2), trainset.bal3$FraudFound_P)
acc3.m2 = confmat3.m2$overall[1]
sens3.m2 = confmat3.m2$byClass[1]
vif(log3.m2)

#Model 2 Testset Prediction
testprob3.m2 = predict(log3.m2, newdata = testset,type="response")
y.hat3.m2.test = ifelse(testprob3.m2>threshold, 1, 0)
confmat3.m2.test = confusionMatrix(data = factor(y.hat3.m2.test), testset$FraudFound_P)
acc3.m2.test = confmat3.m2.test$overall[1]
sens3.m2.test = confmat3.m2.test$byClass[1]


# Model 3: Remove Multicollinear Variables GVIF > 2: Age ==========================
log3.m3 = glm(FraudFound_P~.
              -DayOfWeekClaimed
              -MonthClaimed
              -Age
              , family=binomial, data = trainset.bal3)
summary(log3.m3)

#Model 3 Trainset Prediction
trainprob3.m3 = predict(log3.m3,type = 'response')
y.hat3.m3 = ifelse(trainprob3.m3>threshold, 1, 0)
confmat3.m3 = confusionMatrix(data = factor(y.hat3.m3), trainset.bal3$FraudFound_P)
acc3.m3 = confmat3.m3$overall[1]
sens3.m3 = confmat3.m3$byClass[1]
vif(log3.m3)

#Model 3 Testset Prediction
testprob3.m3 = predict(log3.m3, newdata = testset,type="response")
y.hat3.m3.test = ifelse(testprob3.m3>threshold, 1, 0)
confmat3.m3.test = confusionMatrix(data = factor(y.hat3.m3.test), testset$FraudFound_P)
acc3.m3.test = confmat3.m3.test$overall[1]
sens3.m3.test = confmat3.m3.test$byClass[1]

# Model 4: Remove Multicollinear Variables GVIF > 2 ==========================
log3.m4 = glm(FraudFound_P~.
              -DayOfWeekClaimed
              -MonthClaimed
              -Age
              -VehicleCategory
              , family=binomial, data = trainset.bal3)
summary(log3.m4)

#Model 4 Trainset Prediction
trainprob3.m4 = predict(log3.m4,type = 'response')
y.hat3.m4 = ifelse(trainprob3.m4>threshold, 1, 0)
confmat3.m4 = confusionMatrix(data = factor(y.hat3.m4), trainset.bal3$FraudFound_P)
acc3.m4 = confmat3.m4$overall[1]
sens3.m4 = confmat3.m4$byClass[1]
vif(log3.m4)

#Model 4 Testset Prediction
testprob3.m4 = predict(log3.m4, newdata = testset,type="response")
y.hat3.m4.test = ifelse(testprob3.m4>threshold, 1, 0)
confmat3.m4.test = confusionMatrix(data = factor(y.hat3.m4.test), testset$FraudFound_P)
acc3.m4.test = confmat3.m4.test$overall[1]
sens3.m4.test = confmat3.m4.test$byClass[1]



#================================================================================================
# Prediction Accuracy and Sensitivity Testing
#===================================================================================================
#DataFrame to Compare Logistic Regression Models
table1 <- data.frame('Trainset Accuracy' = 1:9,'Trainset Sensitivity' = 1:9, 'Testset Accuracy' = 1:9, 'Testset Sensitivity' = 1:9)

# Convert all column to string as CI is a confidence interval and not one number.
table1 <- data.frame(lapply(table1, as.character))
rownames(table1) <- c('Undersampling-Full Model',
                      'Undersampling-Insig. Var. Removed',
                      'Undersampling-Multicollinear/Add. Insig. Var. Removed',
                      'SMOTE-Full Model',
                      'SMOTE-Insig. Var. Removed',
                      'SMOTE-Multicollinear/Add. Insig. Var. Removed',
                      'SMOTE with K-Full Model',
                      'SMOTE with K-Insig. Var. Removed',
                      'SMOTE with K-Multicollinear/Add. Insig. Var. Removed')
table1$Trainset.Accuracy <- NA
table1$Trainset.Sensitivity <- NA
table1$Testset.Accuracy <- NA
table1$Testset.Sensitivity <- NA

#Insert Values
table1[1,1]= acc1.full
table1[1,2]= sens1.full
table1[1,3]= NA
table1[1,4]= NA

table1[2,1]= acc1.m1
table1[2,2]= sens1.m1
table1[2,3]= acc1.m1.test
table1[2,4]= sens1.m1.test

table1[3,1]= acc1.m2
table1[3,2]= sens1.m2
table1[3,3]= acc1.m2.test
table1[3,4]= sens1.m2.test



table1[4,1]= acc2.full
table1[4,2]= sens2.full
table1[4,3]= acc2.full.test
table1[4,4]= sens2.full.test

table1[5,1]= acc2.m2
table1[5,2]= sens2.m2
table1[5,3]= acc2.m2.test
table1[5,4]= sens2.m2.test

table1[6,1]= acc2.m4
table1[6,2]= sens2.m4
table1[6,3]= acc2.m4.test
table1[6,4]= sens2.m4.test



table1[7,1]= acc3.full
table1[7,2]= sens3.full
table1[7,3]= acc3.full.test
table1[7,4]= sens3.full.test

table1[8,1]= acc3.m2
table1[8,2]= sens3.m2
table1[8,3]= acc3.m2.test
table1[8,4]= sens3.m2.test

table1[9,1]= acc3.m4
table1[9,2]= sens3.m4
table1[9,3]= acc3.m4.test
table1[9,4]= sens3.m4.test



#================================================================================================
# BOOTSTRAPPING LOGISTIC REGRESSION
#===================================================================================================


# sample.beta = function(formula, data, indices) {
#   bs = data[indices,]
#   return(glm(formula,data=bs, family='binomial'))
# }
# 
# boot.reg <- boot(data=trainset, statistic = sample.beta, R =10000, formula = FraudFound_P~.)
# boot.reg

