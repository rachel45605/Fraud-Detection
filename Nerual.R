#https://www.r-bloggers.com/2015/09/fitting-a-neural-network-in-r-neuralnet-package/
install.packages('caret')
library(neuralnet)
library("dplyr")
library("scales")
library(data.table)
library(mltools)
library(caTools)
library(car)
library(DMwR)
library(caret)
set.seed(2022)

setwd("~/Desktop/Bc2407/Project")
fraud_data <- read.csv("fraud_clean.csv", stringsAsFactors = TRUE)
View(fraud_data)

sapply(fraud_data, class)
dim(fraud_data)

# ==============================================================================================================
### normalize your data before training a neural network ###
select_if(fraud_data, is.numeric)

################# skip as max-min function is used instead################
#fraud$WeekOfMonth <- factor(fraud$WeekOfMonth)
#fraud$WeekOfMonthClaimed <- factor(fraud$WeekOfMonthClaimed)
#fraud$DriverRating <- factor(fraud$DriverRating, levels = c("1", "2", "3", "4"))

###base on logistic regression, slect those factors that is important 
new2<-subset(fraud_data, select = c(FraudFound_P,Fault, VehicleCategory, Deductible,AgeOfVehicle,
                                   AddressChange_Claim,BasePolicy) )

#normalise using one hot encoding 
newdata2 <- one_hot(as.data.table(new2))
dim(newdata2)
sapply(newdata2, class)
View(newdata2)

#normalise using max-min(as variables are no 0-1 yet
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x)))}
newdata2 <- as.data.frame(lapply(newdata2, normalize))
View(newdata2)


# ==============================================================================================================
###create train and test set####
names(newdata2) <- make.names(names(newdata2)) #Make syntactically valid names out of character vectors
train = sample.split(Y = newdata2$FraudFound_P, SplitRatio=0.7)
trainset = subset(newdata2,train == T)
testset = subset(newdata2, train == F)
prop.table(table(trainset$FraudFound_P))
prop.table(table(testset$FraudFound_P))
#highly unbalance dataset 

#=====================================================

# Data Balancing ==========================
#follow logistic regression model 
#method 3 
trainset.bal3 <- SMOTE(FraudFound_P ~ ., trainset,k=5, perc.over=1490, perc.under=110) # error 
prop.table(table(trainset.bal3$FraudFound_P))

#Method 1
majority <- trainset[which(trainset$FraudFound_P == 0), ]
minority <- trainset[which(trainset$FraudFound_P == 1), ]
chosen <- sample(seq(1:nrow(majority)), size = nrow(minority))
majority.chosen <- majority[chosen,]
trainset.bal <- rbind(majority.chosen, minority)
prop.table(table(trainset.bal$FraudFound_P))
dim(trainset.bal)
#=============================================================================================================
# Neural Network comprising 2 hidden layer with 2/3 of the input size as hidden nodes for binary categorical target
m1 <- neuralnet(FraudFound_P ~ . , data=trainset.bal, hidden=c(2,2), err.fct="ce", linear.output=FALSE)
par(mfrow=c(1,1))
plot(m1)


m1$net.result  # predicted outputs. 
m1$result.matrix  # summary. Error = 0.017634157
m1$startweights
m1$weights
# The generalized weight is defined as the contribution of the ith input variable to the log-odds:
m1$generalized.weights
## Easier to view GW as plots instead


out <- as.data.frame(cbind(m1$covariate, m1$net.result[[1]]))
threshold <- 0.5
y.hat <- as.factor(ifelse(out$V12 > threshold, 1, 0))
y.hat
results1 <- data.frame(actual = trainset.bal$FraudFound_P, prediction = y.hat)
results1
cf_matrix1 <- confusionMatrix(data=(y.hat), reference = as.factor(trainset.bal$FraudFound_P))
cf_matrix1

#######Heat map representation ##########
#install.packages("yardstick")
library(yardstick)
library(ggplot2)
truth_predicted <- data.frame(pred=as.factor(y.hat),obs=as.factor(trainset.bal$FraudFound_P))
cm <- conf_mat(truth_predicted, obs, pred)

autoplot(cm, type = "heatmap") +scale_fill_gradient(low = "grey", high = "light blue")


#predict FraudFound_P using the neural network
pr.m1 <- neuralnet::compute(m1,testset)
pr.m1_ <- pr.m1$net.result*(max(newdata2$FraudFound_P)-min(newdata2$FraudFound_P))+min(newdata2$FraudFound_P)
test.r <- (testset$FraudFound_P)*(max(newdata2$FraudFound_P)-min(newdata2$FraudFound_P))+min(newdata2$FraudFound_P)
MSE.nn <- sum((test.r - pr.m1_)^2)/nrow(testset)
MSE.nn

#out<-cbind(m1$covariate,pr.m1$net.result[[1]])
#pr.m2 = ifelse(out$"FraudFound_P">0.5,1,0)
pr.m1 <- ifelse(pr.m1$net.result>0.5,1,0)

results <- data.frame(actual = testset$FraudFound_P, prediction = pr.m1)
results  

cf_matrix2 <- confusionMatrix(data=as.factor(pr.m1), reference = as.factor(testset$FraudFound_P))
cf_matrix2

#######Heat map representation ##########
#install.packages("yardstick")
truth_predicted <- data.frame(pred=as.factor(pr.m1),obs=as.factor(testset$FraudFound_P))
cm <- conf_mat(truth_predicted, obs, pred)

autoplot(cm, type = "heatmap") +scale_fill_gradient(low = "grey", high = "light blue")


#print  table to show performance, predictions made by the neural network 
table <- data.frame('Trainset Accuracy' = 1,'Trainset Sensitivity' = 1, 'Testset Accuracy' = 1, 'Testset Sensitivity' = 1)

acc3.m1.test = cf_matrix2$overall[1]
sens3.m1.test = cf_matrix2$byClass[1]
acc3.m1.train = cf_matrix1$overall[1]
sens3.m1.train = cf_matrix1$byClass[1]

table[1,1]= acc3.m1.train
table[1,2]= sens3.m1.train
table[1,3]=acc3.m1.test
table[1,4]=sens3.m1.test
View(table)

