#Data: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

library(dplyr)
library(tidyverse)
library(readxl)
library(VIM)
library(caret)
library(dummy)
library(ggplot)
library(modelr)
library(glmnet)

#Data Import
setwd("C:/Users/tony/Desktop/New folder/1")
train <- read_csv("train.csv")
test <- read_csv("test.csv")

#Adding dependant variable for row binding
test$SalePrice <-  0

#Removing outliers in training
train <- train %>% filter(GrLivArea < 4000 & `1stFlrSF` < 4000 & LotFrontage < 300 & LotArea < 100000 & BsmtFinSF1 < 5000 & BsmtFinSF2 < 1400 & TotalBsmtSF < 6000)

#Binding training and test
my_data <- rbind(train, test)


#Removing NA's based on Author's data dictionary
my_data$Alley[is.na(my_data$Alley)] <- "None"
my_data$MasVnrType[is.na(my_data$MasVnrType)] <- "None"
my_data$BsmtCond[is.na(my_data$BsmtCond)] <- "None"
my_data$BsmtExposure[is.na(my_data$BsmtExposure)] <- "None"
my_data$BsmtFinType1[is.na(my_data$BsmtFinType1)] <- "None"
my_data$BsmtFinType2[is.na(my_data$BsmtFinType2)] <- "None"
my_data$BsmtQual[is.na(my_data$BsmtQual)] <- "None"
my_data$FireplaceQu[is.na(my_data$FireplaceQu)] <- "None"
my_data$GarageType[is.na(my_data$GarageType)] <- "None"
my_data$GarageFinish[is.na(my_data$GarageFinish)] <- "None"
my_data$GarageQual[is.na(my_data$GarageQual)] <- "None"
my_data$GarageCond[is.na(my_data$GarageCond)] <- "None"
my_data$PoolQC[is.na(my_data$PoolQC)] <- "None"
my_data$Fence[is.na(my_data$Fence)] <- "None"
my_data$MiscFeature[is.na(my_data$MiscFeature)] <- "None" 
my_data$Electrical[is.na(my_data$Electrical)] <- "SBrkr"
my_data$MSZoning[is.na(my_data$MSZoning)] <- "RL"
my_data$LotFrontage[is.na(my_data$LotFrontage)] <- 69
my_data$Exterior1st[is.na(my_data$Exterior1st)] <- "VinylSd"
my_data$Exterior2nd[is.na(my_data$Exterior2nd)] <- "VinylSd"
my_data$MasVnrArea[is.na(my_data$MasVnrArea)] <- 102
my_data$BsmtFinSF1[is.na(my_data$BsmtFinSF1)] <- 441
my_data$BsmtFinSF2[is.na(my_data$BsmtFinSF2)] <- 50
my_data$BsmtUnfSF[is.na(my_data$BsmtUnfSF)] <- 561
my_data$BsmtFullBath[is.na(my_data$BsmtFullBath)] <- 0
my_data$BsmtHalfBath[is.na(my_data$BsmtHalfBath)] <- 0
my_data$Functional[is.na(my_data$Functional)] <- "Typ"
my_data$GarageYrBlt[is.na(my_data$GarageYrBlt)] <- 1978
my_data$GarageCars[is.na(my_data$GarageCars)] <- 2
my_data$GarageArea[is.na(my_data$GarageArea)] <- 473
my_data$SaleType[is.na(my_data$SaleType)] <- "WD"
my_data$TotalBsmtSF[is.na(my_data$TotalBsmtSF)] <- 1052
my_data$Utilities[is.na(my_data$Utilities)] <- "AllPub"
my_data$KitchenQual[is.na(my_data$KitchenQual)] <- "TA"


#Converting data type for data that should be categorical
my_data[sapply(my_data, is.character)] <- lapply(my_data[sapply(my_data, is.character)], as.factor)
my_data$MSSubClass <- as.factor(my_data$MSSubClass)
my_data$OverallCond <= as.factor(my_data$OverallCond)
my_data$KitchenAbvGr <- as.factor(my_data$KitchenAbvGr)
my_data$MoSold <- as.factor(my_data$MoSold)


#Feature Engineering
my_data$YearRemodAdded <- my_data$YearRemodAdd - my_data$YearBuilt
my_data$HouseAge <- my_data$YrSold - my_data$YearBuilt
my_data$TotalHouseSF <- (my_data$`1stFlrSF` + my_data$`2ndFlrSF` + my_data$TotalBsmtSF)
my_data$PorchSF <- my_data$OpenPorchSF + my_data$`3SsnPorch` + my_data$ScreenPorch + my_data$EnclosedPorch
my_data$GarageAge <- my_data$YrSold - my_data$GarageYrBlt

#Error in Data fix
my_data$GarageYrBlt[my_data$GarageYrBlt == 2207] <- 2007

#Checking distribution of dependant variable, notice that it is skewed, therefore will be taking the log of it for normalization
hist(log(train$SalePrice))

#Removing columnns that are no longer needed due to the features we added
my_data <- my_data[, -c(10,35,37,38,39, 44, 45, 68, 69, 70, 71, 21, 20, 78, 60)]


#For the regressions below, I did not bother splitting my training data into validation as I am able to just submit my results directly to Kaggle to measure performance.
#However, you should ALWAYS split your data into training, validation, and testing.


#Stepwise Regression
my_dummy <- data.frame(predict(dummyVars("~ .", data= my_data[sapply(my_data, is.factor)]), newdata= my_data))
my_data <- cbind(my_data, my_dummy)
my_data <- my_data[, !sapply(my_data, is.factor)]

train <-subset(my_data, Id<=1460)
test <- subset(my_data, Id>1460)

final_reg <- step(lm(log(SalePrice) ~ . - Id - GrLivArea + log(GrLivArea), train))

predict1 <- exp(predict(final_reg, test))

#LASSO Regression
train <-subset(my_data, Id<=1460)
test <- subset(my_data, Id>1460)

my_data1 <- rbind(train, test)

X <- model.matrix(Id ~ . - SalePrice - GrLivArea - LotArea + log(GrLivArea) + log(LotArea) , my_data1)[, -1]

X <- cbind(my_data1$Id, X)

X_training<-subset(X,X[,1]<=1460)
X_prediction<-subset(X,X[,1]>=1461)

y_real <- log(train$SalePrice)
y_fake <- log(test$SalePrice)


lasso <- glmnet(x = X_training, y = y_real, alpha = 1)
plot(lasso)
CV <-  cv.glmnet(x = X_training, y = y_real, alpha = 1)
plot(CV)
plot(CV$lambda.min)
lasso.opt.fit <-glmnet(x = X_training, y = y_real, alpha = 1, lambda = CV$lambda.min) #estimate the model with the optimal penalty
lasso.testing <- exp(predict(lasso.opt.fit, s = CV$lambda.min, newx =X_prediction))
coef(lasso.opt.fit)

#Ridge
CV2 <-  cv.glmnet(x = X_training, y = y_real, alpha = 0)
ridge.opt.fit <-glmnet(x = X_training, y = y_real, alpha = 0, lambda = CV2$lambda.min) #estimate the model with the optimal penalty
ridge.testing <- exp(predict(ridge.opt.fit, s = CV2$lambda.min, newx =X_prediction))

