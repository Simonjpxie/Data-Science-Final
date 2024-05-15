rm(list = ls())

# Set working directory
setwd("/Users/xiejiapeng/Desktop/Data Science")

# Load required libraries
library(readxl)
library(httr)
library(tidyverse)
library(lubridate)
library(dplyr)
library(tibble)
library(glmnet)
library(caret)
library(doParallel)
library(xgboost)
library(Metrics)
library(randomForest)
library(forecast)
library(tseries)

# Download and load train_data
url <- "https://github.com/zhentaoshi/Econ5821/raw/main/data_example/US_PCE_training.xlsx"
temp_file <- tempfile()
download.file(url, temp_file, mode = "wb")
train_data <- read_excel(temp_file, col_names = TRUE)

# Train_data processing
train_data <- train_data[-c(2, 3, 4), ]
train_data <- as.data.frame(t(train_data))
colnames(train_data) <- as.character(unlist(train_data[1, ]))
train_data <- train_data[-1, ]
train_data$Month <- row.names(train_data)
train_data$Month <- as.Date(train_data$Month, format = "%b-%Y")
train_data[] <- lapply(train_data, function(x) as.numeric(as.character(x)))
names(train_data)[1] <- "PCE"
names(train_data)[175] <- "Religious organizations' services to households.1"
names(train_data)[201] <- "Religious organizations' services to households.2"
names(train_data)[176] <- "Foundations and grantmaking and giving services to households.1"
names(train_data)[202] <- "Foundations and grantmaking and giving services to households.2"
train_data <- train_data[, -206]
train_data <- train_data %>% mutate(Inflation_Rate = (log(PCE / lag(PCE))) * 12, Inflation_Rate = replace_na(Inflation_Rate, 0))
train_data$month <- 1:732
train_data <- train_data[,-1]
scaled_features <- scale(train_data[setdiff(names(train_data), c('month', 'Inflation_Rate'))])
colnames(scaled_features) <- paste("scaled", colnames(scaled_features), sep="_")
train_data <- cbind(train_data[c('month', 'Inflation_Rate')], scaled_features)

# Download and load test_data
url <- "https://github.com/zhentaoshi/Econ5821/raw/main/data_example/US_PCE_testing_fake.xlsx"
temp_file <- tempfile()
download.file(url, temp_file, mode = "wb")
test_data <- read_excel(temp_file, col_names = TRUE)

# Test_data processing
test_data <- test_data[-c(2, 3), ]
test_data <- as.data.frame(t(test_data))
colnames(test_data) <- as.character(unlist(test_data[1, ]))
test_data <- test_data[-1, ]
test_data$Month <- row.names(test_data)
test_data$Month <- as.Date(test_data$Month, format = "%b-%Y")
test_data[] <- lapply(test_data, function(x) as.numeric(as.character(x)))
names(test_data)[1] <- "PCE"
names(test_data)[175] <- "Religious organizations' services to households.1"
names(test_data)[201] <- "Religious organizations' services to households.2"
names(test_data)[176] <- "Foundations and grantmaking and giving services to households.1"
names(test_data)[202] <- "Foundations and grantmaking and giving services to households.2"
test_data <- test_data[, -206]
test_data <- test_data %>% mutate(Inflation_Rate = (log(PCE / lag(PCE))) * 12, Inflation_Rate = replace_na(Inflation_Rate, 0))
test_data$month <- 733:782
test_data <- test_data[,-1]
scaled_features <- scale(test_data[setdiff(names(test_data), c('month', 'Inflation_Rate'))])
colnames(scaled_features) <- paste("scaled", colnames(scaled_features), sep="_")
test_data <- cbind(test_data[c('month', 'Inflation_Rate')], scaled_features)

# Lag 1 month
train_data_lag1 <- train_data %>% mutate(Inflation_Rate_lag1 = lag(Inflation_Rate)) %>% na.omit()
test_data_lag1 <- test_data %>% mutate(Inflation_Rate_lag1 = lag(Inflation_Rate))
test_data_lag1 <- test_data_lag1 %>% filter(!is.na(Inflation_Rate_lag1))

# LASSO Model
registerDoParallel(cores = detectCores() - 1)
model_fit <- train(Inflation_Rate ~ ., data = train_data_lag1, method = "glmnet",
                   trControl = trainControl("cv", number = 10), tuneLength = 10)
predictions <- predict(model_fit, newdata = test_data_lag1)
print(predictions)
performance <- postResample(predictions, test_data_lag1$Inflation_Rate)
print(performance)
plot(model_fit$finalModel)

# Gradient Boosting Model
common_features <- intersect(names(train_data_lag1), names(test_data_lag1))
train_data_lag1 <- train_data_lag1[, common_features]
test_data_lag1 <- test_data_lag1[, common_features]
train_x <- train_data_lag1[, setdiff(names(train_data_lag1), "Inflation_Rate")]
train_y <- train_data_lag1$Inflation_Rate
test_x <- test_data_lag1[, setdiff(names(test_data_lag1), "Inflation_Rate")]
test_y <- test_data_lag1$Inflation_Rate

fitControl <- trainControl(method = "cv", number = 5, savePredictions = "final", verboseIter = TRUE)
model <- train(train_x, train_y, method = "xgbTree", trControl = fitControl, metric = "RMSE")
predictions <- predict(model, test_x)
print(predictions)
rmse <- rmse(test_y, predictions)
mae <- mae(test_y, predictions)
r2 <- R2(test_y, predictions)
print(paste("RMSE:", rmse))
print(paste("MAE:", mae))
print(paste("R-squared:", r2))
plot(test_y, type='o', col='blue', pch=20, xlab="Index", ylab="Inflation Rate", main="Actual vs Predicted Inflation Rates")
points(predictions, type='o', col='red', pch=20)

# Random Forest Model
train_x <- train_data_lag1 %>% select(-month, -Inflation_Rate)
train_y <- train_data_lag1$Inflation_Rate
test_x <- test_data_lag1 %>% select(-month, -Inflation_Rate)
test_y <- test_data_lag1$Inflation_Rate
set.seed(123)
rf_model <- randomForest(train_x, train_y, ntree=500)
predictions <- predict(rf_model, test_x)
print(predictions)
performance <- postResample(predictions, test_y)
print(performance)
results <- data.frame(Actual = test_y, Predicted = predictions)
plot(results$Actual, type = 'l', col = 'blue', xlab = 'Month Index', ylab = 'Inflation Rate', main = 'Actual vs Predicted Inflation')
lines(results$Predicted, col = 'red')

# AR Model
data_ts <- ts(train_data_lag1$Inflation_Rate, frequency=12)
adf.test(data_ts, alternative = "stationary")
ar_order <- auto.arima(data_ts, ic = "aic", trace = TRUE, stepwise = FALSE, approximation = FALSE)
ar_model <- Arima(data_ts, order=c(ar_order$arma[1], 0, 0))
future_values <- forecast(ar_model, h=49)
print(future_values)
plot(future_values)
summary(ar_model)

# Lag 3 month
train_data_lag3 <- train_data %>% mutate(Inflation_Rate_lag3 = lag(Inflation_Rate, 3)) %>% na.omit()
test_data_lag3 <- test_data %>% mutate(Inflation_Rate_lag3 = lag(Inflation_Rate, 3))
test_data_lag3 <- test_data_lag3 %>% filter(!is.na(Inflation_Rate_lag3))

# LASSO Model
registerDoParallel(cores = detectCores() - 1)
model_fit <- train(Inflation_Rate ~ ., data = train_data_lag3, method = "glmnet",
                   trControl = trainControl("cv", number = 10), tuneLength = 10)
predictions <- predict(model_fit, newdata = test_data_lag3)
print(predictions)
performance <- postResample(predictions, test_data_lag3$Inflation_Rate)
print(performance)
plot(model_fit$finalModel)

# Gradient Boosting Model
common_features <- intersect(names(train_data_lag3), names(test_data_lag3))
train_data_lag3 <- train_data_lag3[, common_features]
test_data_lag3 <- test_data_lag3[, common_features]
train_x <- train_data_lag3[, setdiff(names(train_data_lag3), "Inflation_Rate")]
train_y <- train_data_lag3$Inflation_Rate
test_x <- test_data_lag3[, setdiff(names(test_data_lag3), "Inflation_Rate")]
test_y <- test_data_lag3$Inflation_Rate

fitControl <- trainControl(method = "cv", number = 5, savePredictions = "final", verboseIter = TRUE)
model <- train(train_x, train_y, method = "xgbTree", trControl = fitControl, metric = "RMSE")
predictions <- predict(model, test_x)
print(predictions)
rmse <- rmse(test_y, predictions)
mae <- mae(test_y, predictions)
r2 <- R2(test_y, predictions)
print(paste("RMSE:", rmse))
print(paste("MAE:", mae))
print(paste("R-squared:", r2))
plot(test_y, type='o', col='blue', pch=20, xlab="Index", ylab="Inflation Rate", main="Actual vs Predicted Inflation Rates")
points(predictions, type='o', col='red', pch=20)

# Random Forest Model
train_x <- train_data_lag3 %>% select(-month, -Inflation_Rate)
train_y <- train_data_lag3$Inflation_Rate
test_x <- test_data_lag3 %>% select(-month, -Inflation_Rate)
test_y <- test_data_lag3$Inflation_Rate
set.seed(123)
rf_model <- randomForest(train_x, train_y, ntree=500)
predictions <- predict(rf_model, test_x)
print(predictions)
performance <- postResample(predictions, test_y)
print(performance)
results <- data.frame(Actual = test_y, Predicted = predictions)
plot(results$Actual, type = 'l', col = 'blue', xlab = 'Month Index', ylab = 'Inflation Rate', main = 'Actual vs Predicted Inflation')
lines(results$Predicted, col = 'red')

# AR Model
data_ts <- ts(train_data_lag3$Inflation_Rate, frequency=12)
adf.test(data_ts, alternative = "stationary")
ar_order <- auto.arima(data_ts, ic = "aic", trace = TRUE, stepwise = FALSE, approximation = FALSE)
ar_model <- Arima(data_ts, order=c(ar_order$arma[1], 0, 0))
future_values <- forecast(ar_model, h=49)
print(future_values)
plot(future_values)
summary(ar_model)

# Lag 12 month
train_data_lag12 <- train_data %>% mutate(Inflation_Rate_lag12 = lag(Inflation_Rate, 12)) %>% na.omit()
test_data_lag12 <- test_data %>% mutate(Inflation_Rate_lag12 = lag(Inflation_Rate, 12))
test_data_lag12 <- test_data_lag12 %>% filter(!is.na(Inflation_Rate_lag12))

# LASSO Model
registerDoParallel(cores = detectCores() - 1)
model_fit <- train(Inflation_Rate ~ ., data = train_data_lag12, method = "glmnet",
                   trControl = trainControl("cv", number = 10), tuneLength = 10)
predictions <- predict(model_fit, newdata = test_data_lag12)
print(predictions)
performance <- postResample(predictions, test_data_lag12$Inflation_Rate)
print(performance)
plot(model_fit$finalModel)

# Gradient Boosting Model
common_features <- intersect(names(train_data_lag12), names(test_data_lag12))
train_data_lag12 <- train_data_lag12[, common_features]
test_data_lag12 <- test_data_lag12[, common_features]
train_x <- train_data_lag12[, setdiff(names(train_data_lag12), "Inflation_Rate")]
train_y <- train_data_lag12$Inflation_Rate
test_x <- test_data_lag12[, setdiff(names(test_data_lag12), "Inflation_Rate")]
test_y <- test_data_lag12$Inflation_Rate

fitControl <- trainControl(method = "cv", number = 5, savePredictions = "final", verboseIter = TRUE)
model <- train(train_x, train_y, method = "xgbTree", trControl = fitControl, metric = "RMSE")
predictions <- predict(model, test_x)
print(predictions)
rmse <- rmse(test_y, predictions)
mae <- mae(test_y, predictions)
r2 <- R2(test_y, predictions)
print(paste("RMSE:", rmse))
print(paste("MAE:", mae))
print(paste("R-squared:", r2))
plot(test_y, type='o', col='blue', pch=20, xlab="Index", ylab="Inflation Rate", main="Actual vs Predicted Inflation Rates")
points(predictions, type='o', col='red', pch=20)

# Random Forest Model
train_x <- train_data_lag12 %>% select(-month, -Inflation_Rate)
train_y <- train_data_lag12$Inflation_Rate
test_x <- test_data_lag12 %>% select(-month, -Inflation_Rate)
test_y <- test_data_lag12$Inflation_Rate
set.seed(123)
rf_model <- randomForest(train_x, train_y, ntree=500)
predictions <- predict(rf_model, test_x)
print(predictions)
performance <- postResample(predictions, test_y)
print(performance)
results <- data.frame(Actual = test_y, Predicted = predictions)
plot(results$Actual, type = 'l', col = 'blue', xlab = 'Month Index', ylab = 'Inflation Rate', main = 'Actual vs Predicted Inflation')
lines(results$Predicted, col = 'red')

# AR Model
data_ts <- ts(train_data_lag12$Inflation_Rate, frequency=12)
adf.test(data_ts, alternative = "stationary")
ar_order <- auto.arima(data_ts, ic = "aic", trace = TRUE, stepwise = FALSE, approximation = FALSE)
ar_model <- Arima(data_ts, order=c(ar_order$arma[1], 0, 0))
future_values <- forecast(ar_model, h=49)
print(future_values)
plot(future_values)
summary(ar_model)




