# HW1 Problem 4b
set.seed(12345)
folds <- 10

# create data matrix of 1000 observations of 10000 Rademacher random variables
# Rademacher RV has P(-1) = 0.5 and P(+1) = 0.5
n <- 1000
p <- 10000
x <- as.matrix(matrix(ifelse(runif(n*p, -1, 1) < 0, -1, 1), nrow = n, ncol = p))

# output vector is the sum of the data rows (i.e., beta is vector of 1's)
y <- as.matrix(rowSums(x))

# create vector of lambdas
lambdas <- c(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)

# initialize matrices for each fold of each lambda
train_mse <- matrix(0, nrow = folds, ncol = length(lambdas))
colnames(train_mse) <- lambdas
test_mse <- matrix(0, nrow = folds, ncol = length(lambdas))
colnames(test_mse) <- lambdas
risk <- matrix(0, nrow = folds, ncol = length(lambdas))
colnames(risk) <- lambdas

# loop through 13 lambdas
for (i in 1:length(lambdas)) {
  # initialize size of each 10-fold CV (1000/10 = 100)
  k <- n/folds
  # create index vector (1-1000 shuffled randomly)
  index <- sample(c(1:1000))
  # loop through 10 folds
  for (j in 1:folds) {
    # create k'th fold split indeces
    fold <- index[(k-99):k]
    k <- k+100
    # split data
    x_train <- x[-fold,]
    x_test <- x[fold,]
    y_train <- y[-fold,]
    y_test <- y[fold,]
    # fit ridge regression model
    model <- glmnet::glmnet(x_train, y_train, alpha = 0, lambda = lambdas[i])
    # calculate training MSE
    train_mse[j,i] <- mean((y_train - predict.glmnet(model, s = lambdas[i], newx = x_train))^2)
    # calculate test MSE
    test_mse[j,i] <- mean((y_test - predict.glmnet(model, s = lambdas[i], newx = x_test))^2)
    # calculate risk (1 + squared euclidian distance between model beta's & vector of 1's)
    risk[j,i] <- 1 + sum((model$beta@x - 1)^2)
  }
}

# training error (best lambda = 0.01)
train <- colSums(train_mse)/folds
boxplot(train, ylab = "Mean squared error")
title("Training error boxplot")
train_min <- which.min(train)
train_best_lambda <- lambdas[train_min]
train_df <- data.frame(lambdas, train)

# test error (best lambda = 100)
test <- colSums(test_mse)/folds
boxplot(test, ylab = "Mean squared error")
title("Test error boxplot")
test_min <- which.min(test)
test_best_lambda <- lambdas[test_min]
test_df <- data.frame(lambdas, test)

# risk (best lambda = 100)
risk <- colSums(risk)/folds
boxplot(risk, ylab = "Risk")
title("Risk boxplot")
risk_min <- which.min(risk)
risk_best_lambda <- lambdas[risk_min]
risk_df <- data.frame(lambdas, risk)
