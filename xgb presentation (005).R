##############################################################################
# Actuarial Data Science Seminar                                 February 2019
# Callum Hughes                                                         AXA XL
# Matthew Evans                                      EMC Actuarial & Analytics
##############################################################################
# This script generates an insurance loss like dataset using a Poisson
# claims frequency and gamma claims severity.
#
# XGBoost, a series of GLMs and a GAM is applied to the dataset, with
# comparisons made using an out-of-sample test set.
##############################################################################

library(tidyverse)
library(data.table)
library(xgboost)
library(caret)
library(e1071) # for caret
library(statmod)
library(tweedie)

##############################################################################
# Data generation
##############################################################################

# Setting the number of simulation rows

number.of.rows <- 100000

# We have three categorical variables, with 2, 3 and 3 levels each.
# Each level has a corresponding relativity which will be used to generate
# a gamma mean.

cat1 = c("A", "B")
rel1 = c(1, 5)

cat2 = c("1", "2", "3")
rel2 = c(1, 4, 6)

cat3 = c("x", "y", "z")
rel3 = c(1, 1, 1)

# To test non-linear characteristics we introduce a numerical variable setting
# the relativity using a polynomial.
set.seed(1)
num4 = runif(1, min = -1, max = 1)
rel4 = 2 * (num4 ^ 2) + 1

# Generating a data table of random numerical variables
set.seed(2)
num4.rel <- data.table(num4 = runif(number.of.rows, min = -1, max = 1))
num4.rel[, rel4 := 2 * (num4 ^ 2) + 1]

# A plot shows our polynomial variable
# Note how it looks a bit like the Age curve in personal lines motor

ggplot(data = num4.rel, mapping = aes(x = num4, y = rel4)) + 
  geom_line() + 
  ylim(min = 0, max = 3)

# To test how well models pick up an interaction, we introduce one between
# cat1 and cat2

rel23 = matrix(c(1, 1, 1, 1, 1, 1, 1, 1.7, 1),
               nrow = 3,
               dimnames = list(c("1", "2", "3"),
                               c("x", "y", "z"))
               )

# Creating a full data table, randomly assigning factor levels
set.seed(3)
data.gen <- data.table(index = 1:number.of.rows,
                       "cat1" = sample(cat1, number.of.rows, replace = TRUE),
                       "cat2" = sample(cat2, number.of.rows, replace = TRUE),
                       "cat3" = sample(cat3, number.of.rows, replace = TRUE),
                       "num4" = runif(number.of.rows, min = -1, max = 1))

# We simulate the gamma shape variable by multiplying relativities together

shape <- function(x1, x2, x3, x4) {
  
  val <- rel1[cat1 %in% x1] * 
         rel2[cat2 %in% x2] *
         rel3[cat3 %in% x3] *
         (2 * (x4 ^ 2) + 1) * 
         rel23[cat2 %in% x2, cat3 %in% x3]   
  
  return(val)
  
}

# The gamma scale parameter is set to 1 for convenience

scale <- 1

# We now simulate a Poisson for claims frequency
set.seed(4)
data.gen[, freq := rpois(n = 1, lambda = 1.2), by = index]
data.gen[, loss.mean := shape(x1 = cat1, x2 = cat2, x3 = cat3, x4 = num4), by = index]
set.seed(5)
data.gen[, combined := sum(rgamma(n = freq, shape = loss.mean, scale = scale)), by = index]

# A proportion of the generated records are assigned to the train dataset

split.ratio <- 0.7
set.seed(6)
data.gen[, split := sample(c("train", "test"), size = number.of.rows, replace = TRUE, prob = c(split.ratio, 1 - split.ratio))]
data.gen


# One hot encoding creates a sparse matrix - with factor levels converted to binary
# columns - suitable for XGBoost

ohe.features <- c("cat1", "cat2", "cat3", "num4")
ohe.string <- paste(" ~ ", paste(ohe.features, collapse=" + "), sep = "")
ohe.dummies <- dummyVars(ohe.string, data = data.gen[, ..ohe.features])

data.ohe <- predict(ohe.dummies, newdata = data.gen[, ..ohe.features]) %>% as.data.table

data.labels <- data.gen[, .(combined)]


# Train and Test datasets are converted into the XGBoost format
# (This allows additional functionality like the watchlist we will see later)

xgb.data <- xgb.DMatrix(data = data.matrix(data.ohe),
                        label = data.labels$combined)

xgb.data.train <- xgb.DMatrix(data = data.matrix(data.ohe[data.gen$split == "train", ]),
                         label = data.labels[data.gen$split == "train", ]$combined)

xgb.data.test <- xgb.DMatrix(data = data.matrix(data.ohe[data.gen$split == "test", ]),
                         label = data.labels[data.gen$split == "test", ]$combined)

# The watchlist is a list of datasets for use during model fitting
# The training dataset will be used for training but at each stage
# the predictive error will be calculated using the test set too.

watchlist <- list(train = xgb.data.train, test = xgb.data.test)

# Running a simple XGBoost model
set.seed(10)
xgb.model <- xgb.train(data = xgb.data.train,
                       nrounds = 100,
                       max.depth = 5,
                       eta = 0.3,
                       gamma = 0,
                       colsample_bytree = 1,
                       min_child_weight =1,
                       subsample = 1,
                       watchlist = watchlist,
                       tweedie_variance_power = 1.75,
                       objective = "reg:tweedie"
                       )

# Plotting the train and test results for each iteration of boosting (adding another tree)
# lets us choose an optimum level, getting the best fit before we begin to overfit.

ggplot(xgb.model[["evaluation_log"]], aes(iter)) +
  geom_line(aes(y = `train_tweedie_nloglik@1.75`, colour = "train")) + 
  geom_line(aes(y = `test_tweedie_nloglik@1.75`, colour = "test"))

# This function shows the minimum test error - the optimum number of iterations

eval.log <- xgb.model$evaluation_log
names(eval.log) <- c("iter", "train", "test")
eval.log[(order(test))][1]

# Predicting values for the test set using our model

#p <- predict(xgb.model, xgb.data.test)

# XGBoost gives the importance of each feature, helping us communicate with 
# underwriters about feature predictiveness

importance.matrix <- xgb.importance(feature_names = xgb.model$feature_names, model = xgb.model)
xgb.plot.importance(importance_matrix = importance.matrix)

# A Grid-Search lets us try out many different combination of hyper-parameters to
# see which combination gives us the best fit.

# To avoid overfitting, we use cross-fold validation, repeating the fit each
# time with a different "fold" as the hold out sample (test set)

train.control <- trainControl(method = "cv",
                              number = 5,
                              verboseIter = TRUE,
                              allowParallel = FALSE
                              )

# Combinations of hyper-parameters

tune.grid <- expand.grid(nrounds = c(10, 15, 25, 50),
                         max_depth = c(1, 3, 5),
                         eta = c(0.3),
                         gamma = c(0),
                         colsample_bytree = c(0.5, 1),
                         min_child_weight = c(0.5, 1),
                         subsample = c(1)
                         )

# Run all combinations, returning the best model under cross-fold validation
set.seed(11)
xgb.model.tuned <- train(x = xgb.data.train,
                         y = data.labels[data.gen$split == "train", ]$combined,
                         watchlist = watchlist,
                         method = "xgbTree",
                         objective = "reg:tweedie",
                         tweedie_variance_power = 1.75,
                         trControl = train.control,
                         tuneGrid = tune.grid
                         )

# Summarise best model

xgb.model.tuned %>% summary
xgb.model.tuned$bestTune

# Present importance matrix (error here.. does this work with caret implementation?)

#importance.matrix.2 <- xgb.importance(feature_names = xgb.model.tuned$feature_names, model = xgb.model.tuned)
#xgb.plot.importance(importance_matrix = importance.matrix.2)

# Utility function to print RMSE. RMSE calculated directly using function from the 
# caret package

f_print_rmse <- function(model, data = data.gen[split == "test"]) {
  
  p <-  predict(model, data, type="response")
  print(paste("Root Mean Square Error is:", round(RMSE(p, data.gen[split == "test"]$combined), 3), sep = " "))
  
}

# Utility function to print RMSE for XGBoost. Model parameter different from GLM/GAM.

f_print_rmse_xgb <- function(model, data = data.gen[split == "test"]) {
  
  p <- predict(model, data)
  print(paste("Root Mean Square Error is:", round(RMSE(p, data.gen[split == "test"]$combined), 3), sep = " "))
  
}

# Fit a simple GLM for comparison
# Note: no interaction term included, no attempt to handle non-linear term

glm.simple <- glm(combined ~ cat1 + cat2 + cat3 + num4,
                  data = data.gen[split == "train"],
                  family = tweedie(link.power = 0, 
                                   var.power = 1.75)
                  ) 
glm.simple %>% summary
glm.simple %>% f_print_rmse

# Fit a GLM with interaction for comparison
# Note: no attempt to handle non-linear term

glm.interaction <- glm(combined ~ cat1 + cat2 + cat3 + num4 + 
                         cat2*cat3,
                       data = data.gen[split == "train"],
                       family = tweedie(link.power = 0, 
                                        var.power = 1.75)
                       )
glm.interaction %>% summary
glm.interaction %>% f_print_rmse


# Fit a GLM with interaction and banded non-linear term for comparison

data.gen[, num4.banded := cut(num4, 
                       breaks = 10)]

glm.banded <- glm(combined ~ cat1 + cat2 + cat3 + 
                    num4.banded + 
                    cat2*cat3,
                  data = data.gen[split == "train"],
                  family = tweedie(link.power = 0, 
                                   var.power = 1.75)
                  ) 
glm.banded %>% summary
glm.banded %>% f_print_rmse

# Best GLM with interaction and correct polynomial for non-linear term

glm.poly <- glm(combined ~ cat1 + cat2 + cat3 + 
                  poly(num4, 2) + 
                  cat2*cat3,
                 data = data.gen[split == "train"],
                 family = tweedie(link.power = 0, 
                                  var.power = 1.75)
)
glm.poly %>% summary
glm.poly %>% f_print_rmse

# GAM challenger with interaction and smoother for num4

gam <- mgcv::gam(formula = combined ~ cat1 + cat2 + cat3 + 
                   cat2*cat3 + 
                   s(num4),
                 data = data.gen[split == "train"],
                 family = mgcv::Tweedie(p=1.7, 
                                        link = power(0)))  
gam %>% summary
gam %>% f_print_rmse


# XGBoost simple model
xgb.model %>% f_print_rmse_xgb(data = xgb.data.test)

# XGBoost model with grid-search
xgb.model.tuned %>% f_print_rmse_xgb(data = xgb.data.test)

# Citations

citation("tidyverse")
citation("glm")
citation("xgboost")
citation("mgcv")
citation("tidyverse")
