###############################################################################
# PA PIZZA SESSION: FEATURE SELECTION
#
# Author: Roger Smeets
# Date:   2 May 2017
# 
###############################################################################

### After specifying the proper location of the csv file directly below, you can
### run the data prep (until line 103) in one go. Of course, feel free to browse
### the code to understand what is going on

### Initializations

# Clear environment
rm(list = ls())

# Specify location of data
infile <- "C:/Users/UTRRS1/Documents/Randstad/Training/patent_litigation.csv"
descr  <- "C:/Users/UTRRS1/Documents/Randstad/Training/patent_litigation_feature_description.csv"

# Load required packages
library(data.table)
library(caret)
library(randomForest)

### Functions

fOutcomeClassifier <- function(raw_outcome){
  # Takes the original litigation outcome and classifies it as:
  #   1. Settlement or
  #   2. Plaintiff win or
  #   3. Defendant win or
  #   4. Other
  #
  # Args:
  #   raw_outcome: the uncleaned original outcome of the case (character type)
  #
  # Returns:
  #   One of the four outcomes listed above
  
  if      (grepl("^Stipulated",raw_outcome))            "Settlement" 
  else if (grepl("^Plaintiff voluntary",raw_outcome))   "Settlement"
  else if (grepl("^Clai",raw_outcome))                  "Plaintiff win"
  else if (grepl("^Defendant",raw_outcome))             "Defendant win"
  else                                                  "Other"
}

fLogPlusOne <- function(x){
  # Take the log of x plus one (to avoid missing values when x is zero)
  #
  # Args:
  #   x: Any number
  #
  # Returns:
  #   Log(x +1)
  log(x+1)
}

# Vectorize functions
vOutcomeClassifier <- Vectorize(fOutcomeClassifier)

### Load and clean data

# Load litigation dataset
litigation    <- fread(infile)
descriptions  <- fread(descr)

# Show dataset description
descriptions

# Clean up outcome data to identify settlements versus trials
litigation[ , cleanOutcome := vOutcomeClassifier(outcome)][
  , toTrial := ifelse(!(cleanOutcome %in% c("Settlement", "Other")), 1, 0)]

# Create foreign firm dummy
litigation[ , foreignFirm := (firmtype == "Foreign")]

# Take logs of skewed features
logcols <- c("employees", "net_sales", "rdi", "subsidiary_num", "subsidiary_us",
             "n_inv", "claims", "bcits", "fcits" , "npats")

litigation[ , (logcols) := lapply(.SD, fLogPlusOne), .SDcols = logcols]

# One-hot-encoding of venue
dumcols   <- c("venue")
litigation[ , (dumcols) := lapply(.SD, factor), .SDcols = dumcols]

fematrix  <- data.table(model.matrix(~ venue, litigation))
fematrix[ , "(Intercept)" := NULL]
fematrix[ , (colnames(fematrix)) := lapply(.SD, factor), .SDcols = colnames(fematrix)]

# Create the data table with all modelling features
modeldata <- cbind(litigation[ , .(toTrial, foreignFirm, us_listing, employees,
                                   net_sales, rdi, cash_share, subsidiary_num,
                                   subsidiary_us, pat_age, n_inv, claims, 
                                   bcits, fcits, firstinv_us, ipr, npats)],
                   fematrix)

# Drop all records with missing data on any of the features
modeldata <- na.omit(modeldata)

# Turn toTrial into a factor
modeldata[ , toTrial := factor(toTrial)]

# Run some of your own exploratory analysis if you want to get a better feel
# for the data

# ...

### Feature selection using caret

# We will use a random forest to model the settlement vs litigation 
# outcome. We will try (at least) three strategies:
#   1. A full model (no explicit feature selection)
#   2. Stepbackward, a.k.a. recursive feature elimination (RFE)
#   3. Regularized random forest (RRF)

# Split data in train and test data
set.seed(666)
index <- sample(seq(nrow(modeldata)), size = floor(0.8*nrow(modeldata)))

train  <- modeldata[index]
trainx <- modeldata[index][ , toTrial := NULL]
trainy <- factor(modeldata[index][ , toTrial])

test  <- modeldata[-index]
testx <- modeldata[-index][ , toTrial := NULL]
testy <- factor(modeldata[-index][ , toTrial])

## Model1: A model without explicit feature selection
##
##         Key parameter(s):
##           1. mtry: A vector of integers (e.g. c(1,2,3,4)) that tells the 
##                    algorithm from how many features it should sample when 
##                    deciding how to split the next part of the tree. The 
##                    algorithm will output results for each value of mtry
##                    that you specify. 
##
##        Warning! The more values you specify in mtry, the longer the algorithm
##                 will run. For reasonable performance, do not specify more than
##                 4.

  # Specify training parameters
  traincontrol <- trainControl(method = "cv", number = 4)
  
  mtryVector   <- #your code here
  traingrid    <- expand.grid(mtry = mtryVector) 
  
  # Train the model
  model1      <- train(trainx, trainy, method = "rf", 
                       trControl = traincontrol, tuneGrid = traingrid)
  
  model1
  
  # Compute feature importance
  importance1   <- varImp(model1, scale = T)
  dfImportance1 <- importance1$importance
  dfImportance1 <-  dfImportance1[order(-dfImportance1$Overall), , drop = F] #Order by decreasing importance
  
  # Count relevant features
  print(sprintf("Number of features in model 1 with >0 importance: %s",
                length(dfImportance1[dfImportance1$Overall>0,])))
  
  dfImportance1$Features <- row.names(dfImportance1)
  barplot(dfImportance1$Overall, names.arg = dfImportance1$Features)

  # Put model accuracy and included features in a list
  model1list  <- list(model1$results$Accuracy[which(model1$results$mtry==model1$bestTune$mtry)],
                      predictors(model1))
  
  print(sprintf("Accuracy of model 1: %s", round(as.numeric(model1list[1]), 3)))


## Model2: A model using the stepbackward wrapper, a.k.a. Recursive Feature 
##         Elimination (RFE). 
##         
##         Key parameter(s):
##           1. sizes: A vector of integers that tells the algorithm which model
##                     sizes (in terms of # features) to consider. It serves as
##                     a set of stopping criteria. The algorithm will output
##                     results for each model size (as well as the full model)

  # Specify training parameters
  rfecontrol  <- rfeControl(functions = rfFuncs, method = "cv", number = 4)
  
  # Train the model
  sizeVector  <- #your code here
  model2      <- rfe(trainx, trainy, sizes = sizeVector, rfeControl = rfecontrol )
  
  model2
  
  # Compute feature performance
  dfImportance2   <- varImp(model2$fit, scale = T)
  dfImportance2 <-  dfImportance2[order(-dfImportance2$Overall), , drop = F] #Order by decreasing importance
  
  # Count relevant features
  print(sprintf("Number of features selected in model 2: %s", model2$bestSubset))
  
  dfImportance2$Features <- row.names(dfImportance2)
  barplot(dfImportance2$Overall, names.arg = dfImportance2$Features)
  
  # Put model accuracy and included features in a list
  model2list  <- list(max(model2$results$Accuracy), predictors(model2))
  
  print(sprintf("Accuracy of model 2: %s", round(as.numeric(model2list[1]), 3)))
  

## Model 3: A model using regularized random forests, which is done using the  
##          RRF package.
##
##          Key parameters:
##            1. mtry:    See "regular" Random Forest algorithm above
##
##            2. coefReg: Controls the strength of regularization. Values close
##                        to 1 imply weak regularization, values close to 0
##                        imply strong regularization. This is the lambda in the
##                        ppt presentation. You can specify this in a vector of
##                        values between 0 and 1
##  
##            3. coefImp: This coefficient adjusts the coefReg as follows:
##  
##                          coefReg = 1 - coefImp + coefImp * RelImp
##  
##                        where RelImp is the relative importance of feature X
##                        compared to the maximum importance of all features, 
##                        calculated from a regular RF. So, if coefImp = 0, 
##                        coefReg = coefReg and we have a "regular" RRF. For 
##                        0 < coefImp < 1, we obtain a so called "guided" RF, 
##                        where features with a small relative importance are
##                        penalized (and more so with a higher coefImp). Again
##                        you can specify this in a vector of values between 0
##                        and 1.
##
##            !Warning:   Do not specify too many separate values of coefImp and
##                        relImp, since performance deteriorates fast. Choosing
##                        2 or 3 separate values for each vector yields fairly
##                        reasonable performance

  # Drop and (re)load relevant packages
  detach("package:randomForest")
  library(RRF)
  library(randomForest)
  
  # Specify the training parameters
  regcontrol  <- trainControl(method = "cv", number = 4)
  
  mtryVector2    <- #your code here
  coefRegVector  <- #your code here
  coefImpVector  <- #your code here
  reggrid        <- expand.grid(mtry = mtryVector2, coefReg = coefRegVector,
                                coefImp = coefImpVector)
  
  #Train the model
  model5      <- train(trainx, trainy, method = "RRF", 
                       trControl = regcontrol, tuneGrid = reggrid)
  
  model5
  
  # Compute feature importance
  dfImportance5 <- model5$importance
  dfImportance5 <- dfImportance5[order(-dfImportance5$Overall), , drop = F] #Order by decreasing importance
  
  # Count relevant features
  print(sprintf("Number of features in model 1 with >0 importance: %s",
                length(dfImportance5[dfImportance5$Overall>0,])))
  
  dfImportance5$Features <- row.names(dfImportance5)
  barplot(dfImportance5$Overall, names.arg = dfImportance5$Features)
  
  # Put model accuracy and included features in a list
  model5accuracy <- model5$results[model5$results$mtry    == model5$bestTune$mtry &
                                     model5$results$coefReg == model5$bestTune$coefReg &
                                     model5$results$coefImp == model5$bestTune$coefImp, 
                                   "Accuracy"]
  
  model5features <- dfImportance5[dfImportance5$Overall>0,]$Features
  
  model5list  <- list(model5accuracy, model5features)


## You can compare the three different models in terms of (1) their CV accuracy,
## and (2) the features included in the model. 

# your code here

### Test set performance

## Assess the performance (accuracy) of the three different models on the test
## set. Which model performs best?

# your code here

  
  
  
## APPENDIX: Simulated annealing and genetic algorithm
  
## Model 4: A model using the genetic algorithm wrapper, which is denoted in 
##          caret by GA. Note that if you have multiple cores available on your
##          machine, it is worthwhile to parallelize this procedure by setting
##          genParallel = T in the gafsControl call below
##
##          Key parameter(s):
##            1. iters:       The number (= integer) of iterations. Be careful when adjusting
##                            this as it affects performance significantly
##
##            2. popSize:     The different number (= integer) of models to start with (which the
##                            algorithm will generate randomly). Be careful when 
##                            adjusting this as it affects performance significantly
##
##            3. prcrossover: The probability (between 0 and 1) that the two parts of the parent
##                            models will be cut and crossed
##
##            4. pmutation:   The probability (between 0 and 1) that the child node will be mutated
##                            before being deployed
##
##            5. elite:       The probability (between 0 and 1) that the best performing models
##                            carry over to the next generation "as is"

  # Specify training parameters
  gacontrol   <- gafsControl(functions = rfGA, method = "cv", number = 4,
                             genParallel = F, allowParallel = T)
  
  # Train the model
  iterNum     <- #your code here
  popNum      <- #your code here
  prCross     <- #your code here
  prMut       <- #your code here
  prElite     <- #your code here
    
  model4      <- gafs(trainx, trainy, iters = iterNum, popSize = popNum, 
                      pcrossover = prCross, pmutation = prMut, 
                      elite = prElite, gafsControl = gacontrol)
  
  model4
  
  # Compute feature performance
  dfImportance4 <- model4$fit$importance
  sortindex     <- order(dfImportance4, decreasing = T)
  sequence      <- seq(1, nrow(dfImportance4))
  dfImportance4 <- data.frame(cbind(dfImportance4, sequence))
  dfImportance4 <- dfImportance4[match(sortindex, dfImportance3$sequence),]
  
  
  # Count relevant features
  print(sprintf("Number of features in model 4 with >0 importance: %s",
                nrow(dfImportance3[as.numeric(dfImportance4$MeanDecreaseGini)>0,])))
  
  barplot(dfImportance4$MeanDecreaseGini, names.arg = row.names(dfImportance3))
  
  
  # Put model accuracy and included features in a list
  model4list  <- list(max(model4$averages$Accuracy), model4$ga$final)
  
  
## Model 5: A model using the simulated annealing wrapper, which is denoted in 
##          caret by SA. 
##
##          Key parameters:
##            1.iters:  The number (= integer) of iterations. Be careful when adjusting
##                      this as it affects performance significantly

  # Specify training parameters
  sacontrol   <- safsControl(functions = rfSA, method = "cv", number = 4,
                             allowParallel = T)
  iterNum     <- #your code here
  
  # Train the model
  model5      <- safs(trainx, trainy, iters = iterNum, safsControl = sacontrol)
  
  model5
  
  # Compute feature importance
  dfImportance5 <- model5$fit$importance
  sortindex     <- order(dfImportance5, decreasing = T)
  sequence      <- seq(1, nrow(dfImportance5))
  dfImportance5 <- data.frame(cbind(dfImportance5, sequence))
  dfImportance5 <- dfImportance5[match(sortindex, dfImportance5$sequence),]
  
  # Count relevant features
  print(sprintf("Number of features in model 5 with >0 importance: %s",
                nrow(dfImportance5[as.numeric(dfImportance5$MeanDecreaseGini)>0,])))
  
  barplot(dfImportance5$MeanDecreaseGini, names.arg = row.names(dfImportance5))
  
  # Put model accuracy and included features in a list
  model5list      <- list(max(model5$averages$Accuracy), model5$sa$final)
  
  
