# Don't forget to load the module:
# module load new gcc/4.8.2 r/3.6.0
# Job submission to Euler using multicore
# bsub -W 24:00 -n 18 "Rscript --vanilla --slave Euler_As_All_plus_minus_20190910.R"

# load workspace with subsets
load("AsymptomaticSubsets_20190910.RData")

# load packages
library("caret")
library("caretEnsemble")
library("randomForest")
library("adabag")
library("foreach")
library("doParallel")
library("caTools")
library("DALEX")
library("ingredients")
library("tidyverse")
library("dplyr")


# register the cluster
registerDoParallel(18) 


# ht.As.All.arm.train ####
## Control list ####
set.seed(2019)
ht.As.All.arm.train.ctrl <- trainControl(method = "repeatedcv",
                                         repeats = 5, number = 10, 
                                         returnResamp = "all", savePredictions = "all", classProbs = TRUE, 
                                         summaryFunction = twoClassSummary, allowParallel = TRUE,
                                         index = createResample(ht.As.All.arm.train$condition))


## Build models within caretList ####
set.seed(2019)
ht.As.All.arm.train.list <- caretList(x = ht.As.All.arm.train[,-1], y = as.factor(ht.As.All.arm.train$condition),
                                      trControl = ht.As.All.arm.train.ctrl,
                                      #methodList = c("rf", "nnet"),
                                      metric = "ROC",
                                      preProc = c("center", "scale"),
                                      trace = FALSE,
                                      tuneList = list(
                                          rf = caretModelSpec(method = "rf", ntree = 3000,
                                                              tuneGrid = expand.grid(mtry = c(3, 5, 10, 15, 21))),
                                          AdaBoost.M1 = caretModelSpec(method = "AdaBoost.M1",
                                                                       tuneGrid = expand.grid(mfinal = 1000, maxdepth = c(1, 3), coeflearn = "Breiman")),
                                          svmRadial = caretModelSpec(method = "svmRadial",
                                                                     tuneGrid = expand.grid(sigma = c(.01, .015, 0.2), C = c(0.75, 0.9, 1, 1.1, 1.25))),
                                          nnet = caretModelSpec(method = "nnet", MaxNWts = 481,
                                                                tuneGrid = expand.grid(size = c(3, 5, 10, 20), 
                                                                                       decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))))
)

## Ensemble the caretList models using AdaBag in caretStack ####
set.seed(2019)
ht.As.All.arm.train.ensemble <- caretStack(
    ht.As.All.arm.train.list,
    method = "AdaBag", 
    metric = "ROC",
    trControl = trainControl(
        method = "boot",
        number = 20,
        savePredictions = "final",
        classProbs = TRUE,
        summaryFunction = twoClassSummary
    )
)

## Build performance table for training set ####
ht.As.All.arm.train.results <- data.frame(id = "ht.As.All.arm", section = "arm", subset = "train", condition = "Asymptomatic", 
                                          subinfection = "All", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"), 
                                          accuracy = c(confusionMatrix(ht.As.All.arm.train.list$rf$pred$pred, ht.As.All.arm.train.list$rf$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.All.arm.train.list$AdaBoost.M1$pred$pred, ht.As.All.arm.train.list$AdaBoost.M1$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.All.arm.train.list$svmRadial$pred$pred, ht.As.All.arm.train.list$svmRadial$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.All.arm.train.list$nnet$pred$pred, ht.As.All.arm.train.list$nnet$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.All.arm.train.ensemble$ens_model$pred$pred, ht.As.All.arm.train.ensemble$ens_model$pred$obs, positive = "M")$overall["Accuracy"][[1]]),
                                          ROC = c(getTrainPerf(ht.As.All.arm.train.list$rf)$TrainROC[[1]],
                                                  getTrainPerf(ht.As.All.arm.train.list$AdaBoost.M1)$TrainROC[[1]],
                                                  getTrainPerf(ht.As.All.arm.train.list$svmRadial)$TrainROC[[1]],
                                                  getTrainPerf(ht.As.All.arm.train.list$nnet)$TrainROC[[1]],
                                                  getTrainPerf(ht.As.All.arm.train.ensemble$ens_model)$TrainROC[[1]]),
                                          sensitivity = c(getTrainPerf(ht.As.All.arm.train.list$rf)$TrainSens[[1]],
                                                          getTrainPerf(ht.As.All.arm.train.list$AdaBoost.M1)$TrainSens[[1]],
                                                          getTrainPerf(ht.As.All.arm.train.list$svmRadial)$TrainSens[[1]],
                                                          getTrainPerf(ht.As.All.arm.train.list$nnet)$TrainSens[[1]],
                                                          getTrainPerf(ht.As.All.arm.train.ensemble$ens_model)$TrainSens[[1]]),
                                          specificity = c(getTrainPerf(ht.As.All.arm.train.list$rf)$TrainSpec[[1]],
                                                          getTrainPerf(ht.As.All.arm.train.list$AdaBoost.M1)$TrainSpec[[1]],
                                                          getTrainPerf(ht.As.All.arm.train.list$svmRadial)$TrainSpec[[1]],
                                                          getTrainPerf(ht.As.All.arm.train.list$nnet)$TrainSpec[[1]],
                                                          getTrainPerf(ht.As.All.arm.train.ensemble$ens_model)$TrainSpec[[1]]),
                                          best = NA
)


## Variable importance using DALEX ####
set.seed(2019)
ht.As.All.arm.train.list.varImp.rf <- replicate(20, feature_importance(explain(ht.As.All.arm.train.list$rf, label = "rf", data = ht.As.All.arm.train[,-1], y = as.numeric(ht.As.All.arm.train$condition))), simplify = FALSE)
ht.As.All.arm.train.list.varImp.rf <- ht.As.All.arm.train.list.varImp.rf %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.rf = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.rf))

set.seed(2019)
ht.As.All.arm.train.list.varImp.ada <- replicate(20, feature_importance(explain(ht.As.All.arm.train.list$AdaBoost.M1, label = "ada", data = ht.As.All.arm.train[,-1], y = as.numeric(ht.As.All.arm.train$condition))), simplify = FALSE)
ht.As.All.arm.train.list.varImp.ada <- ht.As.All.arm.train.list.varImp.ada %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ada = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ada))

set.seed(2019)
ht.As.All.arm.train.list.varImp.svm <- replicate(20, feature_importance(explain(ht.As.All.arm.train.list$svmRadial, label = "svm", data = ht.As.All.arm.train[,-1], y = as.numeric(ht.As.All.arm.train$condition))), simplify = FALSE)
ht.As.All.arm.train.list.varImp.svm <- ht.As.All.arm.train.list.varImp.svm %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.svm = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.svm))

set.seed(2019)
ht.As.All.arm.train.list.varImp.nnet <- replicate(20, feature_importance(explain(ht.As.All.arm.train.list$nnet, label = "nnet", data = ht.As.All.arm.train[,-1], y = as.numeric(ht.As.All.arm.train$condition))), simplify = FALSE)
ht.As.All.arm.train.list.varImp.nnet <- ht.As.All.arm.train.list.varImp.nnet %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.nnet = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.nnet))

set.seed(2019)
ht.As.All.arm.train.list.varImp.ensemble <- replicate(20, feature_importance(explain(ht.As.All.arm.train.ensemble, label = "ensemble", data = ht.As.All.arm.train[,-1], y = as.numeric(ht.As.All.arm.train$condition))), simplify = FALSE)
ht.As.All.arm.train.list.varImp.ensemble <- ht.As.All.arm.train.list.varImp.ensemble %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ensemble = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ensemble))

## Merge variable importances from all models in caretList into one dataframe
ht.As.All.arm.train.list.varImp.all <- ht.As.All.arm.train.list.varImp.rf %>% 
    left_join(ht.As.All.arm.train.list.varImp.ada, by = "variable") %>% 
    left_join(ht.As.All.arm.train.list.varImp.svm, by = "variable") %>% 
    left_join(ht.As.All.arm.train.list.varImp.nnet, by = "variable") %>% 
    left_join(ht.As.All.arm.train.list.varImp.ensemble, by = "variable")


### Get five most important variables per model to add it to the ht.As.All.arm.test.results
ht.As.All.arm.train.list.varImp.rf.bestFive <- ht.As.All.arm.train.list.varImp.rf %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.All.arm.train.list.varImp.ada.bestFive <- ht.As.All.arm.train.list.varImp.ada %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.All.arm.train.list.varImp.svm.bestFive <- ht.As.All.arm.train.list.varImp.svm %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.All.arm.train.list.varImp.nnet.bestFive <- ht.As.All.arm.train.list.varImp.nnet %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.All.arm.train.list.varImp.ensemble.bestFive <- ht.As.All.arm.train.list.varImp.ensemble %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)



## Make predictions in test set ####
## Make predictions in each model of caretList by using lapply
ht.As.All.arm.train.list.preds <- lapply(ht.As.All.arm.train.list, predict, newdata = ht.As.All.arm.test[,-1], type = "raw") #newdata = ht.As.All.arm.test[,-1]

ht.As.All.arm.test.results <- data.frame(id = "ht.As.All.arm", section = "arm", subset = "test", condition = "Asymptomatic", 
                                         subinfection = "All", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"),
                                         accuracy = c(confusionMatrix(ht.As.All.arm.train.list.preds$rf, ht.As.All.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                      confusionMatrix(ht.As.All.arm.train.list.preds$AdaBoost.M1, ht.As.All.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                      confusionMatrix(ht.As.All.arm.train.list.preds$svmRadial, ht.As.All.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                      confusionMatrix(ht.As.All.arm.train.list.preds$nnet, ht.As.All.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                      confusionMatrix(predict(ht.As.All.arm.train.ensemble, ht.As.All.arm.test[,-1]), ht.As.All.arm.test$condition, positive = "M")$overall["Accuracy"][[1]]),
                                         ROC = c(colAUC(predict(ht.As.All.arm.train.list$rf, newdata = ht.As.All.arm.test[,-1], type = "prob"), ht.As.All.arm.test$condition)[[1]],
                                                 colAUC(predict(ht.As.All.arm.train.list$AdaBoost.M1, newdata = ht.As.All.arm.test[,-1], type = "prob"), ht.As.All.arm.test$condition)[[1]],
                                                 colAUC(predict(ht.As.All.arm.train.list$svmRadial, newdata = ht.As.All.arm.test[,-1], type = "prob"), ht.As.All.arm.test$condition)[[1]],
                                                 colAUC(predict(ht.As.All.arm.train.list$nnet, newdata = ht.As.All.arm.test[,-1], type = "prob"), ht.As.All.arm.test$condition)[[1]],
                                                 colAUC(predict(ht.As.All.arm.train.ensemble, newdata = ht.As.All.arm.test[,-1], type = "prob"), ht.As.All.arm.test$condition)[[1]]),
                                         sensitivity = c(confusionMatrix(ht.As.All.arm.train.list.preds$rf, ht.As.All.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                         confusionMatrix(ht.As.All.arm.train.list.preds$AdaBoost.M1, ht.As.All.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                         confusionMatrix(ht.As.All.arm.train.list.preds$svmRadial, ht.As.All.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                         confusionMatrix(ht.As.All.arm.train.list.preds$nnet, ht.As.All.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                         confusionMatrix(predict(ht.As.All.arm.train.ensemble, ht.As.All.arm.test[,-1]), ht.As.All.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]]),
                                         specificity = c(confusionMatrix(ht.As.All.arm.train.list.preds$rf, ht.As.All.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                         confusionMatrix(ht.As.All.arm.train.list.preds$AdaBoost.M1, ht.As.All.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                         confusionMatrix(ht.As.All.arm.train.list.preds$svmRadial, ht.As.All.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                         confusionMatrix(ht.As.All.arm.train.list.preds$nnet, ht.As.All.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                         confusionMatrix(predict(ht.As.All.arm.train.ensemble, ht.As.All.arm.test[,-1]), ht.As.All.arm.test$condition, positive = "M")$byClass["Specificity"][[1]]),
                                         best = c(paste(ht.As.All.arm.train.list.varImp.rf.bestFive[, 1], collapse = ", ", sep = ""), 
                                                  paste(ht.As.All.arm.train.list.varImp.ada.bestFive[, 1], collapse = ", ", sep = ""),
                                                  paste(ht.As.All.arm.train.list.varImp.svm.bestFive[, 1], collapse = ", ", sep = ""),
                                                  paste(ht.As.All.arm.train.list.varImp.nnet.bestFive[, 1], collapse = ", ", sep = ""),
                                                  paste(ht.As.All.arm.train.list.varImp.ensemble.bestFive[, 1], collapse = ", ", sep = ""))
)


# ht.As.All.foot.train ####
## Control list ####
set.seed(2019)
ht.As.All.foot.train.ctrl <- trainControl(method = "repeatedcv",
                                          repeats = 5, number = 10, 
                                          returnResamp = "all", savePredictions = "all", classProbs = TRUE, 
                                          summaryFunction = twoClassSummary, allowParallel = TRUE,
                                          index = createResample(ht.As.All.foot.train$condition))


## Build models within caretList ####
set.seed(2019)
ht.As.All.foot.train.list <- caretList(x = ht.As.All.foot.train[,-1], y = as.factor(ht.As.All.foot.train$condition),
                                       trControl = ht.As.All.foot.train.ctrl,
                                       #methodList = c("rf", "nnet"),
                                       metric = "ROC",
                                       preProc = c("center", "scale"),
                                       trace = FALSE,
                                       tuneList = list(
                                           rf = caretModelSpec(method = "rf", ntree = 3000,
                                                               tuneGrid = expand.grid(mtry = c(3, 5, 10, 15, 21))),
                                           AdaBoost.M1 = caretModelSpec(method = "AdaBoost.M1",
                                                                        tuneGrid = expand.grid(mfinal = 1000, maxdepth = c(1, 3), coeflearn = "Breiman")),
                                           svmRadial = caretModelSpec(method = "svmRadial",
                                                                      tuneGrid = expand.grid(sigma = c(.01, .015, 0.2), C = c(0.75, 0.9, 1, 1.1, 1.25))),
                                           nnet = caretModelSpec(method = "nnet", MaxNWts = 481,
                                                                 tuneGrid = expand.grid(size = c(3, 5, 10, 20), 
                                                                                        decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))))
)

## Ensemble the caretList models using AdaBag in caretStack ####
set.seed(2019)
ht.As.All.foot.train.ensemble <- caretStack(
    ht.As.All.foot.train.list,
    method = "AdaBag", 
    metric = "ROC",
    trControl = trainControl(
        method = "boot",
        number = 20,
        savePredictions = "final",
        classProbs = TRUE,
        summaryFunction = twoClassSummary
    )
)

## Build performance table for training set ####
ht.As.All.foot.train.results <- data.frame(id = "ht.As.All.foot", section = "foot", subset = "train", condition = "Asymptomatic", 
                                           subinfection = "All", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"), 
                                           accuracy = c(confusionMatrix(ht.As.All.foot.train.list$rf$pred$pred, ht.As.All.foot.train.list$rf$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.All.foot.train.list$AdaBoost.M1$pred$pred, ht.As.All.foot.train.list$AdaBoost.M1$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.All.foot.train.list$svmRadial$pred$pred, ht.As.All.foot.train.list$svmRadial$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.All.foot.train.list$nnet$pred$pred, ht.As.All.foot.train.list$nnet$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.All.foot.train.ensemble$ens_model$pred$pred, ht.As.All.foot.train.ensemble$ens_model$pred$obs, positive = "M")$overall["Accuracy"][[1]]),
                                           ROC = c(getTrainPerf(ht.As.All.foot.train.list$rf)$TrainROC[[1]],
                                                   getTrainPerf(ht.As.All.foot.train.list$AdaBoost.M1)$TrainROC[[1]],
                                                   getTrainPerf(ht.As.All.foot.train.list$svmRadial)$TrainROC[[1]],
                                                   getTrainPerf(ht.As.All.foot.train.list$nnet)$TrainROC[[1]],
                                                   getTrainPerf(ht.As.All.foot.train.ensemble$ens_model)$TrainROC[[1]]),
                                           sensitivity = c(getTrainPerf(ht.As.All.foot.train.list$rf)$TrainSens[[1]],
                                                           getTrainPerf(ht.As.All.foot.train.list$AdaBoost.M1)$TrainSens[[1]],
                                                           getTrainPerf(ht.As.All.foot.train.list$svmRadial)$TrainSens[[1]],
                                                           getTrainPerf(ht.As.All.foot.train.list$nnet)$TrainSens[[1]],
                                                           getTrainPerf(ht.As.All.foot.train.ensemble$ens_model)$TrainSens[[1]]),
                                           specificity = c(getTrainPerf(ht.As.All.foot.train.list$rf)$TrainSpec[[1]],
                                                           getTrainPerf(ht.As.All.foot.train.list$AdaBoost.M1)$TrainSpec[[1]],
                                                           getTrainPerf(ht.As.All.foot.train.list$svmRadial)$TrainSpec[[1]],
                                                           getTrainPerf(ht.As.All.foot.train.list$nnet)$TrainSpec[[1]],
                                                           getTrainPerf(ht.As.All.foot.train.ensemble$ens_model)$TrainSpec[[1]]),
                                           best = NA
)


## Variable importance using DALEX ####
set.seed(2019)
ht.As.All.foot.train.list.varImp.rf <- replicate(20, feature_importance(explain(ht.As.All.foot.train.list$rf, label = "rf", data = ht.As.All.foot.train[,-1], y = as.numeric(ht.As.All.foot.train$condition))), simplify = FALSE)
ht.As.All.foot.train.list.varImp.rf <- ht.As.All.foot.train.list.varImp.rf %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.rf = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.rf))

set.seed(2019)
ht.As.All.foot.train.list.varImp.ada <- replicate(20, feature_importance(explain(ht.As.All.foot.train.list$AdaBoost.M1, label = "ada", data = ht.As.All.foot.train[,-1], y = as.numeric(ht.As.All.foot.train$condition))), simplify = FALSE)
ht.As.All.foot.train.list.varImp.ada <- ht.As.All.foot.train.list.varImp.ada %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ada = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ada))

set.seed(2019)
ht.As.All.foot.train.list.varImp.svm <- replicate(20, feature_importance(explain(ht.As.All.foot.train.list$svmRadial, label = "svm", data = ht.As.All.foot.train[,-1], y = as.numeric(ht.As.All.foot.train$condition))), simplify = FALSE)
ht.As.All.foot.train.list.varImp.svm <- ht.As.All.foot.train.list.varImp.svm %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.svm = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.svm))

set.seed(2019)
ht.As.All.foot.train.list.varImp.nnet <- replicate(20, feature_importance(explain(ht.As.All.foot.train.list$nnet, label = "nnet", data = ht.As.All.foot.train[,-1], y = as.numeric(ht.As.All.foot.train$condition))), simplify = FALSE)
ht.As.All.foot.train.list.varImp.nnet <- ht.As.All.foot.train.list.varImp.nnet %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.nnet = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.nnet))

set.seed(2019)
ht.As.All.foot.train.list.varImp.ensemble <- replicate(20, feature_importance(explain(ht.As.All.foot.train.ensemble, label = "ensemble", data = ht.As.All.foot.train[,-1], y = as.numeric(ht.As.All.foot.train$condition))), simplify = FALSE)
ht.As.All.foot.train.list.varImp.ensemble <- ht.As.All.foot.train.list.varImp.ensemble %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ensemble = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ensemble))

## Merge variable importances from all models in caretList into one dataframe
ht.As.All.foot.train.list.varImp.all <- ht.As.All.foot.train.list.varImp.rf %>% 
    left_join(ht.As.All.foot.train.list.varImp.ada, by = "variable") %>% 
    left_join(ht.As.All.foot.train.list.varImp.svm, by = "variable") %>% 
    left_join(ht.As.All.foot.train.list.varImp.nnet, by = "variable") %>% 
    left_join(ht.As.All.foot.train.list.varImp.ensemble, by = "variable")


### Get five most important variables per model to add it to the ht.As.All.foot.test.results
ht.As.All.foot.train.list.varImp.rf.bestFive <- ht.As.All.foot.train.list.varImp.rf %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.All.foot.train.list.varImp.ada.bestFive <- ht.As.All.foot.train.list.varImp.ada %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.All.foot.train.list.varImp.svm.bestFive <- ht.As.All.foot.train.list.varImp.svm %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.All.foot.train.list.varImp.nnet.bestFive <- ht.As.All.foot.train.list.varImp.nnet %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.All.foot.train.list.varImp.ensemble.bestFive <- ht.As.All.foot.train.list.varImp.ensemble %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)



## Make predictions in test set ####
## Make predictions in each model of caretList by using lapply
ht.As.All.foot.train.list.preds <- lapply(ht.As.All.foot.train.list, predict, newdata = ht.As.All.foot.test[,-1], type = "raw") #newdata = ht.As.All.foot.test[,-1]

ht.As.All.foot.test.results <- data.frame(id = "ht.As.All.foot", section = "foot", subset = "test", condition = "Asymptomatic", 
                                          subinfection = "All", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"),
                                          accuracy = c(confusionMatrix(ht.As.All.foot.train.list.preds$rf, ht.As.All.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.All.foot.train.list.preds$AdaBoost.M1, ht.As.All.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.All.foot.train.list.preds$svmRadial, ht.As.All.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.All.foot.train.list.preds$nnet, ht.As.All.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(predict(ht.As.All.foot.train.ensemble, ht.As.All.foot.test[,-1]), ht.As.All.foot.test$condition, positive = "M")$overall["Accuracy"][[1]]),
                                          ROC = c(colAUC(predict(ht.As.All.foot.train.list$rf, newdata = ht.As.All.foot.test[,-1], type = "prob"), ht.As.All.foot.test$condition)[[1]],
                                                  colAUC(predict(ht.As.All.foot.train.list$AdaBoost.M1, newdata = ht.As.All.foot.test[,-1], type = "prob"), ht.As.All.foot.test$condition)[[1]],
                                                  colAUC(predict(ht.As.All.foot.train.list$svmRadial, newdata = ht.As.All.foot.test[,-1], type = "prob"), ht.As.All.foot.test$condition)[[1]],
                                                  colAUC(predict(ht.As.All.foot.train.list$nnet, newdata = ht.As.All.foot.test[,-1], type = "prob"), ht.As.All.foot.test$condition)[[1]],
                                                  colAUC(predict(ht.As.All.foot.train.ensemble, newdata = ht.As.All.foot.test[,-1], type = "prob"), ht.As.All.foot.test$condition)[[1]]),
                                          sensitivity = c(confusionMatrix(ht.As.All.foot.train.list.preds$rf, ht.As.All.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                          confusionMatrix(ht.As.All.foot.train.list.preds$AdaBoost.M1, ht.As.All.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                          confusionMatrix(ht.As.All.foot.train.list.preds$svmRadial, ht.As.All.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                          confusionMatrix(ht.As.All.foot.train.list.preds$nnet, ht.As.All.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                          confusionMatrix(predict(ht.As.All.foot.train.ensemble, ht.As.All.foot.test[,-1]), ht.As.All.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]]),
                                          specificity = c(confusionMatrix(ht.As.All.foot.train.list.preds$rf, ht.As.All.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                          confusionMatrix(ht.As.All.foot.train.list.preds$AdaBoost.M1, ht.As.All.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                          confusionMatrix(ht.As.All.foot.train.list.preds$svmRadial, ht.As.All.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                          confusionMatrix(ht.As.All.foot.train.list.preds$nnet, ht.As.All.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                          confusionMatrix(predict(ht.As.All.foot.train.ensemble, ht.As.All.foot.test[,-1]), ht.As.All.foot.test$condition, positive = "M")$byClass["Specificity"][[1]]),
                                          best = c(paste(ht.As.All.foot.train.list.varImp.rf.bestFive[, 1], collapse = ", ", sep = ""), 
                                                   paste(ht.As.All.foot.train.list.varImp.ada.bestFive[, 1], collapse = ", ", sep = ""),
                                                   paste(ht.As.All.foot.train.list.varImp.svm.bestFive[, 1], collapse = ", ", sep = ""),
                                                   paste(ht.As.All.foot.train.list.varImp.nnet.bestFive[, 1], collapse = ", ", sep = ""),
                                                   paste(ht.As.All.foot.train.list.varImp.ensemble.bestFive[, 1], collapse = ", ", sep = ""))
)


# ht.As.plus.arm.train ####
## Control list ####
set.seed(2019)
ht.As.plus.arm.train.ctrl <- trainControl(method = "repeatedcv",
                                          repeats = 5, number = 10, 
                                          returnResamp = "all", savePredictions = "all", classProbs = TRUE, 
                                          summaryFunction = twoClassSummary, allowParallel = TRUE,
                                          index = createResample(ht.As.plus.arm.train$condition))


## Build models within caretList ####
set.seed(2019)
ht.As.plus.arm.train.list <- caretList(x = ht.As.plus.arm.train[,-1], y = as.factor(ht.As.plus.arm.train$condition),
                                       trControl = ht.As.plus.arm.train.ctrl,
                                       #methodList = c("rf", "nnet"),
                                       metric = "ROC",
                                       preProc = c("center", "scale"),
                                       trace = FALSE,
                                       tuneList = list(
                                           rf = caretModelSpec(method = "rf", ntree = 3000,
                                                               tuneGrid = expand.grid(mtry = c(3, 5, 10, 15, 21))),
                                           AdaBoost.M1 = caretModelSpec(method = "AdaBoost.M1",
                                                                        tuneGrid = expand.grid(mfinal = 1000, maxdepth = c(1, 3), coeflearn = "Breiman")),
                                           svmRadial = caretModelSpec(method = "svmRadial",
                                                                      tuneGrid = expand.grid(sigma = c(.01, .015, 0.2), C = c(0.75, 0.9, 1, 1.1, 1.25))),
                                           nnet = caretModelSpec(method = "nnet", MaxNWts = 481,
                                                                 tuneGrid = expand.grid(size = c(3, 5, 10, 20), 
                                                                                        decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))))
)

## Ensemble the caretList models using AdaBag in caretStack ####
set.seed(2019)
ht.As.plus.arm.train.ensemble <- caretStack(
    ht.As.plus.arm.train.list,
    method = "AdaBag", 
    metric = "ROC",
    trControl = trainControl(
        method = "boot",
        number = 20,
        savePredictions = "final",
        classProbs = TRUE,
        summaryFunction = twoClassSummary
    )
)

## Build performance table for training set ####
ht.As.plus.arm.train.results <- data.frame(id = "ht.As.plus.arm", section = "arm", subset = "train", condition = "Asymptomatic", 
                                           subinfection = "plus", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"), 
                                           accuracy = c(confusionMatrix(ht.As.plus.arm.train.list$rf$pred$pred, ht.As.plus.arm.train.list$rf$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.plus.arm.train.list$AdaBoost.M1$pred$pred, ht.As.plus.arm.train.list$AdaBoost.M1$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.plus.arm.train.list$svmRadial$pred$pred, ht.As.plus.arm.train.list$svmRadial$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.plus.arm.train.list$nnet$pred$pred, ht.As.plus.arm.train.list$nnet$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.plus.arm.train.ensemble$ens_model$pred$pred, ht.As.plus.arm.train.ensemble$ens_model$pred$obs, positive = "M")$overall["Accuracy"][[1]]),
                                           ROC = c(getTrainPerf(ht.As.plus.arm.train.list$rf)$TrainROC[[1]],
                                                   getTrainPerf(ht.As.plus.arm.train.list$AdaBoost.M1)$TrainROC[[1]],
                                                   getTrainPerf(ht.As.plus.arm.train.list$svmRadial)$TrainROC[[1]],
                                                   getTrainPerf(ht.As.plus.arm.train.list$nnet)$TrainROC[[1]],
                                                   getTrainPerf(ht.As.plus.arm.train.ensemble$ens_model)$TrainROC[[1]]),
                                           sensitivity = c(getTrainPerf(ht.As.plus.arm.train.list$rf)$TrainSens[[1]],
                                                           getTrainPerf(ht.As.plus.arm.train.list$AdaBoost.M1)$TrainSens[[1]],
                                                           getTrainPerf(ht.As.plus.arm.train.list$svmRadial)$TrainSens[[1]],
                                                           getTrainPerf(ht.As.plus.arm.train.list$nnet)$TrainSens[[1]],
                                                           getTrainPerf(ht.As.plus.arm.train.ensemble$ens_model)$TrainSens[[1]]),
                                           specificity = c(getTrainPerf(ht.As.plus.arm.train.list$rf)$TrainSpec[[1]],
                                                           getTrainPerf(ht.As.plus.arm.train.list$AdaBoost.M1)$TrainSpec[[1]],
                                                           getTrainPerf(ht.As.plus.arm.train.list$svmRadial)$TrainSpec[[1]],
                                                           getTrainPerf(ht.As.plus.arm.train.list$nnet)$TrainSpec[[1]],
                                                           getTrainPerf(ht.As.plus.arm.train.ensemble$ens_model)$TrainSpec[[1]]),
                                           best = NA
)


## Variable importance using DALEX ####
set.seed(2019)
ht.As.plus.arm.train.list.varImp.rf <- replicate(20, feature_importance(explain(ht.As.plus.arm.train.list$rf, label = "rf", data = ht.As.plus.arm.train[,-1], y = as.numeric(ht.As.plus.arm.train$condition))), simplify = FALSE)
ht.As.plus.arm.train.list.varImp.rf <- ht.As.plus.arm.train.list.varImp.rf %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.rf = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.rf))

set.seed(2019)
ht.As.plus.arm.train.list.varImp.ada <- replicate(20, feature_importance(explain(ht.As.plus.arm.train.list$AdaBoost.M1, label = "ada", data = ht.As.plus.arm.train[,-1], y = as.numeric(ht.As.plus.arm.train$condition))), simplify = FALSE)
ht.As.plus.arm.train.list.varImp.ada <- ht.As.plus.arm.train.list.varImp.ada %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ada = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ada))

set.seed(2019)
ht.As.plus.arm.train.list.varImp.svm <- replicate(20, feature_importance(explain(ht.As.plus.arm.train.list$svmRadial, label = "svm", data = ht.As.plus.arm.train[,-1], y = as.numeric(ht.As.plus.arm.train$condition))), simplify = FALSE)
ht.As.plus.arm.train.list.varImp.svm <- ht.As.plus.arm.train.list.varImp.svm %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.svm = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.svm))

set.seed(2019)
ht.As.plus.arm.train.list.varImp.nnet <- replicate(20, feature_importance(explain(ht.As.plus.arm.train.list$nnet, label = "nnet", data = ht.As.plus.arm.train[,-1], y = as.numeric(ht.As.plus.arm.train$condition))), simplify = FALSE)
ht.As.plus.arm.train.list.varImp.nnet <- ht.As.plus.arm.train.list.varImp.nnet %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.nnet = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.nnet))

set.seed(2019)
ht.As.plus.arm.train.list.varImp.ensemble <- replicate(20, feature_importance(explain(ht.As.plus.arm.train.ensemble, label = "ensemble", data = ht.As.plus.arm.train[,-1], y = as.numeric(ht.As.plus.arm.train$condition))), simplify = FALSE)
ht.As.plus.arm.train.list.varImp.ensemble <- ht.As.plus.arm.train.list.varImp.ensemble %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ensemble = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ensemble))

## Merge variable importances from all models in caretList into one dataframe
ht.As.plus.arm.train.list.varImp.plus <- ht.As.plus.arm.train.list.varImp.rf %>% 
    left_join(ht.As.plus.arm.train.list.varImp.ada, by = "variable") %>% 
    left_join(ht.As.plus.arm.train.list.varImp.svm, by = "variable") %>% 
    left_join(ht.As.plus.arm.train.list.varImp.nnet, by = "variable") %>% 
    left_join(ht.As.plus.arm.train.list.varImp.ensemble, by = "variable")


### Get five most important variables per model to add it to the ht.As.plus.arm.test.results
ht.As.plus.arm.train.list.varImp.rf.bestFive <- ht.As.plus.arm.train.list.varImp.rf %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.plus.arm.train.list.varImp.ada.bestFive <- ht.As.plus.arm.train.list.varImp.ada %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.plus.arm.train.list.varImp.svm.bestFive <- ht.As.plus.arm.train.list.varImp.svm %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.plus.arm.train.list.varImp.nnet.bestFive <- ht.As.plus.arm.train.list.varImp.nnet %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.plus.arm.train.list.varImp.ensemble.bestFive <- ht.As.plus.arm.train.list.varImp.ensemble %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)



## Make predictions in test set ####
## Make predictions in each model of caretList by using lapply
ht.As.plus.arm.train.list.preds <- lapply(ht.As.plus.arm.train.list, predict, newdata = ht.As.plus.arm.test[,-1], type = "raw") #newdata = ht.As.plus.arm.test[,-1]

ht.As.plus.arm.test.results <- data.frame(id = "ht.As.plus.arm", section = "arm", subset = "test", condition = "Asymptomatic", 
                                          subinfection = "plus", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"),
                                          accuracy = c(confusionMatrix(ht.As.plus.arm.train.list.preds$rf, ht.As.plus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.plus.arm.train.list.preds$AdaBoost.M1, ht.As.plus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.plus.arm.train.list.preds$svmRadial, ht.As.plus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(ht.As.plus.arm.train.list.preds$nnet, ht.As.plus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                       confusionMatrix(predict(ht.As.plus.arm.train.ensemble, ht.As.plus.arm.test[,-1]), ht.As.plus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]]),
                                          ROC = c(colAUC(predict(ht.As.plus.arm.train.list$rf, newdata = ht.As.plus.arm.test[,-1], type = "prob"), ht.As.plus.arm.test$condition)[[1]],
                                                  colAUC(predict(ht.As.plus.arm.train.list$AdaBoost.M1, newdata = ht.As.plus.arm.test[,-1], type = "prob"), ht.As.plus.arm.test$condition)[[1]],
                                                  colAUC(predict(ht.As.plus.arm.train.list$svmRadial, newdata = ht.As.plus.arm.test[,-1], type = "prob"), ht.As.plus.arm.test$condition)[[1]],
                                                  colAUC(predict(ht.As.plus.arm.train.list$nnet, newdata = ht.As.plus.arm.test[,-1], type = "prob"), ht.As.plus.arm.test$condition)[[1]],
                                                  colAUC(predict(ht.As.plus.arm.train.ensemble, newdata = ht.As.plus.arm.test[,-1], type = "prob"), ht.As.plus.arm.test$condition)[[1]]),
                                          sensitivity = c(confusionMatrix(ht.As.plus.arm.train.list.preds$rf, ht.As.plus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                          confusionMatrix(ht.As.plus.arm.train.list.preds$AdaBoost.M1, ht.As.plus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                          confusionMatrix(ht.As.plus.arm.train.list.preds$svmRadial, ht.As.plus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                          confusionMatrix(ht.As.plus.arm.train.list.preds$nnet, ht.As.plus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                          confusionMatrix(predict(ht.As.plus.arm.train.ensemble, ht.As.plus.arm.test[,-1]), ht.As.plus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]]),
                                          specificity = c(confusionMatrix(ht.As.plus.arm.train.list.preds$rf, ht.As.plus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                          confusionMatrix(ht.As.plus.arm.train.list.preds$AdaBoost.M1, ht.As.plus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                          confusionMatrix(ht.As.plus.arm.train.list.preds$svmRadial, ht.As.plus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                          confusionMatrix(ht.As.plus.arm.train.list.preds$nnet, ht.As.plus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                          confusionMatrix(predict(ht.As.plus.arm.train.ensemble, ht.As.plus.arm.test[,-1]), ht.As.plus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]]),
                                          best = c(paste(ht.As.plus.arm.train.list.varImp.rf.bestFive[, 1], collapse = ", ", sep = ""), 
                                                   paste(ht.As.plus.arm.train.list.varImp.ada.bestFive[, 1], collapse = ", ", sep = ""),
                                                   paste(ht.As.plus.arm.train.list.varImp.svm.bestFive[, 1], collapse = ", ", sep = ""),
                                                   paste(ht.As.plus.arm.train.list.varImp.nnet.bestFive[, 1], collapse = ", ", sep = ""),
                                                   paste(ht.As.plus.arm.train.list.varImp.ensemble.bestFive[, 1], collapse = ", ", sep = ""))
)


# ht.As.plus.foot.train ####
## Control list ####
set.seed(2019)
ht.As.plus.foot.train.ctrl <- trainControl(method = "repeatedcv",
                                           repeats = 5, number = 10, 
                                           returnResamp = "all", savePredictions = "all", classProbs = TRUE, 
                                           summaryFunction = twoClassSummary, allowParallel = TRUE,
                                           index = createResample(ht.As.plus.foot.train$condition))


## Build models within caretList ####
set.seed(2019)
ht.As.plus.foot.train.list <- caretList(x = ht.As.plus.foot.train[,-1], y = as.factor(ht.As.plus.foot.train$condition),
                                        trControl = ht.As.plus.foot.train.ctrl,
                                        #methodList = c("rf", "nnet"),
                                        metric = "ROC",
                                        preProc = c("center", "scale"),
                                        trace = FALSE,
                                        tuneList = list(
                                            rf = caretModelSpec(method = "rf", ntree = 3000,
                                                                tuneGrid = expand.grid(mtry = c(3, 5, 10, 15, 21))),
                                            AdaBoost.M1 = caretModelSpec(method = "AdaBoost.M1",
                                                                         tuneGrid = expand.grid(mfinal = 1000, maxdepth = c(1, 3), coeflearn = "Breiman")),
                                            svmRadial = caretModelSpec(method = "svmRadial",
                                                                       tuneGrid = expand.grid(sigma = c(.01, .015, 0.2), C = c(0.75, 0.9, 1, 1.1, 1.25))),
                                            nnet = caretModelSpec(method = "nnet", MaxNWts = 481,
                                                                  tuneGrid = expand.grid(size = c(3, 5, 10, 20), 
                                                                                         decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))))
)

## Ensemble the caretList models using AdaBag in caretStack ####
set.seed(2019)
ht.As.plus.foot.train.ensemble <- caretStack(
    ht.As.plus.foot.train.list,
    method = "AdaBag", 
    metric = "ROC",
    trControl = trainControl(
        method = "boot",
        number = 20,
        savePredictions = "final",
        classProbs = TRUE,
        summaryFunction = twoClassSummary
    )
)

## Build performance table for training set ####
ht.As.plus.foot.train.results <- data.frame(id = "ht.As.plus.foot", section = "foot", subset = "train", condition = "Asymptomatic", 
                                            subinfection = "plus", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"), 
                                            accuracy = c(confusionMatrix(ht.As.plus.foot.train.list$rf$pred$pred, ht.As.plus.foot.train.list$rf$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.plus.foot.train.list$AdaBoost.M1$pred$pred, ht.As.plus.foot.train.list$AdaBoost.M1$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.plus.foot.train.list$svmRadial$pred$pred, ht.As.plus.foot.train.list$svmRadial$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.plus.foot.train.list$nnet$pred$pred, ht.As.plus.foot.train.list$nnet$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.plus.foot.train.ensemble$ens_model$pred$pred, ht.As.plus.foot.train.ensemble$ens_model$pred$obs, positive = "M")$overall["Accuracy"][[1]]),
                                            ROC = c(getTrainPerf(ht.As.plus.foot.train.list$rf)$TrainROC[[1]],
                                                    getTrainPerf(ht.As.plus.foot.train.list$AdaBoost.M1)$TrainROC[[1]],
                                                    getTrainPerf(ht.As.plus.foot.train.list$svmRadial)$TrainROC[[1]],
                                                    getTrainPerf(ht.As.plus.foot.train.list$nnet)$TrainROC[[1]],
                                                    getTrainPerf(ht.As.plus.foot.train.ensemble$ens_model)$TrainROC[[1]]),
                                            sensitivity = c(getTrainPerf(ht.As.plus.foot.train.list$rf)$TrainSens[[1]],
                                                            getTrainPerf(ht.As.plus.foot.train.list$AdaBoost.M1)$TrainSens[[1]],
                                                            getTrainPerf(ht.As.plus.foot.train.list$svmRadial)$TrainSens[[1]],
                                                            getTrainPerf(ht.As.plus.foot.train.list$nnet)$TrainSens[[1]],
                                                            getTrainPerf(ht.As.plus.foot.train.ensemble$ens_model)$TrainSens[[1]]),
                                            specificity = c(getTrainPerf(ht.As.plus.foot.train.list$rf)$TrainSpec[[1]],
                                                            getTrainPerf(ht.As.plus.foot.train.list$AdaBoost.M1)$TrainSpec[[1]],
                                                            getTrainPerf(ht.As.plus.foot.train.list$svmRadial)$TrainSpec[[1]],
                                                            getTrainPerf(ht.As.plus.foot.train.list$nnet)$TrainSpec[[1]],
                                                            getTrainPerf(ht.As.plus.foot.train.ensemble$ens_model)$TrainSpec[[1]]),
                                            best = NA
)


## Variable importance using DALEX ####
set.seed(2019)
ht.As.plus.foot.train.list.varImp.rf <- replicate(20, feature_importance(explain(ht.As.plus.foot.train.list$rf, label = "rf", data = ht.As.plus.foot.train[,-1], y = as.numeric(ht.As.plus.foot.train$condition))), simplify = FALSE)
ht.As.plus.foot.train.list.varImp.rf <- ht.As.plus.foot.train.list.varImp.rf %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.rf = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.rf))

set.seed(2019)
ht.As.plus.foot.train.list.varImp.ada <- replicate(20, feature_importance(explain(ht.As.plus.foot.train.list$AdaBoost.M1, label = "ada", data = ht.As.plus.foot.train[,-1], y = as.numeric(ht.As.plus.foot.train$condition))), simplify = FALSE)
ht.As.plus.foot.train.list.varImp.ada <- ht.As.plus.foot.train.list.varImp.ada %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ada = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ada))

set.seed(2019)
ht.As.plus.foot.train.list.varImp.svm <- replicate(20, feature_importance(explain(ht.As.plus.foot.train.list$svmRadial, label = "svm", data = ht.As.plus.foot.train[,-1], y = as.numeric(ht.As.plus.foot.train$condition))), simplify = FALSE)
ht.As.plus.foot.train.list.varImp.svm <- ht.As.plus.foot.train.list.varImp.svm %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.svm = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.svm))

set.seed(2019)
ht.As.plus.foot.train.list.varImp.nnet <- replicate(20, feature_importance(explain(ht.As.plus.foot.train.list$nnet, label = "nnet", data = ht.As.plus.foot.train[,-1], y = as.numeric(ht.As.plus.foot.train$condition))), simplify = FALSE)
ht.As.plus.foot.train.list.varImp.nnet <- ht.As.plus.foot.train.list.varImp.nnet %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.nnet = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.nnet))

set.seed(2019)
ht.As.plus.foot.train.list.varImp.ensemble <- replicate(20, feature_importance(explain(ht.As.plus.foot.train.ensemble, label = "ensemble", data = ht.As.plus.foot.train[,-1], y = as.numeric(ht.As.plus.foot.train$condition))), simplify = FALSE)
ht.As.plus.foot.train.list.varImp.ensemble <- ht.As.plus.foot.train.list.varImp.ensemble %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ensemble = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ensemble))

## Merge variable importances from all models in caretList into one dataframe
ht.As.plus.foot.train.list.varImp.plus <- ht.As.plus.foot.train.list.varImp.rf %>% 
    left_join(ht.As.plus.foot.train.list.varImp.ada, by = "variable") %>% 
    left_join(ht.As.plus.foot.train.list.varImp.svm, by = "variable") %>% 
    left_join(ht.As.plus.foot.train.list.varImp.nnet, by = "variable") %>% 
    left_join(ht.As.plus.foot.train.list.varImp.ensemble, by = "variable")


### Get five most important variables per model to add it to the ht.As.plus.foot.test.results
ht.As.plus.foot.train.list.varImp.rf.bestFive <- ht.As.plus.foot.train.list.varImp.rf %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.plus.foot.train.list.varImp.ada.bestFive <- ht.As.plus.foot.train.list.varImp.ada %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.plus.foot.train.list.varImp.svm.bestFive <- ht.As.plus.foot.train.list.varImp.svm %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.plus.foot.train.list.varImp.nnet.bestFive <- ht.As.plus.foot.train.list.varImp.nnet %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.plus.foot.train.list.varImp.ensemble.bestFive <- ht.As.plus.foot.train.list.varImp.ensemble %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)



## Make predictions in test set ####
## Make predictions in each model of caretList by using lapply
ht.As.plus.foot.train.list.preds <- lapply(ht.As.plus.foot.train.list, predict, newdata = ht.As.plus.foot.test[,-1], type = "raw") #newdata = ht.As.plus.foot.test[,-1]

ht.As.plus.foot.test.results <- data.frame(id = "ht.As.plus.foot", section = "foot", subset = "test", condition = "Asymptomatic", 
                                           subinfection = "plus", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"),
                                           accuracy = c(confusionMatrix(ht.As.plus.foot.train.list.preds$rf, ht.As.plus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.plus.foot.train.list.preds$AdaBoost.M1, ht.As.plus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.plus.foot.train.list.preds$svmRadial, ht.As.plus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.plus.foot.train.list.preds$nnet, ht.As.plus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(predict(ht.As.plus.foot.train.ensemble, ht.As.plus.foot.test[,-1]), ht.As.plus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]]),
                                           ROC = c(colAUC(predict(ht.As.plus.foot.train.list$rf, newdata = ht.As.plus.foot.test[,-1], type = "prob"), ht.As.plus.foot.test$condition)[[1]],
                                                   colAUC(predict(ht.As.plus.foot.train.list$AdaBoost.M1, newdata = ht.As.plus.foot.test[,-1], type = "prob"), ht.As.plus.foot.test$condition)[[1]],
                                                   colAUC(predict(ht.As.plus.foot.train.list$svmRadial, newdata = ht.As.plus.foot.test[,-1], type = "prob"), ht.As.plus.foot.test$condition)[[1]],
                                                   colAUC(predict(ht.As.plus.foot.train.list$nnet, newdata = ht.As.plus.foot.test[,-1], type = "prob"), ht.As.plus.foot.test$condition)[[1]],
                                                   colAUC(predict(ht.As.plus.foot.train.ensemble, newdata = ht.As.plus.foot.test[,-1], type = "prob"), ht.As.plus.foot.test$condition)[[1]]),
                                           sensitivity = c(confusionMatrix(ht.As.plus.foot.train.list.preds$rf, ht.As.plus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                           confusionMatrix(ht.As.plus.foot.train.list.preds$AdaBoost.M1, ht.As.plus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                           confusionMatrix(ht.As.plus.foot.train.list.preds$svmRadial, ht.As.plus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                           confusionMatrix(ht.As.plus.foot.train.list.preds$nnet, ht.As.plus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                           confusionMatrix(predict(ht.As.plus.foot.train.ensemble, ht.As.plus.foot.test[,-1]), ht.As.plus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]]),
                                           specificity = c(confusionMatrix(ht.As.plus.foot.train.list.preds$rf, ht.As.plus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                           confusionMatrix(ht.As.plus.foot.train.list.preds$AdaBoost.M1, ht.As.plus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                           confusionMatrix(ht.As.plus.foot.train.list.preds$svmRadial, ht.As.plus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                           confusionMatrix(ht.As.plus.foot.train.list.preds$nnet, ht.As.plus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                           confusionMatrix(predict(ht.As.plus.foot.train.ensemble, ht.As.plus.foot.test[,-1]), ht.As.plus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]]),
                                           best = c(paste(ht.As.plus.foot.train.list.varImp.rf.bestFive[, 1], collapse = ", ", sep = ""), 
                                                    paste(ht.As.plus.foot.train.list.varImp.ada.bestFive[, 1], collapse = ", ", sep = ""),
                                                    paste(ht.As.plus.foot.train.list.varImp.svm.bestFive[, 1], collapse = ", ", sep = ""),
                                                    paste(ht.As.plus.foot.train.list.varImp.nnet.bestFive[, 1], collapse = ", ", sep = ""),
                                                    paste(ht.As.plus.foot.train.list.varImp.ensemble.bestFive[, 1], collapse = ", ", sep = ""))
)


# ht.As.minus.arm.train ####
## Control list ####
set.seed(2019)
ht.As.minus.arm.train.ctrl <- trainControl(method = "repeatedcv",
                                           repeats = 5, number = 10, 
                                           returnResamp = "all", savePredictions = "all", classProbs = TRUE, 
                                           summaryFunction = twoClassSummary, allowParallel = TRUE,
                                           index = createResample(ht.As.minus.arm.train$condition))


## Build models within caretList ####
set.seed(2019)
ht.As.minus.arm.train.list <- caretList(x = ht.As.minus.arm.train[,-1], y = as.factor(ht.As.minus.arm.train$condition),
                                        trControl = ht.As.minus.arm.train.ctrl,
                                        #methodList = c("rf", "nnet"),
                                        metric = "ROC",
                                        preProc = c("center", "scale"),
                                        trace = FALSE,
                                        tuneList = list(
                                            rf = caretModelSpec(method = "rf", ntree = 3000,
                                                                tuneGrid = expand.grid(mtry = c(3, 5, 10, 15, 21))),
                                            AdaBoost.M1 = caretModelSpec(method = "AdaBoost.M1",
                                                                         tuneGrid = expand.grid(mfinal = 1000, maxdepth = c(1, 3), coeflearn = "Breiman")),
                                            svmRadial = caretModelSpec(method = "svmRadial",
                                                                       tuneGrid = expand.grid(sigma = c(.01, .015, 0.2), C = c(0.75, 0.9, 1, 1.1, 1.25))),
                                            nnet = caretModelSpec(method = "nnet", MaxNWts = 481,
                                                                  tuneGrid = expand.grid(size = c(3, 5, 10, 20), 
                                                                                         decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))))
)

## Ensemble the caretList models using AdaBag in caretStack ####
set.seed(2019)
ht.As.minus.arm.train.ensemble <- caretStack(
    ht.As.minus.arm.train.list,
    method = "AdaBag", 
    metric = "ROC",
    trControl = trainControl(
        method = "boot",
        number = 20,
        savePredictions = "final",
        classProbs = TRUE,
        summaryFunction = twoClassSummary
    )
)

## Build performance table for training set ####
ht.As.minus.arm.train.results <- data.frame(id = "ht.As.minus.arm", section = "arm", subset = "train", condition = "Asymptomatic", 
                                            subinfection = "minus", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"), 
                                            accuracy = c(confusionMatrix(ht.As.minus.arm.train.list$rf$pred$pred, ht.As.minus.arm.train.list$rf$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.minus.arm.train.list$AdaBoost.M1$pred$pred, ht.As.minus.arm.train.list$AdaBoost.M1$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.minus.arm.train.list$svmRadial$pred$pred, ht.As.minus.arm.train.list$svmRadial$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.minus.arm.train.list$nnet$pred$pred, ht.As.minus.arm.train.list$nnet$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.minus.arm.train.ensemble$ens_model$pred$pred, ht.As.minus.arm.train.ensemble$ens_model$pred$obs, positive = "M")$overall["Accuracy"][[1]]),
                                            ROC = c(getTrainPerf(ht.As.minus.arm.train.list$rf)$TrainROC[[1]],
                                                    getTrainPerf(ht.As.minus.arm.train.list$AdaBoost.M1)$TrainROC[[1]],
                                                    getTrainPerf(ht.As.minus.arm.train.list$svmRadial)$TrainROC[[1]],
                                                    getTrainPerf(ht.As.minus.arm.train.list$nnet)$TrainROC[[1]],
                                                    getTrainPerf(ht.As.minus.arm.train.ensemble$ens_model)$TrainROC[[1]]),
                                            sensitivity = c(getTrainPerf(ht.As.minus.arm.train.list$rf)$TrainSens[[1]],
                                                            getTrainPerf(ht.As.minus.arm.train.list$AdaBoost.M1)$TrainSens[[1]],
                                                            getTrainPerf(ht.As.minus.arm.train.list$svmRadial)$TrainSens[[1]],
                                                            getTrainPerf(ht.As.minus.arm.train.list$nnet)$TrainSens[[1]],
                                                            getTrainPerf(ht.As.minus.arm.train.ensemble$ens_model)$TrainSens[[1]]),
                                            specificity = c(getTrainPerf(ht.As.minus.arm.train.list$rf)$TrainSpec[[1]],
                                                            getTrainPerf(ht.As.minus.arm.train.list$AdaBoost.M1)$TrainSpec[[1]],
                                                            getTrainPerf(ht.As.minus.arm.train.list$svmRadial)$TrainSpec[[1]],
                                                            getTrainPerf(ht.As.minus.arm.train.list$nnet)$TrainSpec[[1]],
                                                            getTrainPerf(ht.As.minus.arm.train.ensemble$ens_model)$TrainSpec[[1]]),
                                            best = NA
)


## Variable importance using DALEX ####
set.seed(2019)
ht.As.minus.arm.train.list.varImp.rf <- replicate(20, feature_importance(explain(ht.As.minus.arm.train.list$rf, label = "rf", data = ht.As.minus.arm.train[,-1], y = as.numeric(ht.As.minus.arm.train$condition))), simplify = FALSE)
ht.As.minus.arm.train.list.varImp.rf <- ht.As.minus.arm.train.list.varImp.rf %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.rf = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.rf))

set.seed(2019)
ht.As.minus.arm.train.list.varImp.ada <- replicate(20, feature_importance(explain(ht.As.minus.arm.train.list$AdaBoost.M1, label = "ada", data = ht.As.minus.arm.train[,-1], y = as.numeric(ht.As.minus.arm.train$condition))), simplify = FALSE)
ht.As.minus.arm.train.list.varImp.ada <- ht.As.minus.arm.train.list.varImp.ada %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ada = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ada))

set.seed(2019)
ht.As.minus.arm.train.list.varImp.svm <- replicate(20, feature_importance(explain(ht.As.minus.arm.train.list$svmRadial, label = "svm", data = ht.As.minus.arm.train[,-1], y = as.numeric(ht.As.minus.arm.train$condition))), simplify = FALSE)
ht.As.minus.arm.train.list.varImp.svm <- ht.As.minus.arm.train.list.varImp.svm %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.svm = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.svm))

set.seed(2019)
ht.As.minus.arm.train.list.varImp.nnet <- replicate(20, feature_importance(explain(ht.As.minus.arm.train.list$nnet, label = "nnet", data = ht.As.minus.arm.train[,-1], y = as.numeric(ht.As.minus.arm.train$condition))), simplify = FALSE)
ht.As.minus.arm.train.list.varImp.nnet <- ht.As.minus.arm.train.list.varImp.nnet %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.nnet = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.nnet))

set.seed(2019)
ht.As.minus.arm.train.list.varImp.ensemble <- replicate(20, feature_importance(explain(ht.As.minus.arm.train.ensemble, label = "ensemble", data = ht.As.minus.arm.train[,-1], y = as.numeric(ht.As.minus.arm.train$condition))), simplify = FALSE)
ht.As.minus.arm.train.list.varImp.ensemble <- ht.As.minus.arm.train.list.varImp.ensemble %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ensemble = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ensemble))

## Merge variable importances from all models in caretList into one dataframe
ht.As.minus.arm.train.list.varImp.minus <- ht.As.minus.arm.train.list.varImp.rf %>% 
    left_join(ht.As.minus.arm.train.list.varImp.ada, by = "variable") %>% 
    left_join(ht.As.minus.arm.train.list.varImp.svm, by = "variable") %>% 
    left_join(ht.As.minus.arm.train.list.varImp.nnet, by = "variable") %>% 
    left_join(ht.As.minus.arm.train.list.varImp.ensemble, by = "variable")


### Get five most important variables per model to add it to the ht.As.minus.arm.test.results
ht.As.minus.arm.train.list.varImp.rf.bestFive <- ht.As.minus.arm.train.list.varImp.rf %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.minus.arm.train.list.varImp.ada.bestFive <- ht.As.minus.arm.train.list.varImp.ada %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.minus.arm.train.list.varImp.svm.bestFive <- ht.As.minus.arm.train.list.varImp.svm %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.minus.arm.train.list.varImp.nnet.bestFive <- ht.As.minus.arm.train.list.varImp.nnet %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.minus.arm.train.list.varImp.ensemble.bestFive <- ht.As.minus.arm.train.list.varImp.ensemble %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)



## Make predictions in test set ####
## Make predictions in each model of caretList by using lapply
ht.As.minus.arm.train.list.preds <- lapply(ht.As.minus.arm.train.list, predict, newdata = ht.As.minus.arm.test[,-1], type = "raw") #newdata = ht.As.minus.arm.test[,-1]

ht.As.minus.arm.test.results <- data.frame(id = "ht.As.minus.arm", section = "arm", subset = "test", condition = "Asymptomatic", 
                                           subinfection = "minus", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"),
                                           accuracy = c(confusionMatrix(ht.As.minus.arm.train.list.preds$rf, ht.As.minus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.minus.arm.train.list.preds$AdaBoost.M1, ht.As.minus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.minus.arm.train.list.preds$svmRadial, ht.As.minus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(ht.As.minus.arm.train.list.preds$nnet, ht.As.minus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                        confusionMatrix(predict(ht.As.minus.arm.train.ensemble, ht.As.minus.arm.test[,-1]), ht.As.minus.arm.test$condition, positive = "M")$overall["Accuracy"][[1]]),
                                           ROC = c(colAUC(predict(ht.As.minus.arm.train.list$rf, newdata = ht.As.minus.arm.test[,-1], type = "prob"), ht.As.minus.arm.test$condition)[[1]],
                                                   colAUC(predict(ht.As.minus.arm.train.list$AdaBoost.M1, newdata = ht.As.minus.arm.test[,-1], type = "prob"), ht.As.minus.arm.test$condition)[[1]],
                                                   colAUC(predict(ht.As.minus.arm.train.list$svmRadial, newdata = ht.As.minus.arm.test[,-1], type = "prob"), ht.As.minus.arm.test$condition)[[1]],
                                                   colAUC(predict(ht.As.minus.arm.train.list$nnet, newdata = ht.As.minus.arm.test[,-1], type = "prob"), ht.As.minus.arm.test$condition)[[1]],
                                                   colAUC(predict(ht.As.minus.arm.train.ensemble, newdata = ht.As.minus.arm.test[,-1], type = "prob"), ht.As.minus.arm.test$condition)[[1]]),
                                           sensitivity = c(confusionMatrix(ht.As.minus.arm.train.list.preds$rf, ht.As.minus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                           confusionMatrix(ht.As.minus.arm.train.list.preds$AdaBoost.M1, ht.As.minus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                           confusionMatrix(ht.As.minus.arm.train.list.preds$svmRadial, ht.As.minus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                           confusionMatrix(ht.As.minus.arm.train.list.preds$nnet, ht.As.minus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                           confusionMatrix(predict(ht.As.minus.arm.train.ensemble, ht.As.minus.arm.test[,-1]), ht.As.minus.arm.test$condition, positive = "M")$byClass["Sensitivity"][[1]]),
                                           specificity = c(confusionMatrix(ht.As.minus.arm.train.list.preds$rf, ht.As.minus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                           confusionMatrix(ht.As.minus.arm.train.list.preds$AdaBoost.M1, ht.As.minus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                           confusionMatrix(ht.As.minus.arm.train.list.preds$svmRadial, ht.As.minus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                           confusionMatrix(ht.As.minus.arm.train.list.preds$nnet, ht.As.minus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                           confusionMatrix(predict(ht.As.minus.arm.train.ensemble, ht.As.minus.arm.test[,-1]), ht.As.minus.arm.test$condition, positive = "M")$byClass["Specificity"][[1]]),
                                           best = c(paste(ht.As.minus.arm.train.list.varImp.rf.bestFive[, 1], collapse = ", ", sep = ""), 
                                                    paste(ht.As.minus.arm.train.list.varImp.ada.bestFive[, 1], collapse = ", ", sep = ""),
                                                    paste(ht.As.minus.arm.train.list.varImp.svm.bestFive[, 1], collapse = ", ", sep = ""),
                                                    paste(ht.As.minus.arm.train.list.varImp.nnet.bestFive[, 1], collapse = ", ", sep = ""),
                                                    paste(ht.As.minus.arm.train.list.varImp.ensemble.bestFive[, 1], collapse = ", ", sep = ""))
)


# ht.As.minus.foot.train ####
## Control list ####
set.seed(2019)
ht.As.minus.foot.train.ctrl <- trainControl(method = "repeatedcv",
                                            repeats = 5, number = 10, 
                                            returnResamp = "all", savePredictions = "all", classProbs = TRUE, 
                                            summaryFunction = twoClassSummary, allowParallel = TRUE,
                                            index = createResample(ht.As.minus.foot.train$condition))


## Build models within caretList ####
set.seed(2019)
ht.As.minus.foot.train.list <- caretList(x = ht.As.minus.foot.train[,-1], y = as.factor(ht.As.minus.foot.train$condition),
                                         trControl = ht.As.minus.foot.train.ctrl,
                                         #methodList = c("rf", "nnet"),
                                         metric = "ROC",
                                         preProc = c("center", "scale"),
                                         trace = FALSE,
                                         tuneList = list(
                                             rf = caretModelSpec(method = "rf", ntree = 3000,
                                                                 tuneGrid = expand.grid(mtry = c(3, 5, 10, 15, 21))),
                                             AdaBoost.M1 = caretModelSpec(method = "AdaBoost.M1",
                                                                          tuneGrid = expand.grid(mfinal = 1000, maxdepth = c(1, 3), coeflearn = "Breiman")),
                                             svmRadial = caretModelSpec(method = "svmRadial",
                                                                        tuneGrid = expand.grid(sigma = c(.01, .015, 0.2), C = c(0.75, 0.9, 1, 1.1, 1.25))),
                                             nnet = caretModelSpec(method = "nnet", MaxNWts = 481,
                                                                   tuneGrid = expand.grid(size = c(3, 5, 10, 20), 
                                                                                          decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))))
)

## Ensemble the caretList models using AdaBag in caretStack ####
set.seed(2019)
ht.As.minus.foot.train.ensemble <- caretStack(
    ht.As.minus.foot.train.list,
    method = "AdaBag", 
    metric = "ROC",
    trControl = trainControl(
        method = "boot",
        number = 20,
        savePredictions = "final",
        classProbs = TRUE,
        summaryFunction = twoClassSummary
    )
)

## Build performance table for training set ####
ht.As.minus.foot.train.results <- data.frame(id = "ht.As.minus.foot", section = "foot", subset = "train", condition = "Asymptomatic", 
                                             subinfection = "minus", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"), 
                                             accuracy = c(confusionMatrix(ht.As.minus.foot.train.list$rf$pred$pred, ht.As.minus.foot.train.list$rf$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                          confusionMatrix(ht.As.minus.foot.train.list$AdaBoost.M1$pred$pred, ht.As.minus.foot.train.list$AdaBoost.M1$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                          confusionMatrix(ht.As.minus.foot.train.list$svmRadial$pred$pred, ht.As.minus.foot.train.list$svmRadial$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                          confusionMatrix(ht.As.minus.foot.train.list$nnet$pred$pred, ht.As.minus.foot.train.list$nnet$pred$obs, positive = "M")$overall["Accuracy"][[1]],
                                                          confusionMatrix(ht.As.minus.foot.train.ensemble$ens_model$pred$pred, ht.As.minus.foot.train.ensemble$ens_model$pred$obs, positive = "M")$overall["Accuracy"][[1]]),
                                             ROC = c(getTrainPerf(ht.As.minus.foot.train.list$rf)$TrainROC[[1]],
                                                     getTrainPerf(ht.As.minus.foot.train.list$AdaBoost.M1)$TrainROC[[1]],
                                                     getTrainPerf(ht.As.minus.foot.train.list$svmRadial)$TrainROC[[1]],
                                                     getTrainPerf(ht.As.minus.foot.train.list$nnet)$TrainROC[[1]],
                                                     getTrainPerf(ht.As.minus.foot.train.ensemble$ens_model)$TrainROC[[1]]),
                                             sensitivity = c(getTrainPerf(ht.As.minus.foot.train.list$rf)$TrainSens[[1]],
                                                             getTrainPerf(ht.As.minus.foot.train.list$AdaBoost.M1)$TrainSens[[1]],
                                                             getTrainPerf(ht.As.minus.foot.train.list$svmRadial)$TrainSens[[1]],
                                                             getTrainPerf(ht.As.minus.foot.train.list$nnet)$TrainSens[[1]],
                                                             getTrainPerf(ht.As.minus.foot.train.ensemble$ens_model)$TrainSens[[1]]),
                                             specificity = c(getTrainPerf(ht.As.minus.foot.train.list$rf)$TrainSpec[[1]],
                                                             getTrainPerf(ht.As.minus.foot.train.list$AdaBoost.M1)$TrainSpec[[1]],
                                                             getTrainPerf(ht.As.minus.foot.train.list$svmRadial)$TrainSpec[[1]],
                                                             getTrainPerf(ht.As.minus.foot.train.list$nnet)$TrainSpec[[1]],
                                                             getTrainPerf(ht.As.minus.foot.train.ensemble$ens_model)$TrainSpec[[1]]),
                                             best = NA
)


## Variable importance using DALEX ####
set.seed(2019)
ht.As.minus.foot.train.list.varImp.rf <- replicate(20, feature_importance(explain(ht.As.minus.foot.train.list$rf, label = "rf", data = ht.As.minus.foot.train[,-1], y = as.numeric(ht.As.minus.foot.train$condition))), simplify = FALSE)
ht.As.minus.foot.train.list.varImp.rf <- ht.As.minus.foot.train.list.varImp.rf %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.rf = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.rf))

set.seed(2019)
ht.As.minus.foot.train.list.varImp.ada <- replicate(20, feature_importance(explain(ht.As.minus.foot.train.list$AdaBoost.M1, label = "ada", data = ht.As.minus.foot.train[,-1], y = as.numeric(ht.As.minus.foot.train$condition))), simplify = FALSE)
ht.As.minus.foot.train.list.varImp.ada <- ht.As.minus.foot.train.list.varImp.ada %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ada = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ada))

set.seed(2019)
ht.As.minus.foot.train.list.varImp.svm <- replicate(20, feature_importance(explain(ht.As.minus.foot.train.list$svmRadial, label = "svm", data = ht.As.minus.foot.train[,-1], y = as.numeric(ht.As.minus.foot.train$condition))), simplify = FALSE)
ht.As.minus.foot.train.list.varImp.svm <- ht.As.minus.foot.train.list.varImp.svm %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.svm = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.svm))

set.seed(2019)
ht.As.minus.foot.train.list.varImp.nnet <- replicate(20, feature_importance(explain(ht.As.minus.foot.train.list$nnet, label = "nnet", data = ht.As.minus.foot.train[,-1], y = as.numeric(ht.As.minus.foot.train$condition))), simplify = FALSE)
ht.As.minus.foot.train.list.varImp.nnet <- ht.As.minus.foot.train.list.varImp.nnet %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.nnet = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.nnet))

set.seed(2019)
ht.As.minus.foot.train.list.varImp.ensemble <- replicate(20, feature_importance(explain(ht.As.minus.foot.train.ensemble, label = "ensemble", data = ht.As.minus.foot.train[,-1], y = as.numeric(ht.As.minus.foot.train$condition))), simplify = FALSE)
ht.As.minus.foot.train.list.varImp.ensemble <- ht.As.minus.foot.train.list.varImp.ensemble %>% 
    reduce(left_join, by = "variable") %>% 
    select(variable, contains("dropout_loss")) %>% 
    mutate(loss.ensemble = rowMeans(select(., contains("dropout_loss")))) %>% 
    select(-contains("dropout_loss")) %>% 
    arrange(desc(loss.ensemble))

## Merge variable importances from all models in caretList into one dataframe
ht.As.minus.foot.train.list.varImp.minus <- ht.As.minus.foot.train.list.varImp.rf %>% 
    left_join(ht.As.minus.foot.train.list.varImp.ada, by = "variable") %>% 
    left_join(ht.As.minus.foot.train.list.varImp.svm, by = "variable") %>% 
    left_join(ht.As.minus.foot.train.list.varImp.nnet, by = "variable") %>% 
    left_join(ht.As.minus.foot.train.list.varImp.ensemble, by = "variable")


### Get five most important variables per model to add it to the ht.As.minus.foot.test.results
ht.As.minus.foot.train.list.varImp.rf.bestFive <- ht.As.minus.foot.train.list.varImp.rf %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.minus.foot.train.list.varImp.ada.bestFive <- ht.As.minus.foot.train.list.varImp.ada %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.minus.foot.train.list.varImp.svm.bestFive <- ht.As.minus.foot.train.list.varImp.svm %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.minus.foot.train.list.varImp.nnet.bestFive <- ht.As.minus.foot.train.list.varImp.nnet %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)

ht.As.minus.foot.train.list.varImp.ensemble.bestFive <- ht.As.minus.foot.train.list.varImp.ensemble %>% 
    filter(str_detect(variable, pattern = "_", negate = TRUE)) %>% 
    slice(1:5) %>% 
    select(variable)



## Make predictions in test set ####
## Make predictions in each model of caretList by using lapply
ht.As.minus.foot.train.list.preds <- lapply(ht.As.minus.foot.train.list, predict, newdata = ht.As.minus.foot.test[,-1], type = "raw") #newdata = ht.As.minus.foot.test[,-1]

ht.As.minus.foot.test.results <- data.frame(id = "ht.As.minus.foot", section = "foot", subset = "test", condition = "Asymptomatic", 
                                            subinfection = "minus", model = c("rf", "AdaBoost", "svm", "nnet", "ensemble"),
                                            accuracy = c(confusionMatrix(ht.As.minus.foot.train.list.preds$rf, ht.As.minus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.minus.foot.train.list.preds$AdaBoost.M1, ht.As.minus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.minus.foot.train.list.preds$svmRadial, ht.As.minus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(ht.As.minus.foot.train.list.preds$nnet, ht.As.minus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]],
                                                         confusionMatrix(predict(ht.As.minus.foot.train.ensemble, ht.As.minus.foot.test[,-1]), ht.As.minus.foot.test$condition, positive = "M")$overall["Accuracy"][[1]]),
                                            ROC = c(colAUC(predict(ht.As.minus.foot.train.list$rf, newdata = ht.As.minus.foot.test[,-1], type = "prob"), ht.As.minus.foot.test$condition)[[1]],
                                                    colAUC(predict(ht.As.minus.foot.train.list$AdaBoost.M1, newdata = ht.As.minus.foot.test[,-1], type = "prob"), ht.As.minus.foot.test$condition)[[1]],
                                                    colAUC(predict(ht.As.minus.foot.train.list$svmRadial, newdata = ht.As.minus.foot.test[,-1], type = "prob"), ht.As.minus.foot.test$condition)[[1]],
                                                    colAUC(predict(ht.As.minus.foot.train.list$nnet, newdata = ht.As.minus.foot.test[,-1], type = "prob"), ht.As.minus.foot.test$condition)[[1]],
                                                    colAUC(predict(ht.As.minus.foot.train.ensemble, newdata = ht.As.minus.foot.test[,-1], type = "prob"), ht.As.minus.foot.test$condition)[[1]]),
                                            sensitivity = c(confusionMatrix(ht.As.minus.foot.train.list.preds$rf, ht.As.minus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                            confusionMatrix(ht.As.minus.foot.train.list.preds$AdaBoost.M1, ht.As.minus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                            confusionMatrix(ht.As.minus.foot.train.list.preds$svmRadial, ht.As.minus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                            confusionMatrix(ht.As.minus.foot.train.list.preds$nnet, ht.As.minus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]],
                                                            confusionMatrix(predict(ht.As.minus.foot.train.ensemble, ht.As.minus.foot.test[,-1]), ht.As.minus.foot.test$condition, positive = "M")$byClass["Sensitivity"][[1]]),
                                            specificity = c(confusionMatrix(ht.As.minus.foot.train.list.preds$rf, ht.As.minus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                            confusionMatrix(ht.As.minus.foot.train.list.preds$AdaBoost.M1, ht.As.minus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                            confusionMatrix(ht.As.minus.foot.train.list.preds$svmRadial, ht.As.minus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                            confusionMatrix(ht.As.minus.foot.train.list.preds$nnet, ht.As.minus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]],
                                                            confusionMatrix(predict(ht.As.minus.foot.train.ensemble, ht.As.minus.foot.test[,-1]), ht.As.minus.foot.test$condition, positive = "M")$byClass["Specificity"][[1]]),
                                            best = c(paste(ht.As.minus.foot.train.list.varImp.rf.bestFive[, 1], collapse = ", ", sep = ""), 
                                                     paste(ht.As.minus.foot.train.list.varImp.ada.bestFive[, 1], collapse = ", ", sep = ""),
                                                     paste(ht.As.minus.foot.train.list.varImp.svm.bestFive[, 1], collapse = ", ", sep = ""),
                                                     paste(ht.As.minus.foot.train.list.varImp.nnet.bestFive[, 1], collapse = ", ", sep = ""),
                                                     paste(ht.As.minus.foot.train.list.varImp.ensemble.bestFive[, 1], collapse = ", ", sep = ""))
)


# Save workspace ####
save.image("As_All_plus_minus_Results_20190910.RData")