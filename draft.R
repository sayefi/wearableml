

url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

download.file(url,"data/training.csv")


url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(url,"data/testing.csv")

trainData<-read.csv("data/training.csv")

testData<-read.csv("data/testing.csv")

# trainData0<-read.csv("data/training.csv",nrows=500)
# 
# colnames(trainData0)


reqdCols<-c("roll_belt",
            "pitch_belt",
            "yaw_belt",
            "total_accel_belt",
            "gyros_belt_x",
            "gyros_belt_y",
            "gyros_belt_z",
            "accel_belt_x",
            "accel_belt_y",
            "accel_belt_z",
            "magnet_belt_x",
            "magnet_belt_y",
            "magnet_belt_z",
            "roll_arm",
            "pitch_arm",
            "yaw_arm",
            "total_accel_arm",
            "gyros_arm_x",
            "gyros_arm_y",
            "gyros_arm_z",
            "accel_arm_x",
            "accel_arm_y",
            "accel_arm_z",
            "magnet_arm_x",
            "magnet_arm_y",
            "magnet_arm_z",
            "roll_dumbbell",
            "pitch_dumbbell",
            "yaw_dumbbell",
            "gyros_dumbbell_x",
            "gyros_dumbbell_y",
            "gyros_dumbbell_z",
            "accel_dumbbell_x",
            "accel_dumbbell_y",
            "accel_dumbbell_z",
            "magnet_dumbbell_x",
            "magnet_dumbbell_y",
            "magnet_dumbbell_z",
            "roll_forearm",
            "pitch_forearm",
            "yaw_forearm",
            "gyros_forearm_x",
            "gyros_forearm_y",
            "gyros_forearm_z",
            "accel_forearm_x",
            "accel_forearm_y",
            "accel_forearm_z",
            "magnet_forearm_x",
            "magnet_forearm_y",
            "magnet_forearm_z",
            "classe")



library(dplyr)

library(caret)

trainDataX<-select(trainData,reqdCols)

compCases<-complete.cases(trainDataX)

trainDataX<-trainDataX[compCases,]

summary(trainDataX)

## Creating folds
##------------------------------------------------

folds<-createFolds(trainDataX$classe,k=10,list=TRUE)

trainDataFold<-trainDataX[-folds$Fold01,]
trainDataValidate<-trainDataX[folds$Fold01,]

str(trainDataFold)

##--------------------------------------------------

trainDataP<-createDataPartition(trainDataX$classe,p=0.01,list=FALSE)

trainDataPt<-trainDataX[trainDataP,]

validateDataPt<-trainDataX[-trainDataP,]

str(trainDataPt)

# trainSVD<-svd(trainDataPt)

modelFit<-train(classe~.,data=trainDataPt,method="rf")

res<-predict(modelFit,validateDataPt)

confusionMatrix(res,validateDataPt$classe)

# Accuracy : 0.9553 

#-----------------------------------------------------

modelFit<-train(classe~.,data=trainDataPt,method="rf",
                              preProcess=c("center","scale"))

modelFit
res<-predict(modelFit,validateDataPt)

confusionMatrix(res,validateDataPt$classe)

# Accuracy : 0.9538 

#----------------------------------------------------

fitControl<-trainControl(method="oob")

modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("center","scale"),
                trainControl=fitControl)

modelFit
res<-predict(modelFit,validateDataPt)

confusionMatrix(res,validateDataPt$classe)

# Accuracy : 0.9554

#----------------------------------------------------------
## seeding up

install.packages("doParallel")

library(parallel)
library(doParallel)

plot(modelFit$finalModel)


cluster <- makeCluster(detectCores()-1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl<-trainControl(method="oob",allowParallel = TRUE)

start_time <- Sys.time()

modelFit<-train(classe~.,data=trainDataPt,method="rf",ntree = 100,
                preProcess=c("center","scale","pca"),
                trainControl=fitControl)

end_time <- Sys.time()

start_time
end_time
stopCluster(cluster)
registerDoSEQ()

res<-predict(modelFit,validateDataPt)

confusionMatrix(res,validateDataPt$classe)

# Accuracy : 0.8529
#------------------------------------------------------------------
## Now use the full dataset for training

trainDataP<-createDataPartition(trainDataX$classe,p=0.8,list=FALSE)

trainDataPt<-trainDataX[trainDataP,]

validateDataPt<-trainDataX[-trainDataP,]

str(trainDataPt)

cluster <- makeCluster(detectCores()-1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl<-trainControl(method="oob",allowParallel = TRUE)

start_time <- Sys.time()


modelFit<-train(classe~.,data=trainDataPt,method="rf",ntree = 200,
                preProcess=c("center","scale","pca"),
                trainControl=fitControl)

end_time <- Sys.time()


end_time <- Sys.time()

start_time
end_time
stopCluster(cluster)
registerDoSEQ()

res<-predict(modelFit,validateDataPt)

confusionMatrix(res,validateDataPt$classe)

saveRDS(modelFit,"warableModel.rds")

#Accuracy : 0.9845





trainDataPtx<-trainDataPt
trainDataPtx$classe<-as.numeric(trainDataPtx$classe)

x<-as.matrix(trainDataPtx)

x<-scale(x,center = TRUE)

trainSVD<-svd(trainDataPtx)

plot(trainSVD$d)


#-------------------------------------------------------

preProc<-preProcess(trainDataPt,method="pca",thrash)

trainPCA<-predict(preProc,trainDataPt)

modelFit<-train(classe~.,data=trainPCA,model="rf")

validatePCA<-predict(preProc,validateDataPt)

res<-predict(modelFit,validatePCA)

confusionMatrix(res,validateDataPt$classe)




set.seed(3523)

library(AppliedPredictiveModeling)
library(glmnet)

dim(trainDataPt)
str(trainDataPt[,1:50])

x<-as.matrix(trainDataPt[,1:50])

y<-as.numeric(trainDataPt$classe)

fit<-glmnet(x,y,alpha=1)

plot(fit)
plot(fit, xvar="lambda", label=TRUE)


impCols<-c(1,2,30,44,18,19,31,7,12,9,5,20,32,6,42,51)
#--------------------------------------------------------

trainDataRidge<-trainDataPt[,impCols]

str(trainDataRidge)

modelFit<-train(classe~.,data=trainDataRidge,method="rf")

res<-predict(modelFit,validateDataPt)

confusionMatrix(res,validateDataPt$classe)

#-----------------------------



modelFit<-train(classe~.,data=trainDataX[folds$Fold01,],method="rf")

res<-predict(modelFit,trainDataX[folds$Fold02,])

confusionMatrix(res,trainDataX[folds$Fold02,]$classe)


## accuracy : 0.9526


modelFit<-train(classe~.,data=trainDataX[folds$Fold01,],method="rf",
                preProcess=c("pca"))

res<-predict(modelFit,trainDataX[folds$Fold02,])

confusionMatrix(res,trainDataX[folds$Fold02,]$classe)


## Accuracy : 0.8532

modelFit<-train(classe~.,data=trainDataX[folds$Fold01,],method="rf",
                preProcess=c("center","scale"))

res<-predict(modelFit,trainDataX[folds$Fold02,])

confusionMatrix(res,trainDataX[folds$Fold02,]$classe)


## Accuracy : 0.9541

res<-predict(modelFit,trainDataX[folds$Fold03,])

confusionMatrix(res,trainDataX[folds$Fold03,]$classe)


## Accuracy : 0.9546

res<-predict(modelFit,trainDataX[folds$Fold04,])

confusionMatrix(res,trainDataX[folds$Fold04,]$classe)


## -----


trainDataP<-createDataPartition(trainDataX$classe,p=0.8,list=FALSE)

trainDataPartitioned<-trainDataX[trainDataP,]
trainDataValidate<-trainDataX[-trainDataP,]

dim(trainDataPartitioned)
dim(trainDataValidate)

modelFit<-train(classe~.,data=trainDataPartitioned,method="rf",
                preProcess=c("center","scale"))

res<-predict(modelFit,trainDataValidate)

confusionMatrix(res,trainDataValidate$classe)


str(testDataX)


colsNotNull<-NULL

for(i in 1:158) 
{
     if(is.na(trainData[1,i])==FALSE)
     {
          colsNotNull<-rbind(colsNotNull,colnames(trainData)[i])
     }
     
}

trainDataX<-trainData[,colsNotNull]

apply(trainData,2, sum)



impute::impute.knn(trainData)


trainDataClean<-complete.cases(trainData)

trainDataN<-trainData[trainDataClean,]

trainDataSVD<-svd(trainDataN)

trainDataSVD$



dim(trainDataN)

trainDataSVD<-svd(trainDataN)



trainDataN<-trainDataN[]

str(trainDataN)


trainData<-trainData[,-c(1,2)]

trainData$classe<-factor(trainData$classe)

# trainData<-trainData[1:10,]

length(trainData)

traincolsum<-colSums(trainData)

colsNotNull<-NULL

for(i in 1:158) 
{
     #if(is.na(trainData[1,i]))
     if(trainData[1,i]=="")
     {
          print(i)
          colsNotNull<-rbind(colsNotNull,colnames(trainData)[i])
     }
          
        
}
     ==FALSE||(trainData[1,i]==""))
        
colnames(trainData)[15]
trainData[1,15]==""
        
colsNotNull
        
trainData<-trainData[,colsNotNull]

dim(trainData)

# sum(is.na(trainData$classe))


trainDataSVD<-svd(trainData)

library(caret)

folds<-createFolds(y=trainData$classe,k=20,list=TRUE,returnTrain = T)

sapply(folds,length)

str(folds)

folds[[1]][1:10]

modelFit<-train(classe~.,data=trainData,method="rf")

modelFit

summary(trainData)
