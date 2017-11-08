

url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

download.file(url,"data/training.csv")


url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(url,"data/testing.csv")

trainData<-read.csv("data/training.csv")

testData<-read.csv("data/testing.csv")

trainData0<-read.csv("data/training.csv",nrows=500)

colnames(trainData0)


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

trainDataP<-createDataPartition(trainDataX$classe,p=0.01,list=FALSE)



trainDataPt<-trainDataX[trainDataP,]

# trainSVD<-svd(trainDataPt)

modelFit<-train(classe~.,data=trainDataPt,method="rf")

res<-predict(modelFit,testData)


folds<-createFolds(trainDataX$classe,k=10,list=TRUE)




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
