url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url,"data/training.csv")
url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url,"data/testing.csv")
trainData<-read.csv("data/training.csv")
testData<-read.csv("data/testing.csv")
summary(trainData)

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

end_time - start_time
stopCluster(cluster)
registerDoSEQ()

res<-predict(modelFit,validateDataPt)

confusionMatrix(res,validateDataPt$classe)

saveRDS(modelFit,"warableModel.rds")



library(randomForest)
model<-readRDS(file = "warableModel.rds")

res1<-predict(model,testData)

#-------------------------------------------------------------------

trainDataP<-createDataPartition(trainDataX$classe,p=0.01,list=FALSE)

trainDataPt<-trainDataX[trainDataP,]

validateDataPt<-trainDataX[-trainDataP,]

dim(trainDataPt)[1]

timeChart<-data.frame()


set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf")
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[1,1]<-"rf"
timeChart[1,2]<-"None"
timeChart[1,3]<-dim(trainDataPt)[1]
timeChart[1,4]<-round(end_time-start_time,2)
timeChart[1,5]<-round(cm$overall[1],3)

names(timeChart)<-c("Method","Pre-Processing/Parameters","No.Records",
                 "Duration","Accuracy")

timeChart

#-----------------------------------------------

set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("scale","center"))
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[2,1]<-"rf"
timeChart[2,2]<-"scale,center"
timeChart[2,3]<-dim(trainDataPt)[1]
timeChart[2,4]<-round(end_time-start_time,2)
timeChart[2,5]<-round(cm$overall[1],3)



timeChart

#---------------------------------------------------

set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("pca"))
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[3,1]<-"rf"
timeChart[3,2]<-"pca"
timeChart[3,3]<-dim(trainDataPt)[1]
timeChart[3,4]<-round(end_time-start_time,2)
timeChart[3,5]<-round(cm$overall[1],3)


timeChart

##------------------------------------------------
set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("center","scale","pca"))
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[4,1]<-"rf"
timeChart[4,2]<-"center,scale,pca"
timeChart[4,3]<-dim(trainDataPt)[1]
timeChart[4,4]<-round(end_time-start_time,2)
timeChart[4,5]<-round(cm$overall[1],3)


timeChart

#----------------------------------------------------
trainDataP<-createDataPartition(trainDataX$classe,p=0.05,list=FALSE)

trainDataPt<-trainDataX[trainDataP,]

validateDataPt<-trainDataX[-trainDataP,]


#-------------------------------------------------------------
set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("scale","center"))
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[5,1]<-"rf"
timeChart[5,2]<-"scale,center"
timeChart[5,3]<-dim(trainDataPt)[1]
timeChart[5,4]<-round(end_time-start_time,2)
timeChart[5,5]<-round(cm$overall[1],3)

timeChart


#----------------------------------------------------------------
set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("pca"))
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[6,1]<-"rf"
timeChart[6,2]<-"pca"
timeChart[6,3]<-dim(trainDataPt)[1]
timeChart[6,4]<-round(end_time-start_time,2)
timeChart[6,5]<-round(cm$overall[1],3)

timeChart

#----------------------------------------------------------------
set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("pca"))
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[6,1]<-"rf"
timeChart[6,2]<-"pca"
timeChart[6,3]<-dim(trainDataPt)[1]
timeChart[6,4]<-round(end_time-start_time,2)
timeChart[6,5]<-round(cm$overall[1],3)

timeChart

#--------------------------------------------------------------
set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("scale","center","pca"))
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[7,1]<-"rf"
timeChart[7,2]<-"center, scale, pca"
timeChart[7,3]<-dim(trainDataPt)[1]
timeChart[7,4]<-round(end_time-start_time,2)
timeChart[7,5]<-round(cm$overall[1],3)

timeChart

#--------------------------------------------------------------
trainDataP<-createDataPartition(trainDataX$classe,p=0.1,list=FALSE)

trainDataPt<-trainDataX[trainDataP,]

validateDataPt<-trainDataX[-trainDataP,]
#-----------------------------------------------------------------
set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("scale","center"))
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[8,1]<-"rf"
timeChart[8,2]<-"scale,center"
timeChart[8,3]<-dim(trainDataPt)[1]
timeChart[8,4]<-round(end_time-start_time,2)
timeChart[8,5]<-round(cm$overall[1],3)

timeChart

#----------------------------------------------------------------------
set.seed(100)
start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("pca"))
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[9,1]<-"rf"
timeChart[9,2]<-"pca"
timeChart[9,3]<-dim(trainDataPt)[1]
timeChart[9,4]<-round(end_time-start_time,2)
timeChart[9,5]<-round(cm$overall[1],3)


timeChart
#------------------------------------------------------------------------
set.seed(100)
fitControl<-trainControl(method="oob")

start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",
                preProcess=c("pca"),
                trainControl=fitControl)
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[10,1]<-"rf"
timeChart[10,2]<-"pca with oob resampling"
timeChart[10,3]<-dim(trainDataPt)[1]
timeChart[10,4]<-round(end_time-start_time,2)
timeChart[10,5]<-round(cm$overall[1],3)

timeChart

plot(modelFit$finalModel)

#------------------------------------------------------------------------

set.seed(100)
fitControl<-trainControl(method="oob")

start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",ntree = 200,
                preProcess=c("pca"),
                trainControl=fitControl)
end_time<-Sys.time()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[11,1]<-"rf"
timeChart[11,2]<-"pca, oob resampling, 200 trees"
timeChart[11,3]<-dim(trainDataPt)[1]
timeChart[11,4]<-round(end_time-start_time,2)
timeChart[11,5]<-round(cm$overall[1],3)

timeChart

#--------------------------------------------------------------------------
library(parallel)
library(doParallel)

set.seed(100)
fitControl<-trainControl(method="oob",allowParallel = TRUE)

cluster <- makeCluster(detectCores()-1) # convention to leave 1 core for OS
registerDoParallel(cluster)

start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",ntree = 200,
                preProcess=c("pca"),
                trainControl=fitControl)
end_time<-Sys.time()

stopCluster(cluster)
registerDoSEQ()

res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)

end_time-start_time

timeChart[12,1]<-"rf"
timeChart[12,2]<-"pca,oob, 200 trees, pp"
timeChart[12,3]<-dim(trainDataPt)[1]
timeChart[12,4]<-round(end_time-start_time,2)
timeChart[12,5]<-round(cm$overall[1],3)

timeChart

saveRDS(modelFit,"wearableModel2000.rds")
#-------------------------------------------------------------------------
trainDataP<-createDataPartition(trainDataX$classe,p=0.8,list=FALSE)

trainDataPt<-trainDataX[trainDataP,]

validateDataPt<-trainDataX[-trainDataP,]

set.seed(100)
fitControl<-trainControl(method="oob")



start_time<-Sys.time()
modelFit<-train(classe~.,data=trainDataPt,method="rf",ntree = 200,
                preProcess=c("pca"),
                trainControl=fitControl)
end_time<-Sys.time()


res<-predict(modelFit,validateDataPt)

cm<-confusionMatrix(res,validateDataPt$classe)


timeChart[13,1]<-"rf"
timeChart[13,2]<-"pca,oob resampling, 200 trees"
timeChart[13,3]<-dim(trainDataPt)[1]
timeChart[13,4]<-round(end_time-start_time,2)
timeChart[13,5]<-round(cm$overall[1],3)

timeChart

cm

saveRDS(modelFit,"wearableModel.rds")

saveRDS(timeChart,"timechart.rds")

timeChart

