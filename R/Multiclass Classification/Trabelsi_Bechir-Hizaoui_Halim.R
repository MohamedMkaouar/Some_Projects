################################################################################################
#                                  Projet STA203                                               #
#                          Hizoui Halim   -  Trabelsi bechir                                   # 
################################################################################################
library(e1071)
library(corrplot)
library(MASS)
require(ElemStatLearn)
require(class)
require(rpart)
require(rpart.plot)
require(caret)
require(doParallel)
require(randomForest)
require(xgboost)
library(keras)
library(dplyr)

### Partie I :

# Analyse#s descriptives : 
music<-read.table("music.txt",sep=';',header = TRUE)
musicM<-read.table("music.txt",sep=';',header = TRUE)
musicM$PAR_SC_V<-log(music$PAR_SC_V)
musicM$PAR_ASC_V<-log(music$PAR_ASC_V)
music1<-music[1:147]
music2<-music[168:192]
music<-data.frame(music1,music2)
#GENRE=ifelse(music$GENRE=="Jazz",1,0)
propotion<-function(X){
  j=0
  k=0
  for(i in 1:length(X)){
    if(X[i]==1){
      j=j+1
    }
    else{ k=k+1}
    
  }
  propJazz<-j/(j+k)
  propClassic<-k/(j+k)
  res<-c(propJazz,propClassic)
  names(res)<-c("propJAzz","PropClassics")
  return(res)
}

getCor<-function(X)
{
  C<-cor(X[,-length(X)])
  nrow(C)
  I<-c()
  J<-c()
  for(i in 1:nrow(C) ){
    for(j in 1:ncol(C)){
      if((abs(C[i,j])>0.99) & (i!=j)){
        I<-c(I,i)
        J<-c(J,j)
      }
    }
  }
  res<-rbind(I,J)
  
  return(res)
}
GenreBin<-ifelse(music$GENRE=="Jazz",1,0)
Props<-propotion(GenreBin)
Props
##bonne répartition 50% 50% 
CorIdx<-getCor(music)
#multicollinearity 

#perte information ou perte precision 

#cleaning data
CorIdx<-as.vector(CorIdx)
CordIdx<-unique(CorIdx)
music<-music[-CordIdx]
# for(i in 1:length(CordIdx)){
#     if(i==1){
#     music<-music[,-CorIdx[i]]
#     }
#     else{
#       music<-music[,-(CordIdx[i]-1)]
#     }
# }
music<-music[- grep("^PAR_ASE_M$", colnames(music))]
music<-music[-grep("^PAR_ASE_MV$", colnames(music))]
music<-music[-grep("^PAR_SFM_M$", colnames(music))]
music<-music[-grep("^PAR_SFM_MV$", colnames(music))]


C<-cor(as.matrix(music[-length(music)]))
corrplot(C, tl.cex=0.1)



###############################Models##############################################


GENRE<-music2$GENRE
set.seed(103)
train=sample(c(TRUE,FALSE),6447,rep=TRUE,prob=c(2/3,1/3))


#################
## mod0
################
df<-c("PAR_TC", "PAR_SC", "PAR_SC_V", "PAR_ASE_M",
      "PAR_ASE_MV", "PAR_SFM_M","PAR_SFM_MV","GENRE")
musicM.tr<-musicM[train,]
musicM.tr<-musicM.tr[df]
musicM.test<-musicM[!train,]
musicM.test<-musicM.test[df]
Mod0<-glm(GENRE~.,data = musicM.tr ,family = binomial)
Mod0.probs=predict(Mod0,newdata = musicM.test,type="response")
Mod0.pred=ifelse( Mod0.probs>.5,"Jazz","Classic")
Mod0.probsTr=predict(Mod0,newdata = musicM.tr,type="response")
Mod0.predtr=ifelse( Mod0.probsTr>.5,"Jazz","Classic")
table(Mod0.pred,musicM.test$GENRE)
Mod0TestAccuracy<-sum(diag(table(Mod0.pred,musicM.test$GENRE)))/length(musicM.test$GENRE)
Mod0TestpredictionError<-1-Mod0TestAccuracy
Mod0TrainAccuracy<-sum(diag(table(Mod0.predtr,musicM.tr$GENRE)))/length(musicM.tr$GENRE)
Mod0TrainpredictionError<-1-Mod0TrainAccuracy
## (TP+TN)/(total)= 0.7368182
seuil=seq(0,1,0.01)
n<-nrow(musicM.test)
# calcul de la table de classification pour différents seuils
sap=sapply(seq(0.03,0.99,0.01),
           function(x){c(table(Y=musicM.test$GENRE,pred=Mod0.probs>x))})

GENREn<-ifelse(musicM.test$GENRE=="Jazz",1,0)
# ajout des cas spéciaux s=0 et s=1
tc=data.frame(rbind(c(0,0,n-sum(GENREn), sum(GENREn)),
                    c(0,0,n-sum(GENREn), sum(GENREn)),c(0,0,n-sum(GENREn), sum(GENREn)),t(sap),
                    c( n-sum(GENREn),sum(GENREn),0,0))) 
names(tc)=c("VN","FN","FP","VP")

# le plot
matplot(seuil,cbind(tc$FN,tc$FP,tc$FN+tc$FP),
        type="l",ylab="erreur",
        lty=c(2,3,1),col=c(2,3,1))




################
#### modT
################


musicT.tr<-music[train,]
musicT.test<-music[!train,]

ModT<-glm(GENRE~.,data = musicT.tr ,family = binomial)
ModT.probs=predict(ModT,newdata = musicT.test,type="response")
ModT.probsTr=predict(ModT,newdata = musicT.tr,type="response")
## seuil de Bayes 0.5
ModT.pred=ifelse( ModT.probs>.5,"Jazz","Classic") 
ModT.predtr=ifelse( ModT.probsTr>.5,"Jazz","Classic") 
table(ModT.pred,musicT.test$GENRE)
ModT_TestAccuracy<-sum(diag(table(ModT.pred,musicT.test$GENRE)))/length(musicT.test$GENRE)
ModT_TestpredictionError<-1-ModT_TestAccuracy
ModT_trainAccuracy<-sum(diag(table(ModT.predtr,musicT.tr$GENRE)))/length(musicT.tr$GENRE)
ModT_trainpredictionError<-1-ModT_trainAccuracy
seuil=seq(0,1,0.01)
n<-nrow(musicT.test)
# calcul de la table de classification pour différents seuils
sap=sapply(seq(0.01,0.99,0.01),
           function(x){c(table(Y=musicT.test$GENRE,pred=ModT.probs>x))})
GENREn<-ifelse(musicT.test$GENRE=="Jazz",1,0)
# ajout des cas spéciaux s=0 et s=1
tc=data.frame(rbind(c(0,0,n-sum(GENREn), sum(GENREn)),
                    t(sap),
                    c( n-sum(GENREn),sum(GENREn),0,0))) 
names(tc)=c("VN","FN","FP","VP")

# le plot
matplot(seuil,cbind(tc$FN,tc$FP,tc$FN+tc$FP),
        type="l",ylab="erreur",
        lty=c(2,3,1),col=c(2,3,1))

legend("top",c("FN","FP","totale"),lty=c(2,3,1),col=c(2,3,1))
#le seuil minimum (estimé)
wm=which.min(tc$FN+tc$FP)
c(seuil[wm], (tc$FN+tc$FP)[wm]/n)    
points(seuil[wm],(tc$FN+tc$FP)[wm],  pch=19,col=2)





############
### mod 1
############

t<-summary(ModT)
list<-c()
for(i in 2:ncol(music)){
  if(t$coefficients[,4][i]<0.05){
    list<-c(list,names(t$coefficients[,4][i]))
  }
}
data<-rep(nrow(music))
for(i in 1:length(list)) {
  data<-cbind(data,music[list[i]])
}
GENRE<-ifelse(music$GENRE[train]=="Jazz",1,0)
data<-data.frame(data)
data.tr<-data[train,]
data.test<-data[!train,]
Mod1<-glm(GENRE~.,data=data.tr,family=binomial)
Mod1.probs=predict(Mod1,newdata = data.test,type="response")
Mod1.pred=ifelse( Mod1.probs>.5,"Jazz","Classic") 
Mod1.probsTr=predict(Mod1,newdata = data.tr,type="response")
Mod1.predtr=ifelse( Mod1.probsTr>.5,"Jazz","Classic") 
table(Mod1.pred,music$GENRE[!train])
Mod1TestAccuracy<-sum(diag(table(Mod1.pred,music$GENRE[!train])))/length(music$GENRE[!train])
Mod1TestpredictionError<-1-Mod1TestAccuracy
Mod1TrainAccuracy<-sum(diag(table(Mod1.predtr,music$GENRE[train])))/length(music$GENRE[train])
Mod1TrainpredictionError<-1-Mod1TrainAccuracy

seuil=seq(0,1,0.01)
n<-nrow(data.test)
# calcul de la table de classification pour différents seuils
sap=sapply(seq(0.01,0.99,0.01),
           function(x){c(table(Y=music$GENRE[!train],pred=Mod1.probs>x))})
GENREn<-ifelse(music$GENRE[!train]=="Jazz",1,0)
# ajout des cas spéciaux s=0 et s=1
tc=data.frame(rbind(c(0,0,n-sum(GENREn), sum(GENREn)),
                    t(sap),
                    c( n-sum(GENREn),sum(GENREn),0,0))) 
names(tc)=c("VN","FN","FP","VP")

# le plot
matplot(seuil,cbind(tc$FN,tc$FP,tc$FN+tc$FP),
        type="l",ylab="erreur",
        lty=c(2,3,1),col=c(2,3,1))
#le seuil minimum (estimé)
wm=which.min(tc$FN+tc$FP)
c(seuil[wm], (tc$FN+tc$FP)[wm]/n)    
points(seuil[wm],(tc$FN+tc$FP)[wm],  pch=19,col=2)




## mod2 
## mod2 
t<-summary(ModT)
list<-c()
for(i in 2:ncol(music)){
  if(t$coefficients[,4][i]<0.2){
    list<-c(list,names(t$coefficients[,4][i]))
  }
}
data2<-rep(nrow(music))
for(i in 1:length(list)) {
  data2<-cbind(data2,music[list[i]])
}
GENRE<-ifelse(music$GENRE[train]=="Jazz",1,0)
data2<-data.frame(data2)
data2.tr<-data2[train,]
data2.test<-data2[!train,]
Mod2<-glm(GENRE~.,data=data2.tr,family=binomial)
Mod2.probs=predict(Mod2,newdata = data2.test,type="response")
Mod2.pred=ifelse( Mod2.probs>.5,"Jazz","Classic") 
Mod2.probstr=predict(Mod2,newdata = data2.tr,type="response")
Mod2.predtr=ifelse( Mod2.probstr>.5,"Jazz","Classic") 
Mod2TestAccuracy<-sum(diag(table(Mod2.pred,music$GENRE[!train])))/length(music$GENRE[!train])
Mod2TestpredictionError<-1-Mod2TestAccuracy
Mod2TrainAccuracy<-sum(diag(table(Mod2.predtr,music$GENRE[train])))/length(music$GENRE[train])
Mod2TrainpredictionError<-1-Mod2TrainAccuracy
table(Mod2.pred,music$GENRE[!train])
####0.8907331
seuil=seq(0,1,0.01)
n<-nrow(data.test)
# calcul de la table de classification pour différents seuils
sap=sapply(seq(0.01,0.99,0.01),
           function(x){c(table(Y=music$GENRE[!train],pred=Mod1.probs>x))})
GENREn<-ifelse(music$GENRE[!train]=="Jazz",1,0)
# ajout des cas spéciaux s=0 et s=1
tc=data.frame(rbind(c(0,0,n-sum(GENREn), sum(GENREn)),
                    t(sap),
                    c( n-sum(GENREn),sum(GENREn),0,0))) 
names(tc)=c("VN","FN","FP","VP")

# le plot
matplot(seuil,cbind(tc$FN,tc$FP,tc$FN+tc$FP),
        type="l",ylab="erreur",
        lty=c(2,3,1),col=c(2,4,1))

##########
##ModAIC
#########

modAIC=stepAIC(ModT)



modAIC.probs=predict(modAIC,newdata = musicT.test,type="response")
modAIC.pred=ifelse( modAIC.probs>.5,"Jazz","Classic") 
modAIC.probstr=predict(modAIC,newdata = musicT.tr,type="response")
modAIC.predtr=ifelse( modAIC.probstr>.5,"Jazz","Classic") 
ModAIC_TestAccuracy<-sum(diag(table(modAIC.pred,musicT.test$GENRE)))/length(musicT.test$GENRE)
ModAIC_TestpredictionError<-1-ModAIC_TestAccuracy
ModAIC_TrainAccuracy<-sum(diag(table(modAIC.predtr,musicT.tr$GENRE)))/length(musicT.tr$GENRE)
ModAIC_TrainpredictionError<-1-ModAIC_TrainAccuracy
table(modAIC.pred,musicT.test$GENRE)
n<-nrow(musicT.test)
# calcul de la table de classification pour différents seuils
sap=sapply(seq(0.01,0.99,0.01),
           function(x){c(table(Y=musicT.test$GENRE,pred=modAIC.probs>x))})
GENREn<-ifelse(musicT.test$GENRE=="Jazz",1,0)
# ajout des cas spéciaux s=0 et s=1
tc=data.frame(rbind(c(0,0,n-sum(GENREn), sum(GENREn)),
                    t(sap),
                    c( n-sum(GENREn),sum(GENREn),0,0))) 
names(tc)=c("VN","FN","FP","VP")

# le plot
matplot(seuil,cbind(tc$FN,tc$FP,tc$FN+tc$FP),
        type="l",ylab="erreur",
        lty=c(2,3,1),col=c(2,4,1))


#ROC
#ROC

predtest=prediction(ModT.probs,musicT.test$GENRE) # pour utiliser performance()
predtr=prediction(ModT.probsTr,musicT.tr$GENRE)
pred0=prediction(Mod0.probs,musicM.test$GENRE)
pred1=prediction(Mod1.probs,music$GENRE[!train])
pred2=prediction(Mod2.probs,music$GENRE[!train])
predAIC=prediction(modAIC.probs,music$GENRE[!train])
plot(performance(predtest,"sens","fpr"),col=3)
plot(performance(predtr,"sens","fpr"),col=2,add=TRUE)
plot(performance(pred0,"sens","fpr"),col=5,add=TRUE)
plot(performance(pred1,"sens","fpr"),col=6,add=TRUE)
plot(performance(pred2,"sens","fpr"),col=7,add=TRUE)
plot(performance(predAIC,"sens","fpr"),col=8,add=TRUE)
perftest <- performance(pred, "auc")                       
AUCapptest=round(perftest@y.values[[1]],3)   
perftr <- performance(predtr, "auc")                       
AUCapptr=round(perftr@y.values[[1]],3)    
perf0 <- performance(pred0, "auc")                       
AUCapp0=round(perf0@y.values[[1]],3)    
perf1<- performance(pred1, "auc")                       
AUCapp1=round(perf1@y.values[[1]],3)
perf2 <- performance(pred2, "auc")                       
AUCapp2=round(perf2@y.values[[1]],3)
perfAIC <- performance(predAIC, "auc")                       
AUCappAIC=round(perfAIC@y.values[[1]],3)
lines(c(0,1),c(0,1),col=4,lty=2)                       
segments(c(0,0),c(0,1),c(0,1),c(1,1),lty=3,lwd=2,col=1) 
legend("bottomright",c(paste("parfait AUC=" ,"1"),
                       paste("testModT AUC=" ,round(AUCapptest,2)),
                       paste("trainModT AUC=",round(AUCapptr,2)),
                       paste("aléatoire AUC=","0.5"),
                       paste("mod0 AUC=" ,round(AUCapp0,2)),
                       paste("mod1 AUC=" ,round(AUCapp1,2)),
                       paste("mod2 AUC=" ,round(AUCapp2,2)),paste("testModAIC AUC=" ,round(AUCappAIC,2))),lty=c(2,1,1,2,1,1,1,1),col=c(1,3,2,4,5,6,7,8),cex=0.5)




##Prediction error ##
ErrpredTrain<-c(Mod0TrainpredictionError,ModT_trainpredictionError,Mod1TrainpredictionError,Mod2TrainpredictionError,ModAIC_TrainpredictionError)
names(ErrpredTrain)<-c("Mod0","ModT","Mod1","Mod2","ModAIC")
ErrpredTrain
which.min(ErrpredTrain)
ErrpredTest<-c(Mod0TestpredictionError,ModT_TestpredictionError,Mod1TestpredictionError,Mod2TestpredictionError,ModAIC_TestpredictionError)
names(ErrpredTest)<-c("Mod0","ModT","Mod1","Mod2","ModAIC")
ErrpredTest
which.min(ErrpredTest)



######################
#### KNN 
######################

x<-music[train,]
xnew<-music[!train,]
xtr  <- music$GENRE[train]
y<-ifelse(xtr=="Jazz",1,0)
ytest<-music$GENRE[!train]
ytest<-ifelse(ytest=="Jazz",1,0)
modKNN<-knn(train=x[-length(x)], test=xnew[-length(xnew)], cl=y, k=1, prob=TRUE) 
table(modKNN,xnew$GENRE)
##69.01
K = seq(1,50,1)
nK = length(K)
# deux vecteurs vides pour stocker les deux types d'erreurs 
ErrTrain = rep(NA, length=nK)
ErrTest  = rep(NA, length=nK)

for (i in 1:nK) 
{
  # valeur courante de k 
  k = K[i] 
  # modÃ¨le pour classer le jeu de donnÃ©es d'apprentissage
  modtrain = knn(train=x[-length(x)], test=x[-length(xnew)], cl=xtr, k=k, prob=TRUE)
  # proportion de mal classÃ©s du jeu de donnÃ©es d'apprentissage
  ErrTrain[i] = mean(modtrain!=x$GENRE)
  # modÃ¨le pour classer le jeu de donnÃ©es de test (appelÃ© aussi validation)
  modtest = knn(train=x[-length(x)], test=xnew[-length(xnew)], cl=xtr, k=k, prob=TRUE)
  # proportion de mal classÃ©s du jeu de donnÃ©es de test
  ErrTest[i] = mean(modtest!=xnew$GENRE)
}

# Figure pour superposer les deux erreurs
# erreur test 

plot(K[1:50], ErrTest, type="b", col="blue", xlab="nombre de voisins",ylab=" erreurs train et test", pch=20, 
     ylim=range(c(ErrTest, ErrTrain)))
# erreur d'apprentissage
lines(K[1:50], ErrTrain,type="b",col="red",pch=20)
# une petite lÃ©gende
legend("bottomright",lty=1,col=c("red","blue"),legend = c("train ", "test "))

k=2
modKnn2<-knn(train=x[-length(x)], test=xnew[-length(xnew)], cl=y, k=k, prob=TRUE)
table(modKnn2,xnew$GENRE)
AccuracyKNN<-sum(diag(table(modKNN,xnew$GENRE)))/length(xnew$GENRE)
AccuracyKNN







############################
#  RIDGE
###########################

library(glmnet)

grid=10^seq(10,-2,length=100) # la grille de lambda
x.train=model.matrix(GENRE~.,musicT.tr)[,-1]
x.test=model.matrix(GENRE~.,musicT.test)[,-1]
y.test=musicT.test$GENRE
y.train=musicT.tr$GENRE
y.test=ifelse(y.test=="Jazz",1,0)
y.train=ifelse(y.train=="Jazz",1,0)

########## Q2 ######################
y.test=musicT.test$GENRE
y.train=musicT.tr$GENRE
y.test=ifelse(y.test=="Jazz",1,0)
y.train=ifelse(y.train=="Jazz",1,0)

ridge.fit=glmnet(x.train,y.train,alpha=0,family="binomial",lambda=grid)

plot(ridge.fit,label = TRUE)


test<-glm(musicT.tr$GENRE~PAR_SFMV24+PAR_SFMV2,data = musicT.tr,family = binomial)
test.probs=predict(test,newdata = musicT.test,type="response")
test.probsTr=predict(test,newdata = musicT.tr,type="response")
## seuil de Bayes 0.5
test.probs
test.pred=ifelse( test.probs>.5,"Jazz","Classic") 
table(test.pred,musicT.test$GENRE)
1046+386
1432/2169

########## plus dans Q2      ##############################
j=0
accuracy <- c()
SCR <- c()
for (i in grid)   #parcours de lambda dans le Grid
{
  ridge.pred=predict(ridge.fit,s=i,newx=x.test) 
  ridge.pred=ifelse(ridge.pred>=0,1,0)  ###########"" à corrigeer
  j=j+1
  accuracy[j]=sum(y.test==ridge.pred)/length(y.test)*100
}

plot(1:100,accuracy,xlab="indice de lambda", 
     ylab="Accuracy",
     main="Accuracy en fonction de lambda",type='l')  #accuracy de la regression en fonction des valeurs prise par s



############### Question 3 ###############
set.seed(341)

cv.out=cv.glmnet(x.train,y.train,alpha=0,family = "binomial",nfolds=10,type.measure="class") 
plot(cv.out)

bestlam=cv.out$lambda.min      #[1] 0.02221237
log(bestlam)  #[1] -3.807106


ridge.pred=predict(ridge.fit,s=bestlam,newx=x.test,type="class")
mean((ridge.pred==y.test))*100     #[1] 89.39604



############### Question 4 ###############
set.seed(4658)
music_all<-read.table("music.txt",sep=';',header = TRUE)
train=sample(c(TRUE,FALSE),6447,rep=TRUE,prob=c(2/3,1/3))
music_all.tr<-music_all[train,]
music_all.test<-music_all[!train,]

x.train=model.matrix(GENRE~.,music_all.tr)[,-1]
x.test=model.matrix(GENRE~.,music_all.test)[,-1]
y.test=music_all.test$GENRE
y.train=music_all.tr$GENRE
y.test=ifelse(y.test=="Jazz",1,0)
y.train=ifelse(y.train=="Jazz",1,0)


cv.out=cv.glmnet(x.train,y.train,alpha=0,family = "binomial",nfolds=10,type.measure="class")     
plot(cv.out)

bestlam=cv.out$lambda.min       #0.04428153                  # 
log(bestlam)  

test=glmnet(x.train,y.train,alpha=0,family="binomial",lambda=bestlam)
ridge.pred=predict(test,s=bestlam,newx=x.test,type="class")
mean((ridge.pred==y.test))*100     #[1] 89.65827%



##########################
#### Random Forest
#########################

cl <- makePSOCKcluster(2)
registerDoParallel(cl)

control <- trainControl(method="repeatedcv", number=5, repeats=10)
rfGrid <-  expand.grid(mtry = 1:7)
GENRE=musicT.tr$GENRE
RFmodel <- train(GENRE ~., data=musicT.tr, method="rf", 
                 trControl=control,
                 ntree=500, 
                 tuneGrid = rfGrid,
                 verbose=FALSE)
stopCluster(cl)
plot(RFmodel)

pred.rf.caret <- predict(RFmodel, musicT.test)
mean(musicT.test$GENRE !=pred.rf.caret) #erreur de prediction
mean(musicT.test$GENRE ==pred.rf.caret) # accuracy = 94.42%



###########################
## gradient boosting 
###########################


control <- trainControl(method="cv", number=2)
boost.grid = expand.grid(eta = 1,
                         nrounds = c(700, 750, 800, 850), # best : 750
                         max_depth = 2,
                         subsample = 1,
                         min_child_weight = 1.,
                         colsample_bytree = 0.5,
                         gamma = 0.)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)
boost.model <- train(musicT.tr[,-length(musicT.tr)], 
                     musicT.tr$GENRE, 
                     method = "xgbTree",
                     metric = "Accuracy",
                     booster = "gbtree", # règle faible : arbre
                     trControl = control,
                     tuneGrid = boost.grid)

stopCluster(cl)
boost.pred = predict(boost.model, musicT.test)
# erreur de test boost
mean(boost.pred!=musicT.test$GENRE) #0.04379899
# accuracy 
mean(boost.pred==musicT.test$GENRE) #0.95601
table(boost.pred, musicT.test$GENRE)

save.image(file = "my_work_space.RData") 


############################
#### Neural Network 
############################

x_train<-c()
x_train <- array_reshape(musicT.tr[-length(musicT.tr)], c(nrow(musicT.tr[-length(musicT.tr)]), 161))
x_test <- array_reshape(musicT.test[-length(musicT.test)], c(nrow(musicT.test[-length(musicT.test)]), 161))
# rescale
x_train <- scale(x_train)
x_test <- scale(x_test)
ytr<-ifelse(musicT.tr$GENRE=="Jazz",1,0)
yte<-ifelse(musicT.test$GENRE=="Jazz",1,0)
y_train_c <- to_categorical(ytr, 2)
y_test_c <- to_categorical(yte, 2)

cl <- makePSOCKcluster(3)
registerDoParallel(cl)
model_1 <- keras_model_sequential() 
model_1 %>% 
  layer_dense(units = 161, activation = 'relu', input_shape = c(161)) %>% 
  layer_dense(units = 2, activation = 'softmax')
summary(model_1)

model_1 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)
model_1 %>% fit(
  x_train, y_train_c, 
  epochs = 80, batch_size = 200, 
  validation_data = list(x_test, y_test_c))
stopCluster(cl)
# erreur de test keras
keras.pred = model_1 %>% predict_classes(x_test)
mean(keras.pred==yte)
table(keras.pred, yte)


######################
### QDA
######################

qda.fit=qda(GENRE~.,data=musicT.tr)
qda.class =predict(qda.fit,musicT.test)$class
mean(qda.class == musicT.test$GENRE)

#################################
## Support Vector Machine SVM 
#################################

x.train=model.matrix(GENRE~.,musicT.tr)[,-1]
x.test=model.matrix(GENRE~.,musicT.test)[,-1]
y.test=musicT.test$GENRE
y.train=musicT.tr$GENRE
y.test=ifelse(y.test=="Jazz",1,0)
y.train=ifelse(y.train=="Jazz",1,0)
svm1 <- svm(y.train ~ ., data=musicT.tr, kernel="linear",   type = 'C-classification')
#svm1 <- svm(y.train ~ ., data=music_all.tr, kernel="polynomial",   type = 'C-classification')
#svm1 <- svm(y.train ~ ., data=music_all.tr, kernel="radial",   type = 'C-classification')
#svm1 <- svm(y.train ~ ., data=music_all.tr, kernel="sigmoid",   type = 'C-classification')
y_pred = predict(svm1, musicT.tr[-length(musicT.tr)]) 
table(y_pred,y.train)
##### pas d'érreur dapprentissage, les variables sont linéairement séparable.
y_pred = predict(svm1, newdata = musicT.test) 
table(y_pred,y.test)
##### le SVM permet une séparation parfaite des données 
dim(svm1$SV)   #809 support vector
