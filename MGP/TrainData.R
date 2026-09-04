loc = round(seq(1,100,length = 40))
source<-array(NA,dim = c(3,40,100)) 
source[1,,] <- t(read.csv("source1.csv"))[loc,]
source[2,,] <- t(read.csv("source2.csv"))[loc,]
source[3,,] <- t(read.csv("source3.csv"))[loc,]

tloc1 = round(seq(1,100,length = 10))
target<-t(read.csv("target1.csv"))[tloc1,]

library(MASS)
#mean trend 

n = 4
#add noise based on 10 replication for source and 3 replication for target
num <- c(100,100,100,5)
tloc = loc
trainy1=lapply(1:(n-1),function(i){lapply(1:num[i],function(j){source[i,,j]})}) #Initial Profiles


#aggregate mean and average noise for all sources 
trainy2=matrix(NA, nrow=(n-1),ncol=length(tloc)) 
ytable<-vector(mode = "list",length = (n-1))
for(i in 1:(n-1)){
  ytable[[i]]<-matrix(unlist(trainy1[[i]]),nrow = num[i],ncol = length(tloc),byrow = TRUE)
  ptsum <- colSums(ytable[[i]]) #sum on single points 
  countpt<-apply(ytable[[i]], 2, function(c)sum(c!=0))
  for(j in 1:length(tloc)){
    if (countpt[j] == 0){
      trainy2[i,j]<-NA
    }
    trainy2[i,j]<- ptsum[j]/countpt[j]
  }
}



trainy31 = lapply(1:num[n],function(j){target[,j]}) #Initial Profiles

testtable<-matrix(unlist(trainy31),nrow = num[n],ncol = length(tloc1),byrow = TRUE)
testsum <- colSums(testtable)
counttest<-apply(testtable, 2, function(c)sum(c!=0))
trainy3<-rep(NA,length = length(tloc1))
for(j in 1:length(tloc1)){ #mean of the target profile
  if(counttest[j] == 0){
    trainy3[j]<-NA
  }
  trainy3[j]= testsum[j]/counttest[j]
}


#important that gonna be used later
train_x<-lapply(1:(n-1),function(i){seq(-5,5,length = 100)[tloc]})
train_y<-lapply(1:(n-1),function(i){trainy2[i,]})
test_x<-seq(-5,5,length = 100)[tloc1]
test_y<-trainy3

same<-t(read.csv("target2.csv"))
pred_samples=lapply(1:80,function(j){same[,j]}) #target profiles
tsame = round(seq(1, 100, length.out = 10))

tpoints = seq(-5,5,length = 100)


