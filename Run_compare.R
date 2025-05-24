options(repos = c(CRAN = "https://cran.r-project.org"))
install.packages("Matrix", type = "binary")
install.packages("MASS", type = "binary")
install.packages("nloptr")
install.packages("minqa")
install.packages("optimx")
install.packages("rootSolve")
install.packages("nlme")
install.packages("mvtnorm")


library(Matrix)
library(nloptr)
library(minqa)
library(optimx)
library(rootSolve)
library(nlme)
library(mvtnorm)
library(MASS)


source('TrainData.R')

index=function(n,len,m,lensame) #creating index for sparse matrix elements
{
  p1=c();p2=c();p3=c();p4=c();p5=c();p6=c();p7=c();p8=c();p9=c();p10=c()
  pp=sum(len)
  for(j in 1:(n-1))
  {
    i1=1 + sum(len[0:(j-1)])
    for(i in i1:(i1+len[j]-1))
    {
      p1=c(p1,i1:i)
      p2=c(p2,rep(i,length(i1:i)))
    }
  }
  p3=rep(1:pp,m)
  for(i in 1:m)
  {
    p4=c(p4,rep(pp+i,pp))
  }
  i2=pp+1
  for(i in i2:(i2+m-1))
  {
    p5=c(p5,i2:i)
    p6=c(p6,rep(i,length(i2:i)))
  }
  p7=rep(1:(pp+m),lensame)
  for(i in 1:lensame)
  {
    p8=c(p8,rep(pp+m+i,pp+m))
  }
  i3 = pp+m+1
  for(i in i3:(i3+lensame-1))
  {
    p9=c(p9,i3:i)
    p10=c(p10,rep(i,length(i3:i)))
  }
  return(list(pfi=c(p1,p3,p5,p7,p9),pfj=c(p2,p4,p6,p8,p10)))
}

cyii=function(a,b,L,i) #construct within-process covariance matrix
{ Sigma<-matrix(L[4],nrow = length(a),ncol = length(b))
diag(Sigma) <- 1; Sigma<-L[3]*Sigma*rep_factor[[i]]
d=outer(a,b,`-`)
d=d[upper.tri(d,diag=T)];error<-Sigma[upper.tri(Sigma,diag=T)]
return(list(full = L[1]^2*exp(-0.25*d^2/L[2]^2) + error, error = Sigma))
}
cyip=function(a,b,L) #construct between-process covariance matrix
{
  d=outer(a,b,`-`)
  L[1]*L[3]*sqrt(2*abs(L[2]*L[4])/(L[2]^2+L[4]^2))*exp(-0.5*d^2/(L[2]^2+L[4]^2))
}

#now the covaiance matrix is defined as follows

C=function(trainx,H) #between-and-within covariance matrix with sparse elements (Eq.10)                         
{ 
  m1 = length(test_x)
  pi = lengths(trainx)
  ppt = sum(pi)+m1+length(tloc2);
  err_mat<-matrix(0,nrow = ppt,ncol = ppt)
  
  zii=list();zip=list();zpp=c()  
  for(i in 1:(n-1)){
    i1=1 + sum(pi[0:(i-1)]);i2 = (i1+pi[i]-1)
    cyi <- cyii(trainx[[i]],trainx[[i]],L = H[c(2*i-1,2*i,4*n-1,4*n)],i)
    zii[[i]] <- cyi$full
    err_mat[i1:i2,i1:i2]<-cyi$error
  }
  
  zip = lapply(1:(n-1), function(i){cyip(trainx[[i]],test_x,H[c(2*i-1,2*i,2*n+2*i-1,2*n+2*i)])})
  K=H[(2*n-1):(4*n)]
  
  Sigma<-matrix(K[length(K)],nrow = length(test_x),ncol = length(test_x));diag(Sigma)<-1
  Sigma<-K[length(K)-1]*Sigma*rep_factor[[n]]
  error<-Sigma[upper.tri(Sigma,diag=T)] 
  D=outer(test_x,test_x,`-`)
  D=D[upper.tri(D,diag=T)]
  
  zpp<-Reduce("+",lapply(1:n, function(i){K[2*i-1]^2*exp(-0.25*D^2/K[2*i]^2)})) + error
  i3<-sum(pi)+1;i4<- sum(pi)+m1;i5 = sum(pi)+m1+1; 
  err_mat[i3:i4,i3:i4]<-Sigma
  
  zsp = lapply(1:(n-1), function(i){cyip(trainx[[i]],xsame,H[c(2*i-1,2*i,2*n+2*i-1,2*n+2*i)])})
  Dd=outer(test_x,xsame,`-`)
  zsp[[n]]<-Reduce("+",lapply(2:n, function(i){K[2*i-1]^2*exp(-0.25*Dd^2/K[2*i]^2)}))
  +cyip(test_x,xsame,H[c(2*n-1,2*n,4*n+3,4*n+4)])
  
  Dss=outer(xsame,xsame,`-`)
  Dss=Dss[upper.tri(Dss,diag=T)]
  Sigma_ss<-matrix(K[length(K)],nrow = length(xsame),ncol = length(xsame));diag(Sigma_ss)<-1
  Sigma_ss<-K[length(K)-1]*Sigma_ss
  
  err_mat[i5:ppt,i5:ppt]<-Sigma_ss
  
  err_ss<-Sigma_ss[upper.tri(Sigma_ss,diag=T)]
  zss<-Reduce("+",lapply(2:n, function(i){K[2*i-1]^2*exp(-0.25*Dss^2/K[2*i]^2)})) + H[4*n+1]^2*exp(-0.25*Dss^2/H[4*n+2]^2) + H[4*n+3]^2*exp(-0.25*Dss^2/H[4*n+4]^2)+err_ss
  
  b1=unlist(zii);b2=as.vector(do.call("rbind",zip)); 
  b3 = as.vector(do.call("rbind",zsp)); 
  return(list(table = sparseMatrix(i=pfi,j=pfj,x=c(b1,b2,zpp,b3,zss),symmetric=T),error = err_mat))
}

logL=function(H,fn) #log-likelihood function (Eq.17)
{ 
  B=C(train_x,H)$table
  deter=det(B)
  if(deter>0) {a=0.5*(log(deter)+t(y)%*%solve(B,y)+log(2*pi)*leny)
  } else {
    ch=chol(B)
    logdeter=2*(sum(log(diag(ch))))
    a=0.5*(logdeter+t(y)%*%solve(B,y)+log(2*pi)*leny)
  }
  penal = 0
  for(i in 1:(n-1)){
    if (abs(H[2*n+2*i-1]<=eta)){
      ait = H[2*n+2*i-1]
      penal = penal + ait^2/(2*eta)
    }
    else {
      penal = penal + abs(H[2*n+2*i-1]) - eta/2
    }
  }
  return(as.numeric(a) + gamma*penal)
}

logL_grad=function(H,fn)
{
  return(nl.grad(H,fn))
}


#scale noise for source process (Eq. 14)
rep_factor = vector(mode = "list",length = n) 
for (i in 1:(n-1)) {
  rep_factor[[i]]<-matrix(NA,nrow = length(train_x[[i]]),ncol = length(train_x[[i]]))
  table<-ytable[[i]][,!is.na(trainy2[i,])]
  for(v1 in 1:length(train_x[[i]])){
    for (v2 in 1:length(train_x[[i]])){
      #count the joint points 
      int_count<-sum(table[,v1] != 0 & table[,v2] != 0)
      v1_count<-sum(table[,v1] != 0)
      v2_count<-sum(table[,v2] != 0)
      rep_factor[[i]][v1,v2]<- int_count/(v1_count*v2_count)
    }
  }
}

#scale noise for target process (Eq. 14)
rep_factor[[n]]<-matrix(NA,nrow = length(tloc1),ncol = length(tloc1))
table<-testtable[,!is.na(trainy3)]
for(v1 in 1:length(test_x)){
  for (v2 in 1:length(test_x)){
    #count the joint points 
    int_count<-sum(table[,v1] != 0 & table[,v2] != 0)
    v1_count<-sum(table[,v1] != 0)
    v2_count<-sum(table[,v2] != 0)
    rep_factor[[n]][v1,v2]<- int_count/(v1_count*v2_count)
  }
}

rep_time = 80
RMSE<-array(0, dim = c(rep_time,1))
H_prop<-lapply(1:4,function(i){array(NA,dim = c(rep_time,4*n+4))})
ypred_prop<-array(NA,dim = c(rep_time,length(tpoints)))
yvar_prop<-vector(mode = "list",length = rep_time)

par(mfrow = c(1,1))
par(mar = c(2, 2, 2, 2))
for(nsame in c(6,8,10)){
  for(jj in 1:rep_time){
    RMSE_jj<-10^5; hot_select<-0
    tloc2<-sample(tsame,nsame,replace = FALSE)
    for (hot_key in 1:3){
      xsame <- tpoints[tloc2]
      trainsame <- pred_samples[[jj]]
      ysame<-trainsame[tloc2]
      xstar<-tpoints  #prediction points
      ystar <- trainsame  #interested to know
      
      pf=index(n,lengths(train_x),m = length(test_x),length(xsame))
      pfi=pf$pfi;pfj=pf$pfj
      
      y=c(unlist(train_y),test_y,ysame) #list of training data
      leny=length(y)
      
      if (hot_key == 1){
        x0<-c(rep(c(1,1),2*n-1),1,0.9,1,1,1,1)}
      else if (hot_key == 2) {
        x0<-c(rep(c(2,2),2*n-1),2,0.9,2,2,2,2)}
      else {
        x0<-c(rep(c(0.5,0.5),2*n-1),0.5,0.9,0.5,0.5,0.5,0.5)} 
      gamma = 0.1 #Gamma_opt
      eta = 0.00001
      opts <- list( "algorithm" = "NLOPT_LD_MMA","maxeval" = 2000) 
      
      pd_N<-max(c(lengths(train_x),length(test_x),length(xsame)))
      one=tryCatch(nloptr(x0=x0,eval_f= logL,eval_grad_f = logL_grad,
                          lb = c(rep(-Inf,4*n-2),0.02,-1/pd_N,rep(-Inf,4)), ub = c(rep(Inf,4*n-2),10,0.999,rep(Inf,4)), opts= opts,fn= logL ), error = function(e) e)
      #what happened with 0.001
      H1=one$solution
      H_prop[[hot_key]][jj,]<-H1
      
      #prediction part for penalized 
      zip_pred=list()                                                 
      zip_pred =lapply(1:(n-1), function(i){cyip(train_x[[i]],xstar,H1[c(2*i-1,2*i,2*n+2*i-1,2*n+2*i)])})
      
      D1=outer(xstar,test_x,`-`) 
      K1=H1[(2*n-1):(4*n)]
      zip_pred[[n]]=t(Reduce("+",lapply(2:n, function(i){K1[(2*i-1)]^2*exp(-0.25*D1^2/K1[(2*i)]^2)+cyip(xstar,test_x,H1[c(2*n-1,2*n,4*n+3,4*n+4)])})))
      
      D11=outer(xstar,xsame,`-`) #12*3
      zip_pred[[n+1]]=t(Reduce("+",lapply(2:n, function(i){K1[2*i-1]^2*exp(-0.25*D11^2/K1[2*i]^2)})) + H1[4*n+1]^2*exp(-0.25*D11^2/H1[4*n+2]^2) + H1[4*n+3]^2*exp(-0.25*D11^2/H1[4*n+4]^2))
      
      
      Pk=t(do.call("rbind",zip_pred))  #get K+ 
      
      D2=outer(xstar,xstar,`-`);
      sigma2<-matrix(K1[length(K1)],nrow = length(xstar),ncol = length(xstar))
      diag(sigma2)<-1; sigma2<-K1[length(K1)-1]*sigma2
      
      sk= Reduce("+",lapply(2:n, function(i){K1[2*i-1]^2*exp(-0.25*D2^2/K1[2*i]^2)})) + H1[4*n+1]^2*exp(-0.25*D2^2/H1[4*n+2]^2) + H1[4*n+3]^2*exp(-0.25*D2^2/H1[4*n+4]^2)+ sigma2
      covM=as.matrix(C(train_x,H1)$table)
      raed=solve(covM,y)
      ypred_hot=as.matrix(Pk%*%raed) # predicted meand
      yvar_hot=as.matrix((sk-Pk%*%solve(covM,t(Pk)))) #predicted variance
      RMSE_hot<-sqrt(sum((ystar-ypred_hot[,1])^2)/length(ystar))
      #RMSE_hot<-sqrt(sum((ysame-ypred_hot[samept,1])^2)/length(ysame))
      if (RMSE_hot< RMSE_jj){
        RMSE_jj<-RMSE_hot
        ypred_correct <-ypred_hot
        yvar1<-yvar_hot
        hot_select<-hot_key}
    }
    H_prop[[4]][jj,]<-H_prop[[hot_select]][jj,]
    ypred_prop[jj,]<-ypred_correct[,1]
    yvar_prop[[jj]]<-yvar1
    
    RMSE[jj,1]<-sqrt(sum((ystar-ypred_correct[,1])^2)/length(ystar)) #0.3562639
  }
  if(nsame ==6){RMSE6 = RMSE}
  if(nsame ==8){RMSE8 = RMSE}
  if(nsame ==10){RMSE10 = RMSE}
}


save(RMSE6,RMSE8,RMSE10, file = "down_original.Rdata")
RMSE6
RMSE8
RMSE10
