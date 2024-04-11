#Facebook Prophet Algorithm, Geometric Brownian Motion,  and Artificial Neural Network Model in LQ 45 Stocks Price Prediction

#Import Library
library(quantmod)
library(ggplot2)
library(MASS)
library(neuralnet)
library(forecast)
library(Sim.DiffProc)
library(prophet)

#Import Data
start_Date = as.Date("2021-09-11")
end_Date = as.Date("2022-09-11")
getSymbols(c("ICBP.JK"), src="yahoo", from = '2021-09-11', to = '2022-09-11') 
stock=data.frame(as.xts(ICBP.JK))
head(stock)
stock=stock[,!apply(is.na(stock),2,all)]
stock=stock[!apply(is.na(stock),1,all),]
stock=cbind(Date=rownames(stock), stock)
rownames(stock)=1:nrow(stock)


#---------------------------------------- FACEBOOK PROPHET ALGORITHM ----------------------------------------#

#Convert dataset as prophet input requires
df=data.frame(ds = stock$Date,
              y = stock$ICBP.JK.Close)

#Getting a glimpse and summary of the data
summary(df)
head(df)

#Call the prophet function to fit the model
Model1=prophet(df) # building the model 
Future1=make_future_dataframe(Model1,periods=180) #predicting future values
tail(Future1)

#Forecasting the models created and forecasting the data for 180 days
forecast=predict(Model1,Future1)
forecast

#The future data is displayed with the actual value and lower and upper value
tail(forecast[c('ds','yhat','yhat_lower','yhat_upper')])

#Plot the model estimates.
dyplot.prophet(Model1,forecast)

#Components of the forecast (trends with weekly and yearly seasonality)
prophet_plot_components(Model1,forecast)

#Creating train prediction datset to compare the real data
dataprediction=data.frame(forecast$ds,forecast$yhat)
dataprediction
trainlen=length(df)
dataprediction1=dataprediction[c(1:trainlen),]
dataprediction1

#Creating cross validation accuracy.
accuracy(dataprediction$forecast.yhat,df$y)

#---------------------------------------- GEOMETRIC BROWNIAN MOTION ----------------------------------------#

#Extract required data
data.gbm=data.frame(stock[,5])

#Splitting data into data train and data testing
train=sample(nrow(data.gbm)*0.8,replace = F)
traindata=data.gbm[train,]
testdata=data.gbm[-train,]
traindata.ts=ts(traindata)
plot(traindata.ts, main="Stock Price for ", ylab="Stock Return", col="blue")

#Total values in data testing
totdat=length(testdata)

#Set Up Initial Value
IV=testdata[1]

#Create drift and volatility equation
d=expression(theta[1]*x) #expression for drift term
s=expression(theta[2]*x) #expression for volatility term

#Parameter Estimation
drift.f=function(s,lag=1){
  N=length(s)
  
  #set the equation for lag difference
  if(N<1+lag){
    stop("S must be greater than 2+lag")
  }
  ct=s[(1+lag):N] #define the next stock value
  pt=s[1:(N-lag)] #define the current stock value
  t=1 #time horizon (daily)
  dt=t/N #change in time
  
  stock.R=(ct-pt)/pt
  miu.hat=sum(stock.R)/(N*dt) #mean stock.R
  miu.hat #display result
}

drift.f(traindata.ts) #drift estimate

#Pseudocode for estimating volatility parameter
volt.f=function(s,lag=1){
  N=length(s)
  
  #set condition for lag difference
  if(N<1+lag){
    stop("S must be greater than 2+lag")
  }
  ct=s[(1+lag):N] #define the next stock value
  pt=s[1:(N-lag)] #define the current stock value
  Diff=ct-pt
  
  tt=1 #time horizon (daily)
  dt=tt/N #change in time
  
  stock.R=(ct-pt)/pt
  miu.hat=mean(stock.R) #mean stock.R
  hat.sig2=sum((stock.R-miu.hat)^2)/((N-1)*dt)
  hat.sig=sqrt(hat.sig2)
  hat.sig #display result
}

volt.f(traindata.ts) #volatility estimate

#Assign estimated values to defined objects
drift=drift.f(traindata.ts)
diffusion=volt.f(traindata.ts)

#Create drift and diffusion equations from the estimated values for simulation
d=eval(substitute(expression(drift*x), list(drift=drift)))
s=eval(substitute(expression(diffusion*x), list(diffusion=diffusion)))

#Number of Simulation
NSimul=1000

#Creates a cumulative sum of the simulated
pred_x=rep(0,totdat)

#All x is used to store all simulated values for cumulative
#the standard deviation for the confidence interval
all_x=data.frame()

library(Sim.DiffProc)
for(i in 1:NSimul){
  #create a new random seed for each simulation
  rand=as.integer(1000*runif(1))
  set.seed(rand)
  
  #simulate the SDE using the euler method for 15 days into the future
  X=snssde1d(N=totdat-1, M=1, x0=IV, Dt=1/totdat, drift=d, diffusion=s, method="euler")
  
  pred_x=pred_x + X$X
  all_x=rbind(all_x, as.numeric(X$X))
}

pred_x=pred_x/NSimul
pred_x

sd_x=sapply(all_x, sd) #standard deviation

#comparing actual vs predicted value
data.frame(testdata, pred_x)

#MAPE
mape.f=function(a.val,p.val){
  (1/length(a.val))*sum(abs((a.val-p.val)/a.val))*100
}

mape.f(testdata, pred_x)

#Create upper and lower confidence interval
upper=pred_x + (1.96*sd_x)
lower=pred_x - (1.96*sd_x)
cint=data.frame(lower,upper)
cint

p=ggplot()
p=p+geom_line(aes(x=1:195, y=traindata, color="Original Data"))
p=p+geom_line(aes(x=196:244, y=testdata, color="Original Data Test"))
p=p+geom_line(aes(x=196:244, y=pred_x, color="Predicted Data Using GBM"))
p=p+ylab("Values of The Stock")
p=p+xlab("Time in Days")
p=p+ggtitle("GBM Forecast for ICBP")
p=p+geom_ribbon(aes(x=c(196:244),y=pred_x, ymin=lower, ymax=upper), linetype=1, alpha=0.1)
p

#---------------------------------------- ARTIFICIAL NEURAL NETWORK ----------------------------------------#
data=stock[-1]
str(data)

set.seed(1000)
apply(data, 2 ,range)


maxValue=apply(data,2,max)
minValue=apply(data,2,min)

data=as.data.frame(scale(data, center= minValue, scale = maxValue-minValue))

train=sample(nrow(data)*0.8,replace = F)
traindata=data[train,]
testdata=data[-train,]

allVars=colnames(data)
predictorVars=allVars[!allVars%in%"ICBP.JK.Close"]
predictorVars=paste(predictorVars,collapse = "+")

form=as.formula(paste("ICBP.JK.Close~",predictorVars,collapse = "+"))

neuralModel=neuralnet(formula=form, hidden=c(4,2), linear.output=T, data = traindata)


neuralnet=function (formula, data, hidden = 1, threshold = 0.01, stepmax = 1e+05, 
                    rep = 1, startweights = NULL, learningrate.limit = NULL, 
                    learningrate.factor = list(minus = 0.5, plus = 1.2), learningrate = NULL, 
                    lifesign = "none", lifesign.step = 1000, algorithm = "rprop+", 
                    err.fct = "sse", act.fct = "logistic", linear.output = TRUE, 
                    exclude = NULL, constant.weights = NULL, likelihood = FALSE) 
{
  call=match.call()
  options(scipen = 100, digits = 10)
  result=varify.variables(data, formula, startweights, learningrate.limit, 
                          learningrate.factor, learningrate, lifesign, algorithm, 
                          threshold, lifesign.step, hidden, rep, stepmax, err.fct, 
                          act.fct)
  data=result$data
  formula=result$formula
  startweights=result$startweights
  learningrate.limit=result$learningrate.limit
  learningrate.factor=result$learningrate.factor
  learningrate.bp=result$learningrate.bp
  lifesign=result$lifesign
  algorithm=result$algorithm
  threshold=result$threshold
  lifesign.step=result$lifesign.step
  hidden=result$hidden
  rep=result$rep
  stepmax=result$stepmax
  model.list=result$model.list
  matrix=NULL
  list.result=NULL
  result=generate.initial.variables(data, model.list, hidden, 
                                    act.fct, err.fct, algorithm, linear.output, formula)
  covariate=result$covariate
  response=result$response
  err.fct=result$err.fct
  err.deriv.fct=result$err.deriv.fct
  act.fct=result$act.fct
  act.deriv.fct=result$act.deriv.fct
  for (i in 1:rep) {
    if (lifesign != "none") {
      lifesign=display(hidden, threshold, rep, i, lifesign)
    }
    utils::flush.console()
    result <- calculate.neuralnet(learningrate.limit = learningrate.limit, 
                                  learningrate.factor = learningrate.factor, covariate = covariate, 
                                  response = response, data = data, model.list = model.list, 
                                  threshold = threshold, lifesign.step = lifesign.step, 
                                  stepmax = stepmax, hidden = hidden, lifesign = lifesign, 
                                  startweights = startweights, algorithm = algorithm, 
                                  err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                                  act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                                  rep = i, linear.output = linear.output, exclude = exclude, 
                                  constant.weights = constant.weights, likelihood = likelihood, 
                                  learningrate.bp = learningrate.bp)
    if (!is.null(result$output.vector)) {
      list.result=c(list.result, list(result))
      matrix=cbind(matrix, result$output.vector)
    }
  }
  utils::flush.console()
  if (!is.null(matrix)) {
    weight.count=length(unlist(list.result[[1]]$weights)) - 
      length(exclude) + length(constant.weights) - sum(constant.weights == 
                                                         0)
    if (!is.null(startweights) && length(startweights) < 
        (rep * weight.count)) {
      warning("some weights were randomly generated, because 'startweights' did not contain enough values", 
              call. = F)
    }
    ncol.matrix=ncol(matrix)
  }
  else ncol.matrix <- 0
  if (ncol.matrix < rep) 
    warning(sprintf("algorithm did not converge in %s of %s repetition(s) within the stepmax", 
                    (rep - ncol.matrix), rep), call. = FALSE)
  nn=generate.output(covariate, call, rep, threshold, matrix, 
                     startweights, model.list, response, err.fct, act.fct, 
                     data, list.result, linear.output, exclude)
  return(nn)
}

#Result options
names(neuralModel)
plot(neuralModel)
neuralModel$result.matrix

out=cbind(neuralModel$covariate,neuralModel$net.result[[1]])
head(out)
dimnames(out)=list(NULL, 
                   c("Open", "High","Low",
                     "Close","Adj.Close","Volume"))

head(neuralModel$generalized.weights[[1]])

predictions=neuralnet::compute(neuralModel,testdata[,-5])
str(predictions)
predictions$net.result

str(testdata)


## predicting and unscalling
predictions=(predictions$net.result)*(max(testdata$ICBP.JK.Close)-min(testdata$ICBP.JK.Close))+min(testdata$ICBP.JK.Close)
actualValues=(testdata$ICBP.JK.Close)*(max(testdata$ICBP.JK.Close)-min(testdata$ICBP.JK.Close))+ min(testdata$ICBP.JK.Close)

MAPE=(1/length(actualValues))*sum(abs((actualValues-predictions)/actualValues))*100
MAPE

plot(predictions,actualValues)
