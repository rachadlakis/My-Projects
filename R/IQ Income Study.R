#Import Data
getwd() 
setwd("E:/USJ/Data Science/Other/Data_sets/csv Datasets")
df <- read.csv("IQByCountry.csv")
colnames(df)

#Cleaning Data
df$Income[df$Income == "#VALUE!"] <- "" 
df$Education_expenditure_per_inhabitant [df$Education_expenditure_per_inhabitant== "#VALUE!"] <- ""
df$Daily_max_temperature

typeof(df$Income)

df$IQ <- as.numeric(as.character( df$IQ) ) 
df$Income <- as.numeric( as.character( df$Income) )
df$Education_expenditure_per_inhabitant <- as.numeric( as.character(
  df$Education_expenditure_per_inhabitant) )

df$IQ
df$Income
df$Education_expenditure_per_inhabitant

#replace NUll values with average of column
df$Income[is.na(df$Income)] <- mean(df$Income, na.rm = TRUE) 
df$Education_expenditure_per_inhabitant[
  is.na(df$Education_expenditure_per_inhabitant)]  <- mean(df$Education_expenditure_per_inhabitant, 
                                                           na.rm = TRUE)

#boxplots
boxplot(df$IQ, main="IQ", sub=paste("Outlier rows: ", 
                                    boxplot.stats(df$IQ)$out)) 
boxplot(df$Income, main="Income", sub=paste("Outlier rows: ",
                                            boxplot.stats(df$Income)$out)) 
boxplot(df$Education_expenditure_per_inhabitant, main="Education_expenditure", 
        sub=paste("Outlier rows: ", 
                  boxplot.stats(df$Education_expenditure_per_inhabitant)$out)) 
boxplot(df$Daily_max_temperature, main="Daily_max_temperature",
        sub=paste("Outlier rows: ",
                                                                          boxplot.stats(df$IQ)$out)) 


df$Income
df$IQ
df$Education_expenditure_per_inhabitant

#plot IQ in function of Income
plot(df$Income, df$IQ, main="IQ / Income ", 
     xlab = "Income in thousands of dollars",
     ylab = "IQ",     
     col='blue', pch=20, cex=1.5, xlim=c(0,60000),ylim=c(40,120))

#plot IQ in function of Education_expenditure_per_inhabitant
plot(df$Education_expenditure_per_inhabitant, 
     df$IQ, main="IQ / Education expenditure per inhabitant ",
     xlab = "Education expenditure per inhabitant in thousands of dollars", 
     ylab = "IQ",col='blue', pch=20, cex=1.5, xlim=c(0,4000),ylim=c(50,120))

#plot IQ in function of Daily Max temperature
plot(df$Daily_max_temperature , df$IQ, main="IQ / Daily_max_temperature ",
     xlab = "Daily_max_temperature", ylab = "IQ", 
     col='blue', pch=20, cex=1.5, xlim=c(5,40),ylim=c(50,120))


#Regression model of IQ in function of Income alone (Simple Linear Regression)
x1 <- df$Income
y1 <-df$IQ

plot(df$Income, df$IQ, main="IQ / Income ", 
     xlab = "Income in thousands of dollars", 
     ylab = "IQ", 
     col='blue', pch=20, cex=1.5, xlim=c(0,50000),ylim=c(50,120))

#linear Regression between IQ and Income
model0 <- lm (y1~x1)
summary(model0)
plot(df$Income, df$IQ, main="IQ / Income ", 
     xlab = "Income in thousands of dollars", 
     ylab = "IQ", 
     col='blue', pch=20, cex=1.5, xlim=c(0,55000),ylim=c(50,120))
eq0 <- function(x){81.31 + 0.0004 *x }
par(new=TRUE)
plot(eq0, xlab="", ylab="", xlim=c(0,55000),ylim=c(50,120)  )

#linear Regression between IQ and LOG(Income)
model1 <- lm(y1~log(x1))
summary(model1)
plot(df$Income, df$IQ, main="IQ / Income ", 
     xlab = "Income in thousands of dollars", 
     ylab = "IQ", 
     col='blue', pch=20, cex=1.5, xlim=c(0,55000),ylim=c(50,120))
eq1 <- function(x){5.3958* log(x) + 41.9324}
par(new=TRUE)
plot(eq1, xlab="", ylab="", xlim=c(0,55000),ylim=c(50,120)  )


#IQ in function of Education Expenditure per inhabitant(Simple Linear Regression)
x2<-df$Education_expenditure_per_inhabitant
y2 <- df$IQ

 #Linear Regression Between IQ and Education Expenditure per Inhabitant 
model2 <- lm(y2~x2)
summary(model2)
eq2 <- function(x){0.009277* x + 81.48}
plot(df$Education_expenditure_per_inhabitant, df$IQ, 
     main="IQ / Education Expenditure Per Inhabitant ",
     xlab = "Education Expenditure Per Inhabitant In Thousands Of Dollars",
     ylab = "IQ", 
     col='blue', pch=20, cex=1.5, xlim=c(0,4000),ylim=c(50,120))
par(new=TRUE)
plot(eq2, xlab="", ylab="", xlim=c(0,4000),ylim=c(50,120)  )


#Linear Regression Between IQ and LOG(Education Expenditure per Inhabitant )
model3 <- lm(y2~log(x2))
summary(model3)
plot(df$Education_expenditure_per_inhabitant, df$IQ, 
     main="IQ / Education Expenditure Per Inhabitant ",
     xlab = "Education Expenditure Per Inhabitant In Thousands Of Dollars",
     ylab = "IQ", 
     col='blue', pch=20, cex=1.5, xlim=c(0,4000),ylim=c(50,120))
eq3 <- function(x){4.77*log(x) + 61.91}
par(new=TRUE)
plot(eq3, xlab="", ylab="", xlim=c(0,4000),ylim=c(50,120)  )


#Relation Between IQ and Daily Max Temperature
x3 <- df$Daily_max_temperature
y3 <-df$IQ

plot(df$Daily_max_temperature, df$IQ, main="IQ / Daily Max Temperature ", 
     xlab = "Daily Max Temperature", 
     ylab = "IQ", 
     col='blue', pch=20, cex=1.5, xlim=c(5,45),ylim=c(50,120))

model4 <- lm(y3~x3)
summary(model4)
eq4 <- function(x) { 112.32 -1.0900 *x}
par(new=TRUE)
plot(eq4, xlab="", ylab = "", xlim=c(5,45), ylim=c(50,120))


#Relation Between IQ and Log(Daily Max Temperature)
x3 <- df$Daily_max_temperature
y3 <- df$IQ

plot(df$Daily_max_temperature, df$IQ, main="IQ / Daily Max Temperature ", 
     xlab = "Daily Max Temperature", 
     ylab = "IQ", 
     col='blue', pch=20, cex=1.5, xlim=c(5,45),ylim=c(50,120))

model5 <- lm(y3~log(x3))
summary(model5)
eq5 <- function(x) { 149.615 - 20.397 *log(x)}
par(new=TRUE)
plot(eq5, xlab="", ylab = "", xlim=c(5,45), ylim=c(50,120))

model6 <- lm(y3~(x3^2))
summary(model6)
eq6 <- function(x) { 112.36 - 1.09 *(x^2)}
par(new=TRUE)
plot(eq6, xlab="", ylab = "", xlim=c(5,45), ylim=c(50,120))

#multiple Linear regression

linMod=lm(IQ~Income + Education_expenditure_per_inhabitant +Daily_max_temperature  , df)
attributes(linMod)   
summary(linMod) 

modelSummary <- summary(linMod) 
 
confint(linMod, level=0.95)

 
Rsquared=modelSummary$r.squared

 
par(mfrow=c(2,2)) # If you would like to have the 4 plots on the same page
plot(linMod)
