setwd("C:/Users/reshm/OneDrive/Desktop/CETM46Assgnment - II")

#Assignment II cetm46

library(caret)
library(ggplot2)
library(tidyverse)
library(robustbase) # to handle outliers
library(glue)
library(stats)

#Reading the data set for Maths and Portuguese students

mathstud <- read.csv("Maths_students.csv", sep = ",")

porstud <- read.csv("Por_students.csv", sep = ',')

##BASIC DATA EXPLORATION

head(mathstud)

#To find the structure and type of attributes in the datasets

str(mathstud)
str(porstud)

#Mathstud has 395 observations and 33 variables
#porstud has 649 observations and 33 variables

#To find if there are common students in the datasets

common <- merge(mathstud,porstud, by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(common))

##This shows there are 382 common students in both Maths and Portuguese datasets

#To find the distribution of attributes in the dataset, we can use summary()

summary(mathstud)

# Some of the findings in the ##mathstud dataset##

## Mean Age = 16.7
## Mean Studytime = 2.035
## Mean health = 3.554
##Mean absences =  5.709
##Maximum Absences = 75

##mean G1 score = 10.91
##mean G2 score = 10.71
##mean G3 score = 10.42

summary(porstud)

# Some of the findings in the ##porstud dataset##

## Mean Age = 16.74
## Mean Studytime = 1.93
## Mean health = 3.536
##Mean absences =  3.659
##Maximum Absences = 32

##mean G1 score = 11.4
##mean G2 score = 11.57
##mean G3 score = 11.91

#To find if there are any missing values in the dataset

sum(is.na(mathstud))
sum(is.na(porstud))

# There are no missing values in the datasets

#To check if there are any duplicate rows in the datasets 

sum(duplicated(mathstud))
sum(duplicated(porstud))
sum(duplicated(common))
#There are no duplicated rows in the datasets
mathstud$school <- as.factor(mathstud$school)
porstud$school <- as.factor(porstud$school)
mathstud$sex <-  as.factor(mathstud$sex)
porstud$sex <-  as.factor(porstud$sex)
mathstud$G1 <- as.numeric(mathstud$G1)
mathstud$G2 <- as.numeric(mathstud$G2)
mathstud$G3 <- as.numeric(mathstud$G3)
mathstud$internet <- as.factor(mathstud$internet)
porstud$internet <-as.factor(porstud$internet)


#Basic DataViz for Dataexploration


hist(mathstud$age)
hist(mathstud$G1)
hist(mathstud$G2)
hist(mathstud$G3)
plot(porstud$school, porstud$G3)
#We can see that there are some outliers for G3 grade of Portuguese
# Calculate the median and interquartile range of the G3 column
Q1 <- quantile(porstud$G3, 0.25)
Q3 <- quantile(porstud$G3, 0.75)
IQR <- Q3 - Q1

# Define the upper and lower bounds for outliers
upper <- Q3 + 1.5 * IQR
lower <- Q1 - 1.5 * IQR

# Remove the outliers from the G3 column
porstud <- porstud[porstud$G3 >= lower & porstud$G3 <= upper, ]

#checking if the outliers are reduced
plot(porstud$school, porstud$G3)

#we can see the outliers are reduced

#Creating a new variable to show the average Maths and Portuguese score from G1, G2 and G3

maths_avg <- rowMeans(mathstud[, c("G1", "G2","G3")])
mean(maths_avg)

por_avg <- rowMeans(porstud[, c("G1", "G2","G3")])
mean(por_avg)

hist(por_avg)
hist(maths_avg)
shapiro.test(mathstud$G1)
shapiro.test(mathstud$G2)
shapiro.test(mathstud$G3)


## since p-value is less than 0.05, G1, G2 and G3 grades are not normally distributed.


#School

plot(mathstud$school, maths_avg, main = "Peformance comparison of GP and MS schools", ylab = "Maths Average Score")
plot(porstud$school, por_avg, main = "Performance comparison of GP and MS school", ylab = "Portuguese Average Score") 
#GP school performs slightly better than MS for Maths. However, the performance is significantly better in Portuguese


plot(mathstud$sex, maths_avg, main="Male and Female student performance in Maths", ylab = " Maths Average Score")
plot(porstud$sex, por_avg,main="Male and Female student performance in Portuguese", ylab = " Portuguese Average Score") 

#From these plots, it can be denoted that Female students perform better for Portugese languages while Male students perform better in Maths. 

#Internet access

internet_students <- table(mathstud$internet)
barplot(internet_students, main = "Internet Access of Students", xlab = "Internet Access", ylab = "Number of Students",
        col = c("#55B4B0", "#EFC050"), border = "white", font.main = 1, cex.main = 1.5, font.lab = 1, cex.lab = 1.2)

plot(mathstud$internet, maths_avg, main="Does internet affect performance in Maths?", ylab="Maths Average Score", col=c("#55B4B0", "#EFC050")[mathstud$internet])
axis(side=1, at=c(1,2), labels=c("No Internet", "Yes Internet"))

plot(porstud$internet, por_avg, main="Does internet affect performance in Portuguese?", ylab="Portuguese Average Score", col=c("#55B4B0", "#EFC050")[porstud$internet])
axis(side=1, at=c(1,2), labels=c("No Internet", "Yes Internet"))

#From these plots, it is clear that Access to internet has a positive affect on the performance of students in Maths and Portuguese


inte <- table(mathstud$internet)
barplot(internet_students, main = "Internet Access of Students", xlab = "Internet Access", ylab = "Number of Students",
        col = c("#55B4B0", "#EFC050"), border = "white", font.main = 1, cex.main = 1.5, font.lab = 1, cex.lab = 1.2)

####

##using Linear regression


# Separating the target variable (G3 grades) and the predictors
X <- mathstud[, c("G1", "G2", "absences", "Fedu", "Medu", "internet", "sex", "Pstatus", "guardian")]
Y <- mathstud$G3

# Fitting the linear regression model
model <- lm(Y ~ G1 + G2 + absences + Fedu + Medu + internet + sex + Pstatus + guardian, data = mathstud)

# Summary of the model to see the coefficients and p-values
summary(model)

#From the linear regression, we can see that the Grade G3 has a very low p-value against G1 and G2. Therefore, the grades students got in G1 and G2 are significant for the grade G3. 
#The level of absense also has a significant influence on the grade G3

#Socio-economic factors such as Fedu, Medu, Internet etc has low significance on the performance



