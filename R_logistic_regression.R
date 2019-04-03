library(ISLR)
names(Smarket)
summary(Smarket)
cor(Smarket[, -9])  # check correlation between columns (except column "Direction")

attach(Smarket)  # by attaching the data into R path, we can call columns directly
plot(Volume)

# when using glm (generalized linear model), we need to specify family=binomial,
## so that R will know this is logistic regression
glm_fits<-glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume , data=Smarket,family=binomial)
summary(glm_fits)
# Direction is the target here
# lag1 has lowest p-value with negative coefficient, 
# indicates that if there is positive return yesterday, the direction is less likely to go up today
# However, 0.145 p-value is still large enough to reject NULL hypothese, which means there is
## no clear correlation between lag1 and Direction.

glm_probs <- predict(glm_fits, type="response")
glm_probs[0:10]
