# # Clean the Environment
# rm(list = ls())
# 
# # Set Working Directory
setwd("G:/Project/Bike-Sharing-Dataset")

# Load the libraries
libraries = c("plyr","dplyr","ggplot2","rpart","DMwR","randomForest","usdm","corrgram","DataCombine","scales","corrplot","dummies","caTools","grid")

lapply(X = libraries,require, character.only = TRUE)

rm(libraries)

# Read the csv file

bike_sharing_train = read.csv(file = "day.csv", header = T, sep=",",na.strings = c(" ", "", "NA"))


############################## Exploratory Data Analysis ##################

# Shape of our dataset
dim(bike_sharing_train)

# Column/Variable Names
names(bike_sharing_train)

# Showing First few rows of dataset
head(bike_sharing_train,5)

# Basic info about dataset
str(bike_sharing_train)

# Checking the data types of the variables
sapply(bike_sharing_train,class)

# Converting the varibales to it's proper data type
categorical <- c('season','yr','mnth','holiday','weekday','workingday','weathersit')

for(i in categorical){
  bike_sharing_train[,i] = as.factor(bike_sharing_train[,i])
}

# After Convertion
str(bike_sharing_train)

# Count of each categorical variable in our data is as follows
sapply(bike_sharing_train[,categorical],table)

# Count of each category of a categorical variable
check_count_of_category <- function(cat){
  
  ggplot(bike_sharing_train, aes_string(x = bike_sharing_train[,cat], fill = bike_sharing_train[,cat])) +
    geom_bar(stat="count") + theme_bw() +
    xlab(cat) + ylab('Count') +
    ggtitle(paste("Count of each category of ",cat," variable in 2years")) +  theme(text=element_text(size=10))
  
}

all_bar_plot <- lapply(categorical,check_count_of_category)

gridExtra::grid.arrange(all_bar_plot[[1]],all_bar_plot[[2]],all_bar_plot[[3]],all_bar_plot[[4]],
                        all_bar_plot[[5]],all_bar_plot[[6]],all_bar_plot[[7]],ncol=3,nrow=3,
                        top=textGrob("Count of each category in Categorical Variables in our data",gp=gpar(fontsize=22,font=8)))

############################## Univariate & Bivariate Analysis  ##################


numeric <- c('temp','atemp','hum','windspeed','casual','registered','cnt')

# Descriptive statistics about the numeric columns
summary(bike_sharing_train[,numeric])


# Univariate analysis of numerical variables
############### Distribution of temp variable ###################

ggplot(data = bike_sharing_train, aes(x=temp)) + 
  geom_histogram(aes(y=..density..),
                 binwidth=.04,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666") +
  ggtitle('Distribution of Temperature Variable') +
  theme(plot.title = element_text(hjust = 0.5,size=22))

############### Distribution of atemp variable ###################

ggplot(data = bike_sharing_train, aes(x=atemp)) + 
  geom_histogram(aes(y=..density..),      
                 binwidth=.04,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666") +
  ggtitle('Distribution of atemp Variable') +
  theme(plot.title = element_text(hjust = 0.5,size=22))

############### Distribution of humidity variable ###################

ggplot(data = bike_sharing_train, aes(x=hum)) + 
  geom_histogram(aes(y=..density..),      
                 binwidth=.04,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +
  ggtitle('Distribution of Humidity Variable') +
  theme(plot.title = element_text(hjust = 0.5,size=22))



############### Distribution of windspeed variable ###################

ggplot(data = bike_sharing_train, aes(x=windspeed)) + 
  geom_histogram(aes(y=..density..),      
                 binwidth=.04,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666") +
  ggtitle('Distribution of Windspeed Variable') +
  theme(plot.title = element_text(hjust = 0.5,size=22))


# Bivariate analysis of numerical variables
# Pair plot
pairs(bike_sharing_train[,10:16])

# For Month
ggplot(data=bike_sharing_train, aes(x=mnth, y=cnt)) +
  geom_bar(stat="identity",fill = 'steelblue') +
  ggtitle('Demand of bikes in different months') +
  theme(plot.title = element_text(hjust = 0.5,size=22))

yr1 = bike_sharing_train[which(bike_sharing_train$yr == 0),]
yr2 = bike_sharing_train[which(bike_sharing_train$yr == 1),]

yr1_plot = ggplot(data=yr1, aes(x=mnth, y=cnt)) +
  geom_bar(stat="identity",fill = 'steelblue') 

yr2_plot = ggplot(data=yr2, aes(x=mnth, y=cnt)) +
  geom_bar(stat="identity",fill = 'red') 

gridExtra::grid.arrange(yr1_plot,yr2_plot,ncol=2,top=textGrob("Demand of bikes in different months in year 2011 & 2012",gp=gpar(fontsize=22,font=8)) )

# For Weekday
ggplot(data=bike_sharing_train, aes(x=weekday, y=cnt)) +
  geom_bar(stat="identity",fill = 'steelblue') +
  ggtitle('Demand of bikes in different days of a week') +
  theme(plot.title = element_text(hjust = 0.5,size=22))

cat("1. From figures we can categorize 5-10th month as one category and rest months as another category.\n2. Similarly,in weekday variables; workindays can be categorized as one and weekends as another category. As in working days demand of bikes found high than weekends.\n")


# Keep on adding the unwanted variables (that we will get by applying different techniques) to remove list and 
# will finally we will remove from our dataset
remove = list("instant","dteday")
head(bike_sharing_train,2)

############################ Missing Value Analysis ###############

print("############################ Missing Value Analysis ###############")
missing_values <- sapply(bike_sharing_train, function(x){sum(is.na(x))})
print(missing_values)

print("No missing value found")

############################Outlier Analysis####################
print("############################ Outlier Analysis ###############")

names(bike_sharing_train)

n_names = colnames(bike_sharing_train[,c("temp","atemp","windspeed","hum")])

for (i in 1:length(n_names))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (n_names[i])), data = subset(bike_sharing_train))+
           stat_boxplot(geom = "errorbar", width =1) +
           geom_boxplot(outlier.colour="red", fill = "blue" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=n_names[i],x="cnt")+
           ggtitle(paste("Box plot of ",n_names[i])))
}
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=4,top="Checking outliers in numerical variables")

print("Outliers found in windspeed and humidity variable")

#Removing Outliers
for(i in n_names){
  print(i)
  val = bike_sharing_train[,i][ bike_sharing_train[,i] %in% boxplot.stats( bike_sharing_train[,i] )$out  ]
  bike_sharing_train = bike_sharing_train[which(!bike_sharing_train[,i] %in% val),]
}

print("Outliers removed")

########################################### Feature Enginnering ########################################################

head(bike_sharing_train,2)

categorical


bike_sharing_train$mnth = as.numeric(bike_sharing_train$mnth)
bike_sharing_train$weekday = as.numeric(bike_sharing_train$weekday)

# Creating new variables  through binning
bike_sharing_train = transform(bike_sharing_train, mnth = case_when(
  mnth <= 4 ~ 0,
  mnth >= 11 ~ 0,
  TRUE   ~ 1
))
colnames(bike_sharing_train)[5] <- 'month_binned'


bike_sharing_train = transform(bike_sharing_train, weekday = case_when(
  weekday < 2 ~ 0,
  TRUE   ~ 1
))
colnames(bike_sharing_train)[7] <- 'weekday_binned'

categorical[3] <- 'month_binned'
categorical[5] <- 'weekday_binned'

################################################# Feature Selection ##################################################################

# correlation plot for numerical feature
corrgram(bike_sharing_train[,numeric], order = FALSE,
         upper.panel = panel.pie, text.panel = panel.txt,
         main = "Correlation analyss of numerical variables")

# heatmap plot for numerical features
corrplot(cor(bike_sharing_train[,numeric]), method = 'color',title ="Correlation analyss of numerical variables through heatmap" ,col=colorRampPalette(c("blue","red"))(200))

cat("From correlation analysis we found,\n    1.temp and atemp are highly correlated.\n    2.registered and cnt also showing high correlation.\n")

remove = append(remove,list('atemp','casual','registered'))

# Chi-Square Test
cat("Chi-square Test\n1. Null Hypothesis: Two variables are independent\n2. Alternate Hypothesis: Two variables are not independent\n3. p-value < 0.05 , can not accept null hypothesis\nThat means p < 0.05 means two categorical variables are dependent, so we will remove one of variable from that pair to avoid sending the same information to our model through 2 variables")

# Create all combinations 
factors_paired <- combn(categorical, 2, simplify = F)

for(i in factors_paired){
  print("############### START ######################")
  print(i)
  print(chisq.test(table(bike_sharing_train[,i[1]], bike_sharing_train[,i[2]])))
  print("################ END #####################")
}

cat("Dependecy:\nSeason with Weathersit-Month\nHoliday with Worikingday-Weekday\nWorkingday with Weekday-Holiday\n")

# finding important features
data = bike_sharing_train[,-c(1,2,14,15)]
importances <- randomForest(cnt ~ ., data = data,
                            ntree = 200, keep.forest = FALSE, importance = TRUE)

important_features <- data.frame(importance(importances, type = 1))

remove = append(remove,'holiday')

# Multi-colinearity test
# vif = 1 / (1-R2)
vif(bike_sharing_train[,c(10,11,12,13)],)

# After removing atemp variable 
vif(bike_sharing_train[,c(10,12,13)],)

# Dummy for categorical

season_dummy = dummy(bike_sharing_train$season, sep = "_")
weather_dummy = dummy(bike_sharing_train$weathersit, sep ="_" )
bike_sharing_train = cbind(bike_sharing_train,season_dummy)
bike_sharing_train = cbind(bike_sharing_train,weather_dummy)
bike_sharing_train[,c("season","weathersit","season_1","weathersit_1")] = NULL

remove
remove = unlist(remove,use.names=FALSE)

# Removing unwanted variables
bike_sharing_train[,remove] = NULL
dim(bike_sharing_train)

cnt = bike_sharing_train[,"cnt"]
bike_sharing_train$cnt = NULL

bike_sharing_train = cbind(bike_sharing_train,cnt)


##########################################################################################################################################
rmExcept(c("bike_sharing_train"))
################################################# Model Development ##################################################################

fit_N_predict <- function(method, train_data, test_data,bike, model_code = ""){
  model_fit <- caret::train(cnt~., data = train_data, method = method)
  
  y_pred <- predict(model_fit, train_data[,-13])
  print("================================")
  print("Score on training data: ")
  print(caret::R2(y_pred, train_data[,13]))
  
  y_pred <- predict(model_fit, test_data[,-13])
  print("================================")
  print("Score on test dataset: ")
  print(caret::R2(y_pred, test_data[,13]))
  
  print("================================")
  
  if(model_code == 'lm'){
    print(summary(model_fit))
  }
}

cross_validation <- function(method, bike){
  ten_folds = createFolds(bike$cnt, k = 10)
  ten_cv = lapply(ten_folds, function(fold) {
    training_fold = bike[-fold, ]
    test_fold = bike[fold, ]
    model_fit <- caret::train(cnt~., data = training_fold, method = method)
    y_pred <- predict(model_fit, test_fold[,-13])
    return(as.numeric(caret::R2(y_pred, test_fold[,13]))) 
  })
  sum = 0
  for(i in ten_cv){
    sum = sum + as.numeric(i)
  }
  print("Mean of 10 cross validation scores = ")
  
  print(sum/10)
}

set.seed(42)
split = sample.split(bike_sharing_train$cnt, SplitRatio = 0.80)
train_set = subset(bike_sharing_train, split == TRUE)
test_set = subset(bike_sharing_train, split == FALSE)
X_train = train_set[,1:12]
y_train = train_set[,13]
X_test = test_set[,1:12]
y_test = test_set[,13]


# Model Development


# 1. LINEAR REGRESSION
print("=============== LINEAR REGRESSION =================")

fit_N_predict('lm', train_set, test_set,bike_sharing_train, model_code="lm")
cross_validation('lm',bike_sharing_train)

# 2. KNN
print("=============== KNN =================")
fit_N_predict('knn', train_set, test_set,bike_sharing_train)
cross_validation('knn',bike_sharing_train)


# 3. SVM
print("=============== SVM  =================")
fit_N_predict('svmLinear3', train_set, test_set,bike_sharing_train)
cross_validation('svmLinear3',bike_sharing_train)


# 4. DECISION TREE
print("=============== DECISION TREE  =================")
fit_N_predict('rpart2', train_set, test_set,bike_sharing_train)
cross_validation('rpart2',bike_sharing_train)


# 5. RANDOM FOREST
print("=============== RANDOM FOREST  =================")
fit_N_predict('rf', train_set, test_set,bike_sharing_train)
cross_validation('rf',bike_sharing_train)


# 6. XGBoost
print("=============== XGBoost  =================")
fit_N_predict('xgbTree', train_set, test_set,bike_sharing_train)
cross_validation('xgbTree',bike_sharing_train)


##################################################### Hyper-parameter tuning for the best models ##############################################

print('######################### Tuning Random Forest #########################')

control <- trainControl(method="cv", number=10, repeats=3)
model_fit <- caret::train(cnt~., data = train_set, method = "rf",trControl = control)
model_fit$bestTune
y_pred <- predict(model_fit, test_set[,-13])
print(caret::R2(y_pred, test_set[,13]))

print('######################### Tuning XGBoost #########################')


control <- trainControl(method="cv", number=10, repeats=3)
model_fit <- caret::train(cnt~., data = train_set, method = "xgbTree",trControl = control)
model_fit$bestTune
y_pred <- predict(model_fit, test_set[,-13])
print(caret::R2(y_pred, test_set[,13]))


