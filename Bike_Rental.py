#!/usr/bin/env python
# coding: utf-8

# In[83]:


import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency
import matplotlib as plt
import matplotlib.pyplot as plt2
import seaborn as sns
import sys


# In[84]:


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[85]:


os.chdir("G:/Project/Bike-Sharing-Dataset")


# In[86]:


bike_sharing_train = pd.read_csv("day.csv")


# In[87]:


print('Shape of our dataset:')
print(bike_sharing_train.shape,'\n')


# ## Exploratory Data Analysis

# In[88]:


print('*'*25,'Exploratory Data Analysis: ','*'*25,'\n')


# In[89]:


print('Column / Variable Names:')
print(bike_sharing_train.columns)


# In[90]:


# Showing 1st few rows of our dataset
print('Showing 1st few rows of our dataset: \n')
print(bike_sharing_train.head(5))


# In[91]:


print("Basic info about dataset:\n")
print(bike_sharing_train.info())


# In[92]:


print("Checking the data types of the variables:\n")
print(bike_sharing_train.dtypes,'\n')


# In[93]:


print("Converting the varibales to it's proper data type: \n\nAfter Convertion:\n")
bike_sharing_train['season'] = bike_sharing_train['season'].astype('category')
bike_sharing_train['yr'] = bike_sharing_train['yr'].astype('category')
bike_sharing_train['mnth'] = bike_sharing_train['mnth'].astype('category')
bike_sharing_train['weekday'] = bike_sharing_train['weekday'].astype('category')
bike_sharing_train['workingday'] = bike_sharing_train['workingday'].astype('category')
bike_sharing_train['weathersit'] = bike_sharing_train['weathersit'].astype('category')
bike_sharing_train['holiday'] = bike_sharing_train['holiday'].astype('category')

bike_sharing_train['temp'] = bike_sharing_train['temp'].astype('float')
bike_sharing_train['atemp'] = bike_sharing_train['atemp'].astype('float')
bike_sharing_train['hum'] = bike_sharing_train['hum'].astype('float')
bike_sharing_train['windspeed'] = bike_sharing_train['windspeed'].astype('float')
bike_sharing_train['cnt'] = bike_sharing_train['cnt'].astype('float')

print(bike_sharing_train.dtypes,'\n')


# In[94]:


categorical = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
print('Count of each categorical variable in our data is as follows:\n')
[print(bike_sharing_train[i].value_counts(),'\n\n') for i in categorical]


# In[95]:


# Count of each category of a categorical variable
print("Checking count of each category of categorical variables in dataset\n\n")
sns.set_style("whitegrid")

def check_count_of_category(categorical_var):
    ax = sns.factorplot(data=bike_sharing_train, x=categorical_var, kind= 'count',size=3,aspect=2)
    title = "Count of each category of "+categorical_var+" variable in 2years"
    plt2.title(title)
    plt2.show()
[check_count_of_category(i) for i in categorical]


# ##  Univariate & Bivariate analysis

# In[96]:


numeric = ['temp','atemp','hum','windspeed','casual','registered','cnt']
print("Descriptive statistics about the numeric columns:")
print(bike_sharing_train[numeric].describe(),'\n')


# In[97]:


print("Univariate analysis of numerical variables")
def dist_plot(i):
    sns.distplot(bike_sharing_train[i])
    title = "Distribution of "+ i + " variable"
    plt2.title(title)
    plt2.show()
    
num = ['temp','atemp','hum','windspeed']   

[dist_plot(i) for i in num]


# In[98]:


print("Bivariate analysis of numerical variables")
sns.pairplot(bike_sharing_train[numeric])
plt2.show()


# ### For Month

# In[99]:


# Month
fig, ax = plt2.subplots(nrows = 1, ncols = 1, figsize= (9,5), squeeze=False)

x1 = 'mnth'
y1='cnt'

sns.barplot(x= x1, y = y1, data = bike_sharing_train, ax=ax[0][0])
title = "Demand of bikes in different months"
plt2.title(title)
plt2.show()


# In[100]:


yr_0 = bike_sharing_train.loc[bike_sharing_train['yr'] == 0]
yr_1 = bike_sharing_train.loc[bike_sharing_train['yr'] == 1]


# In[101]:


fig, ax = plt2.subplots(nrows = 1, ncols = 2, figsize= (12,4), squeeze=False)
fig.suptitle("Demand of bikes in different months in year 2011 & 2012")
x1 = 'mnth'
y1='cnt'
sns.barplot(x= x1, y = y1, data = yr_0, ax=ax[0][0])
sns.barplot(x= x1, y = y1, data = yr_1, ax=ax[0][1])
plt2.show()


# ### For Weekday

# In[102]:


fig, ax = plt2.subplots(nrows = 1, ncols = 1, figsize= (9,4), squeeze=False)
fig.suptitle("Demand of bikes in different days of a week")
x1 = 'weekday'
y1='cnt'

sns.barplot(x= x1, y = y1, data = bike_sharing_train, ax=ax[0][0])


# In[103]:


print("From figures we can categorize 5-10th month as one category and rest months as another category.\n")
print("Similarly,in weekday variables; workindays can be categorized as one and weekends as another category. As in working days demand of bikes found high than weekends.\n")


# In[104]:


# Keep on adding the unwanted variables (that we will get by applying different techniques) to remove list and 
# will finally we will remove from our dataset
remove = ['instant','dteday']


# ## Missing Value Analysis

# In[105]:


print('*'*25,'Missing Value Analysis: ','*'*25,'\n')


# In[106]:


missing_val = pd.DataFrame(bike_sharing_train.isnull().sum())


# In[107]:


print('Missing values in our dataset: \n')
print(missing_val)


# In[108]:


print("No missing values present in our dataset")


# ## Outlier Analysis

# In[109]:


print('*'*25,'Outlier Analysis','*'*25,'\n')


# In[110]:


# Check for outliers in data using boxplot
sns.boxplot(data=bike_sharing_train[['temp','atemp','windspeed','hum']])
fig=plt2.gcf()
title = "Checking outliers in numerical variables"
plt2.title(title)
fig.set_size_inches(6,6)


# In[111]:


print("Outliers found in windspeed and humidity variable")


# In[112]:


# Numeric Variables
num = ['temp','atemp','hum','windspeed']


# In[113]:


# Removing the outliers
for i in num:
    q75, q25 = np.percentile(bike_sharing_train[i], [75, 25])
    iqr = q75 - q25
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    
    bike_sharing_train = bike_sharing_train.drop(bike_sharing_train[bike_sharing_train.loc[:,i] < minimum].index)
    bike_sharing_train = bike_sharing_train.drop(bike_sharing_train[bike_sharing_train.loc[:,i] > maximum].index)
print("Outliers removed")


# ## Feature Engineering

# In[114]:


bike_sharing_train.head()


# In[115]:


categorical = ['season','yr','mnth','holiday','weekday','workingday','weathersit']


# In[116]:


# Creating new variables  through binning
def binned_month(row):
    if row['mnth'] <= 4 or row['mnth'] >=11:
        return(0)
    else:
        return(1)
    
def binned_weekday(row):
    if row['weekday'] < 2:
        return(0)
    else:
        return(1)


# In[117]:


bike_sharing_train['month_binned'] = bike_sharing_train.apply(lambda row : binned_month(row), axis=1)
bike_sharing_train = bike_sharing_train.drop(columns=['mnth'])
bike_sharing_train['weekday_binned'] = bike_sharing_train.apply(lambda row : binned_weekday(row), axis=1)
bike_sharing_train = bike_sharing_train.drop(columns=['weekday'])


# In[118]:


categorical.remove('mnth')
categorical.remove('weekday')
categorical.append('month_binned')
categorical.append('weekday_binned')


# ## Feature Selection

# In[119]:


bike_sharing_train.columns


# ### Correlation Analysis

# In[120]:


df_corr = bike_sharing_train[numeric]


# In[121]:


# Correlation Analysis
f, ax = plt2.subplots(figsize=(14, 10))
corr = df_corr.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt2.title("Correlation analyss of numerical variables through heatmap")


# In[122]:


print("From correlation analysis we found,\n    1.temp and atemp are highly correlated.\n    2.registered and cnt also showing high correlation.\n")


# In[123]:


remove.extend(['atemp','casual','registered'])


# ### Chi-square test

# In[124]:


print("Chi-square Test\n1. Null Hypothesis: Two variables are independent\n2. Alternate Hypothesis: Two variables are not independent\n3. p-value < 0.05 , can not accept null hypothesis\n")
print("That means p < 0.05 means two categorical variables are dependent, so we will remove one of variable from that pair to avoid sending the same information to our model through 2 variables")


# In[125]:


# Create all combinations 
factors_paired = [(i,j) for i in categorical for j in categorical] 


# In[126]:


# Calculating p-values for each pair
p_values = []
from scipy.stats import chi2_contingency
for factor in factors_paired:
    if factor[0] != factor[1]:
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(bike_sharing_train[factor[0]], 
                                                    bike_sharing_train[factor[1]]))
        if(p<0.05):
            p_values.append({factor:p.round(3)})
    else:
        p_values.append('-')

[print(i,'\n') for i in p_values if i != '-']


# In[127]:


print("Season with Weathersit-Month\nHoliday with Worikingday-Weekday\nWorkingday with Weekday-Holiday\n")


# In[128]:


bike_sharing_train.columns


# ### Importance of Features

# In[129]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0, n_jobs=-1)
X = bike_sharing_train.drop(columns=['cnt','casual','registered','instant','dteday'])
y = bike_sharing_train['cnt']
model = clf.fit(X, y)
importances = model.feature_importances_


# In[130]:


X.columns


# In[131]:


print("Checking feature importance: \n")
l = list(zip(X,importances))
l.sort(key = lambda x: x[1]) 
[print(i[0]," : ",i[1].round(3)) for i in l]


# In[132]:


remove.append('holiday')


# ### Multi-colinearity test

# In[133]:


from statsmodels.stats.outliers_influence import variance_inflation_factor as vf
from statsmodels.tools.tools import add_constant
numeric_df = add_constant(bike_sharing_train[['temp', 'atemp', 'hum', 'windspeed']])
vif = pd.Series([vf(numeric_df.values, j) for j in range(numeric_df.shape[1])],index = numeric_df.columns)
vif.round(3)


# In[134]:


print("After removing atemp variable , VIF:\n")
numeric_df = add_constant(bike_sharing_train[['temp', 'hum', 'windspeed']])
vif = pd.Series([vf(numeric_df.values, i) for i in range(numeric_df.shape[1])], 
                 index = numeric_df.columns)
print(vif.round(3))


# ### Dummy for categorical

# In[135]:


season_dm = pd.get_dummies(bike_sharing_train['season'], drop_first=True, prefix='season')
bike_sharing_train = pd.concat([bike_sharing_train, season_dm],axis=1)
bike_sharing_train = bike_sharing_train.drop(columns = ['season'])
weather_dm = pd.get_dummies(bike_sharing_train['weathersit'], prefix= 'weather',drop_first=True)
bike_sharing_train = pd.concat([bike_sharing_train, weather_dm],axis=1)
bike_sharing_train = bike_sharing_train.drop(columns= ['weathersit'])


# In[136]:


remove


# In[137]:


# Removing unwanted variables
bike_sharing_train.drop(columns=remove, inplace=True)


# In[138]:


# Reshaping
cnt = bike_sharing_train['cnt']
bike_sharing_train = bike_sharing_train.drop(columns=['cnt'])
bike_sharing_train['cnt'] = cnt


# In[145]:


bike_sharing_train.shape


# In[141]:


print(bike_sharing_train.head(5), '\n')
print('shape of dataset after all pre-processing\n',bike_sharing_train.shape)


# ### Model Development

# In[70]:


# Modularizing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

def fit_N_predict(model, X_train, y_train, X_test, y_test,model_code=''):
    
    if(model_code == 'OLS'):
        model = model.OLS(y_train,X_train.astype('float')).fit()
        print(model.summary())
        y_pred = model.predict(X_test.astype('float'))
        print("\n================================")
        print('Score on testing data: ',(r2_score(y_test,y_pred)*100).round(3))
        print("================================")
        return
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("================================")
    print("Score on training data: ",(model.score(X_train, y_train)*100.0).round(3))
    print("================================")
    print("Score on testing data: ", (model.score(X_test, y_test)*100.0).round(3)) ## Same as r-squared value
    print("================================")
    
    if(model_code == "DT"):
        from sklearn import tree
        dotfile = open("pt.dot","w")
        df = tree.export_graphviz(model, out_file=dotfile, feature_names = X_train.columns)
    
    
    
        


# In[71]:


from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from statistics import mean 
kf = KFold(n_splits=10, shuffle=True, random_state=42)


def cross_validation(model,X,y):
    l = []
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        l.append(r2_score(y_test,y_pred))
    print("Mean of 10 cross validation scores = ",(mean(l)*100).round(3))


# In[72]:


# Partitioning of dataset
X = bike_sharing_train.drop(columns=['cnt'])
y = bike_sharing_train[['cnt']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[73]:


print('######################### LINEAR REGRESSION #########################')
from sklearn.linear_model import LinearRegression

model = LinearRegression()
fit_N_predict(model, X_train, y_train, X_test, y_test, model_code="SK_LR")
cross_validation(model,X,y)


# In[74]:


import statsmodels.api as sm
model = sm
fit_N_predict(model,X_train, y_train, X_test, y_test,model_code="OLS")


# In[279]:


print('######################### K NEIGHBOURS REGRESSOR #########################')
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)
fit_N_predict(model, X_train, y_train, X_test, y_test,model_code="KNN")
cross_validation(model,X,y)


# In[282]:


print('######################### SVR #########################')
from sklearn.svm import SVR
model = SVR(kernel = 'linear')
fit_N_predict(model, X_train, y_train, X_test, y_test, model_code="SVR")
cross_validation(model,X,y)


# In[75]:


print('######################### Decision Tree #########################')

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)
fit_N_predict(model, X_train, y_train, X_test, y_test, model_code="DT")
cross_validation(model,X,y)


# In[141]:


print('######################### Random Forest #########################')

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=1)
fit_N_predict(model, X_train, y_train, X_test, y_test, model_code="RF")
cross_validation(model,X,y)


# In[144]:


# Pre-processing of data for XGBoost
bike = bike_sharing_train.copy()
yr_dm = pd.get_dummies(bike['yr'], prefix= 'yr',drop_first=True)
bike = pd.concat([bike, yr_dm],axis=1)
bike = bike.drop(columns= ['yr'])

workingday_dm = pd.get_dummies(bike['workingday'], prefix= 'workingday',drop_first=True)
bike = pd.concat([bike, workingday_dm],axis=1)
bike = bike.drop(columns= ['workingday'])

bike['yr_1'] = bike['yr_1'].astype('int')
bike['workingday_1'] = bike['workingday_1'].astype('int')

X1 = bike.drop(columns=['cnt'])
y1 = bike['cnt']
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.2, random_state = 0)


# In[145]:


print('######################### XGB Regressor #########################')

from xgboost import XGBRegressor
model = XGBRegressor(random_state=1)
fit_N_predict(model, X_train1, y_train1, X_test1, y_test1, model_code="XGB")
cross_validation(model,X1,y1)


# ## Hyper-parameter tuning for the best models

# In[147]:


print('######################### Tuning Random Forest #########################')

from sklearn.model_selection import GridSearchCV
model = RandomForestRegressor(random_state=1)
params = [{'n_estimators' : [400, 500, 600],
           'max_features': ['auto', 'sqrt', 'log2'],
           'min_samples_split': [2,4,6],
           'max_depth': [10, 12, 14],
           'min_samples_leaf': [2,3,5],
           'random_state' : [1]
          }]
grid_search = GridSearchCV(estimator=model, param_grid=params,cv = 5,
                           scoring = 'explained_variance', n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
print('Best parameters for Random Forest',grid_search.best_params_)


# In[148]:


# Developing Random Forest model with best params
model = RandomForestRegressor(random_state=1, max_depth=14, n_estimators=500,
                                  max_features='auto', min_samples_leaf=2,min_samples_split=2)
fit_N_predict(model, X_train, y_train, X_test, y_test, model_code="RF")
cross_validation(model,X,y)


# In[149]:


print('######################### Tuning XGBoost #########################')
model = XGBRegressor(random_state=1)
params = [{'n_estimators' : [200, 250, 300,350, 400,450], 
           'max_depth':[2, 3, 5], 
           'learning_rate':[0.01, 0.045, 0.05, 0.055, 0.1, 0.3],
           'gamma':[0, 0.001, 0.01, 0.03],
           'subsample':[1, 0.7, 0.8, 0.9],
           'random_state' :[1]
          }]
grid_search = GridSearchCV(estimator=model, param_grid=params,cv = 5,
                           scoring = 'explained_variance', n_jobs=-1)
grid_search = grid_search.fit(X_train1, y_train1)
print('Best parameters for XGBoost',grid_search.best_params_)


# In[150]:


# Developing XGBoost model with best params
model = XGBRegressor(random_state=1, learning_rate=0.045, max_depth=3, n_estimators=300, 
                         gamma = 0, subsample=0.7)
fit_N_predict(model, X_train1, y_train1, X_test1, y_test1, model_code="XGB")
cross_validation(model,X1,y1)


# In[151]:


# Scatterplot showing the prediction vs actual values for the best model for our dataset

model = XGBRegressor(random_state=1, learning_rate=0.045, max_depth=3, n_estimators=300, 
                     gamma = 0, subsample=0.7)
model.fit(X_train1,y_train1)
y_pred = model.predict(X_test1)
fig, ax = plt2.subplots(figsize=(7,5))
ax.scatter(y_test1, y_pred)
ax.plot([0,8000],[0,8000], 'r--', label='Perfect Prediction')
ax.legend()
plt2.title("Scatter plot between y_test and y_pred")
plt2.xlabel("y_test")
plt2.ylabel("y_pred")
plt2.tight_layout()
plt2.show()

