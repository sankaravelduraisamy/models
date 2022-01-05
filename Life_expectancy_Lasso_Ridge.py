# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:29:20 2021

@author: Dathu
"""


# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
life = pd.read_csv("C:/data science/lasso ridge/Datasets_LassoRidge/Life_expectencey_LR.csv")

# Rearrange the order of the variables
life = life.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
life.columns

life1 = life.dropna()

# Mean Imputation - CLMAGE is a continuous data

mean_value = life1.Life_expectancy.mean()
mean_value
life1.Life_expectancy = life1.Life_expectancy.fillna(mean_value)
life1.Life_expectancy.isna().sum()

mean_value = life1.Alcohol.mean()
mean_value
life1.Alcohol = life1.Alcohol.fillna(mean_value)
life1.Alcohol.isna().sum()

mean_value = life1.percentage_expenditure.mean()
mean_value
life1.percetage_expenditure = life1.percentage_expenditure.fillna(mean_value)
life1.percentage_expenditure.isna().sum()

mean_value = life1.BMI.mean()
mean_value
life1.BMI = life1.BMI.fillna(mean_value)
life1.BMI.isna().sum()

mean_value = life1.Total_expenditure.mean()
mean_value
life1.Total_expenditure = life1.Total_expenditure.fillna(mean_value)
life1.Total_expenditure.isna().sum()

mean_value = life1.HIV_AIDS.mean()
mean_value
life1.HIV_AIDS = life1.HIV_AIDS.fillna(mean_value)
life1.HIV_AIDS.isna().sum()

mean_value = life1.GDP.mean()
mean_value
life1.GDP = life1.GDP.fillna(mean_value)
life1.GDP.isna().sum()

mean_value = life1.thinness.mean()
mean_value
life1.thinness = life1.thinness.fillna(mean_value)
life1.thinness.isna().sum()

mean_value = life1.thinness_yr.mean()
mean_value
life1.thinness_yr = life1.thinness_yr.fillna(mean_value)
life1.thinness_yr.isna().sum()

mean_value = life1.Income_composition.mean()
mean_value
life1.Income_composition = life1.Income_composition.fillna(mean_value)
life1.Income_composition.isna().sum()

mean_value = life1.Schooling.mean()
mean_value
life1.Schooling = life1.Schooling.fillna(mean_value)
life1.Schooling.isna().sum()


###mode for categorical###
mode_Country = life1.Country.mode()
mode_Country
life1.Country = life1.Country.fillna((mode_Country)[0])
life1.Country.isna().sum()


mode_Status = life1.Status.mode()
mode_Status
life1.Status = life1.Status.fillna((mode_Country)[0])
life1.Status.isna().sum()


# Correlation matrix 
a = life1.corr()
a

# EDA
a1 = life1.describe()

# Sctter plot and histogram between variables
sns.pairplot(life1) # 

# Preparing the model on train data 
model_train = smf.ols("Life_expectancy ~ Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling", data = life1).fit()
model_train.summary()

# Prediction
pred = model_train.predict(life1)
# Error
resid  = pred - life1.Life_expectancy
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(life1.iloc[:, 1:], life1.Life_expectancy)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(life1.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(life1.iloc[:, 1:])

# Adjusted r-square
lasso.score(life1.iloc[:, 1:], life1.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_lasso - life1.life_expectancy)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(life1.iloc[:, 1:], life1.Life_expectancy)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(life1.columns[1:]))

rm.alpha

pred_rm = rm.predict(life1.iloc[:, 1:])

# Adjusted r-square
rm.score(life1.iloc[:, 1:], life1.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_rm - life1.Life_expectancy)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(life1.iloc[:, 1:], life1.Life_expectancy) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(life1.columns[1:]))

enet.alpha

pred_enet = enet.predict(life1.iloc[:, 1:])

# Adjusted r-square
enet.score(life1.iloc[:, 1:], life1.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_enet - life1.life_expectancy)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(life1.iloc[:, 1:], life1.Life_expectancy)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(life1.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(life1.iloc[:, 1:], life1.Life_expectancy)

# RMSE
np.sqrt(np.mean((lasso_pred - life1.Life_expectancy)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(life1.iloc[:, 1:], life1.Life_expectancy)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(life1.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(life1.iloc[:, 1:], life1.Life_expectancy)

# RMSE
np.sqrt(np.mean((ridge_pred - life1.Life_expectancy)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(life1.iloc[:, 1:], life1.Life_expectancy)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(life1.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(life1.iloc[:, 1:], life1.Life_expectancy)

# RMSE
np.sqrt(np.mean((enet_pred - life1.Life_expectancy)**2))
