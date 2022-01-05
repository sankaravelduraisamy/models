# -*- coding: utf-8 -*-
"""
Created on Sat May  8 10:08:16 2021

@author: Dathu
"""


# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
startup = pd.read_csv("D:/360DigiTMG/Assignment/Lasso & Ridge Regression/Datasets_LassoRidge/50_Startups (1).csv")

# Rearrange the order of the variables
startup = startup.iloc[:, [4, 0, 1, 2]]
startup.columns


startup = startup.rename(columns = {'R&D Spend': 'R_D_Spend', 'Marketing Spend': 'Marketing_Spend'}, inplace = False)


# Correlation matrix 
a = startup.corr()
a

# EDA
a1 = startup.describe()

# Sctter plot and histogram between variables
sns.pairplot(startup) # 

# Preparing the model on train data 
model_train = smf.ols("Profit ~ R_D_Spend + Administration + Marketing_Spend", data = startup).fit()
model_train.summary()

# Prediction
pred = model_train.predict(startup)
# Error
resid  = pred - startup.Profit
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(startup.iloc[:, 1:], startup.Profit)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(startup.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(startup.iloc[:, 1:])

# Adjusted r-square
lasso.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((pred_lasso - startup.Profit)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(startup.iloc[:, 1:], startup.Profit)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(startup.columns[1:]))

rm.alpha

pred_rm = rm.predict(startup.iloc[:, 1:])

# Adjusted r-square
rm.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((pred_rm - startup.Profit)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(startup.iloc[:, 1:], startup.Profit) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(startup.columns[1:]))

enet.alpha

pred_enet = enet.predict(startup.iloc[:, 1:])

# Adjusted r-square
enet.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((pred_enet - startup.Profit)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(startup.iloc[:, 1:], startup.Profit)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(startup.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((lasso_pred - startup.Profit)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(startup.iloc[:, 1:], startup.Profit)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(startup.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((ridge_pred - startup.Profit)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(startup.iloc[:, 1:], startup.Profit)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(startup.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((enet_pred - startup.Profit)**2))
