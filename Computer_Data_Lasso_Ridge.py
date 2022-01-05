# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:04:18 2021

@author: Dathu
"""


# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
Computer_Data = pd.read_csv("D:/360DigiTMG/Assignment/Lasso & Ridge Regression/Datasets_LassoRidge/Computer_Data (1).csv")

# Rearrange the order of the variables
Computer_Data = Computer_Data.iloc[:, [1,2,3,4,5,9,10]]
Computer_Data.columns


# Correlation matrix 
a = Computer_Data.corr()
a

# EDA
a1 = Computer_Data.describe()

# Sctter plot and histogram between variables
sns.pairplot(Computer_Data) # 

# Preparing the model on train data 
model_train = smf.ols("price ~ speed + hd + ram + screen + ads + trend", data = Computer_Data).fit()
model_train.summary()

# Prediction
pred = model_train.predict(Computer_Data)
# Error
resid  = pred - Computer_Data.price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(Computer_Data.iloc[:, 1:], Computer_Data.price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(Computer_Data.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square
lasso.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((pred_lasso - Computer_Data.price)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(Computer_Data.iloc[:, 1:], Computer_Data.price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(Computer_Data.columns[1:]))

rm.alpha

pred_rm = rm.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square
rm.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((pred_rm - Computer_Data.price)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(Computer_Data.iloc[:, 1:], Computer_Data.price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(Computer_Data.columns[1:]))

enet.alpha

pred_enet = enet.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square
enet.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((pred_enet - Computer_Data.price)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(Computer_Data.iloc[:, 1:], Computer_Data.price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((lasso_pred - Computer_Data.price)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(Computer_Data.iloc[:, 1:], Computer_Data.price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((ridge_pred - Computer_Data.price)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(Computer_Data.iloc[:, 1:], Computer_Data.price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((enet_pred - Computer_Data.price)**2))
