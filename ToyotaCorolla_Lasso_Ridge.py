# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
ToyotaCorolla = pd.read_csv("C:/data science/lasso ridge/Datasets_LassoRidge/ToyotaCorolla (1).csv")

# Rearrange the order of the variables
ToyotaCorolla = ToyotaCorolla.iloc[:, [2,3,6,8,12,13,15,16,17]]
ToyotaCorolla.columns


# Correlation matrix 
a = ToyotaCorolla.corr()
a

# EDA
a1 = ToyotaCorolla.describe()

# Sctter plot and histogram between variables
sns.pairplot(ToyotaCorolla) # 

# Preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight", data = ToyotaCorolla).fit()
model_train.summary()

# Prediction
pred = model_train.predict(ToyotaCorolla)
# Error
resid  = pred - ToyotaCorolla.Price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(ToyotaCorolla.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(ToyotaCorolla.iloc[:, 1:])

# Adjusted r-square
lasso.score(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.price)

# RMSE
np.sqrt(np.mean((pred_lasso - ToyotaCorolla.Price)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(ToyotaCorolla.columns[1:]))

rm.alpha

pred_rm = rm.predict(ToyotaCorolla.iloc[:, 1:])

# Adjusted r-square
rm.score(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)

# RMSE
np.sqrt(np.mean((pred_rm - ToyotaCorolla.Price)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(ToyotaCorolla.columns[1:]))

enet.alpha

pred_enet = enet.predict(ToyotaCorolla.iloc[:, 1:])

# Adjusted r-square
enet.score(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)

# RMSE
np.sqrt(np.mean((pred_enet - ToyotaCorolla.Price)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(ToyotaCorolla.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)

# RMSE
np.sqrt(np.mean((lasso_pred - ToyotaCorolla.Price)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(ToyotaCorolla.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)

# RMSE
np.sqrt(np.mean((ridge_pred - ToyotaCorolla.Price)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(ToyotaCorolla.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(ToyotaCorolla.iloc[:, 1:], ToyotaCorolla.Price)

# RMSE
np.sqrt(np.mean((enet_pred - ToyotaCorolla.Price)**2))
