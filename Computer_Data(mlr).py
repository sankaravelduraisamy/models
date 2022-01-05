# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
Computer_Data = pd.read_csv("C:/data science/multi linear regression/Datasets_MLR/Computer_Data.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

Computer_Data.drop("cd", inplace=True, axis=1)
Computer_Data.drop("multi", inplace=True, axis=1)
Computer_Data.drop("premium", inplace=True, axis=1)



Computer_Data.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Speed
plt.bar(height = Computer_Data.speed, x = np.arange(1, 6260, 1))
plt.hist(Computer_Data.speed) #histogram
plt.boxplot(Computer_Data.speed) #boxplot

# PRICE
plt.bar(height = Computer_Data.price, x = np.arange(1, 6260, 1))
plt.hist(Computer_Data.price) #histogram
plt.boxplot(Computer_Data.price) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=Computer_Data['speed'], y=Computer_Data['price'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(Computer_Data['speed'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(Computer_Data.price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Computer_Data.iloc[:, :])
                             
# Correlation matrix 
Computer_Data.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas# for regression model
         
ml1 = smf.ols('price ~ speed + hd + ram + screen + ads + trend', data = Computer_Data).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index  1043 is showing high influence so we can exclude that entire row

Computer_Data_new = Computer_Data.drop(Computer_Data.index[[1043]])

# Preparing model                  
ml_new = smf.ols('price ~ speed + hd + ram + screen + ads + trend', data = Computer_Data_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_sp = smf.ols('speed ~ hd + ram + screen + ads + trend', data = Computer_Data).fit().rsquared  
vif_sp = 1/(1 - rsq_sp) 

rsq_hd = smf.ols('hd ~ speed + ram + screen + ads + trend', data = Computer_Data).fit().rsquared  
vif_hd = 1/(1 - rsq_hd)

rsq_rm = smf.ols('ram ~ speed + hd + screen + ads + trend', data = Computer_Data).fit().rsquared  
vif_rm = 1/(1 - rsq_rm) 

rsq_sc = smf.ols('screen ~ speed + hd + ram + ads + trend', data = Computer_Data).fit().rsquared  
vif_sc = 1/(1 - rsq_sc) 

rsq_ad = smf.ols('ads ~ speed + hd + ram + screen + trend', data = Computer_Data).fit().rsquared  
vif_ad = 1/(1 - rsq_rm) 

rsq_tr = smf.ols('trend ~ speed + hd + ram + screen + ads', data = Computer_Data).fit().rsquared  
vif_tr = 1/(1 - rsq_tr) 

# Storing vif values in a data frame
d1 = {'Variables':['speed', 'hd', 'ram', 'screen', 'ads', 'trend'], 'VIF':[vif_sp, vif_hd, vif_rm, vif_sc, vif_ad, vif_tr]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As hd is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('price ~ speed + ram + screen + ads + trend', data = Computer_Data).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(Computer_Data)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = Computer_Data.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
Computer_Data_train, Computer_Data_test = train_test_split(Computer_Data, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("price ~ speed + hd + ram + screen + ads + trend", data = Computer_Data_train).fit()

# prediction on test data set 
test_pred = model_train.predict(Computer_Data_test)

# test residual values 
test_resid = test_pred - Computer_Data_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(Computer_Data_train)

# train residual values 
train_resid  = train_pred - Computer_Data_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
