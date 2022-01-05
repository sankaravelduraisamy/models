# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
startup = pd.read_csv("C:/data science/multi linear regression/Datasets_MLR/50_Startups.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

startup.drop("State", inplace=True, axis=1)
startup = startup.rename(columns = {'R&D Spend': 'R_D_Spend', 'Marketing Spend': 'Marketing_Spend'}, inplace = False)



startup.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# R_D_Spend
plt.hist(startup.R_D_Spend) #histogram
plt.boxplot(startup.R_D_Spend) #boxplot

# PROFIT
plt.hist(startup.Profit) #histogram
plt.boxplot(startup.Profit) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=startup['R_D_Spend'], y=startup['Profit'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(startup['R_D_Spend'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(startup.Profit, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup.iloc[:, :])
                             
# Correlation matrix 
startup.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit ~ R_D_Spend + Administration + Marketing_Spend', data = startup).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

startup_new = startup.drop(startup.index[[49]])

# Preparing model                  
ml_new = smf.ols('Profit ~ R_D_Spend + Administration + Marketing_Spend', data = startup_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_rd = smf.ols('R_D_Spend ~ Administration + Marketing_Spend', data = startup).fit().rsquared  
vif_rd = 1/(1 - rsq_rd) 

rsq_ad = smf.ols('Administration ~ R_D_Spend + Marketing_Spend', data = startup).fit().rsquared  
vif_ad = 1/(1 - rsq_ad)

rsq_mr = smf.ols('Marketing_Spend ~ R_D_Spend + Administration', data = startup).fit().rsquared  
vif_mr = 1/(1 - rsq_mr) 

# Storing vif values in a data frame
d1 = {'Variables':['R_D_Spend', 'Administration', 'Marketing_Spend'], 'VIF':[vif_rd, vif_ad, vif_mr]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As R_D_Spend is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Profit ~ Administration + Marketing_Spend', data = startup).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(startup)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startup.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train, startup_test = train_test_split(startup, test_size = 0.3) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Profit ~ R_D_Spend + Administration + Marketing_Spend", data = startup_train).fit()

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_resid = test_pred - startup_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values 
train_resid  = train_pred - startup_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
