# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
Avocado_Price = pd.read_csv("C:/data science/1 finished assignments/multi linear regression/Datasets_MLR/Avacado_Price.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (HistogTOT_AVA2, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)


Avocado_Price.drop('type', inplace=True, axis=1)
Avocado_Price.drop('year', inplace=True, axis=1)
Avocado_Price.drop('region', inplace=True, axis=1)

Avocado_Price.rename(columns = {'XLarge Bags': 'XLarge_Bags'}, inplace = True)

Avocado_Price.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Total_Volume
plt.bar(height = Avocado_Price.Total_Volume, x = np.arange(1, 18250, 1))
plt.hist(Avocado_Price.Total_Volume) #histog
plt.boxplot(Avocado_Price.Total_Volume) #boxplot

# AVERAGEPRICE
plt.bar(height = Avocado_Price.AveragePrice, x = np.arange(1, 18250, 1))
plt.hist(Avocado_Price.AveragePrice) #histogTOT_AVA2
plt.boxplot(Avocado_Price.AveragePrice) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=Avocado_Price['Total_Volume'], y=Avocado_Price['AveragePrice'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(Avocado_Price['Total_Volume'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(Avocado_Price.AveragePrice, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histogTOT_AVA2s
import seaborn as sns
sns.pairplot(Avocado_Price.iloc[:, :])
                             
# Correlation matrix 
Avocado_Price.corr()

# we see there exists High collinearity between input variables especially between
# [TOT_AVA2 & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index  1043 is showing high influence so we can exclude that entire row

Avocado_Price_new = Avocado_Price.drop(Avocado_Price.index[[11271]])

# Preparing model                  
ml_new = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_tt = smf.ols('Total_Volume ~ tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tt= 1/(1 - rsq_tt) 

rsq_tot_ava1 = smf.ols('tot_ava1 ~ Total_Volume + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tot_ava1 = 1/(1 - rsq_tot_ava1)

rsq_tot_ava2 = smf.ols('tot_ava2 ~ Total_Volume + tot_ava1 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tot_ava2 = 1/(1 - rsq_tot_ava2) 

rsq_tot_ava3 = smf.ols('tot_ava3 ~ Total_Volume + tot_ava1 + tot_ava2 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tot_ava3 = 1/(1 - rsq_tot_ava3) 

rsq_tb = smf.ols('Total_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_tb = 1/(1 - rsq_tb) 

rsq_sb = smf.ols('Small_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_sb = 1/(1 - rsq_sb) 

rsq_lb = smf.ols('Large_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags + Total_Bags + + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_lb = 1/(1 - rsq_lb) 

rsq_xl = smf.ols('XLarge_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags + Total_Bags + Large_Bags + XLarge_Bags', data = Avocado_Price).fit().rsquared  
vif_xl = 1/(1 - rsq_xl) 

# Storing vif values in a data fTOT_AVA2e
d1 = {'Variables':['Total_Volume', 'tot_ava1 + tot_ava2 ', 'tot_ava3', 'Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags','Bags'], 'VIF':[vif_tt, vif_tot_ava1, vif_tot_ava2, vif_tot_ava3, vif_tb, vif_sb, vif_lb, vif_xl]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As Large_Bags is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('AveragePrice ~ Total_Volume + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + XLarge_Bags', data = Avocado_Price).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(Avocado_Price)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = Avocado_Price.AveragePrice, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
Avocado_Price_train, Avocado_Price_test = train_test_split(Avocado_Price, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags", data = Avocado_Price_train).fit()

# prediction on test data set 
test_pred = model_train.predict(Avocado_Price_test)

# test residual values 
test_resid = test_pred - Avocado_Price_test.AveragePrice
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(Avocado_Price_train)

# train residual values 
train_resid  = train_pred - Avocado_Price_train.AveragePrice
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
