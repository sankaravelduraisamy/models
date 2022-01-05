# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
#ToyotaCorolla = pd.read_csv("C:/data science/multi linear regression/Datasets_MLR/ToyotaCorolla.csv")
ToyotaCorolla = pd.read_csv("C:/data science/1 finished assignments/multi linear regression/Datasets_MLR/ToyotaCorolla.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (HistogHP, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)


ToyotaCorolla[['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight']]




ToyotaCorolla.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Age_08_04
plt.bar(height = ToyotaCorolla.Age_08_04, x = np.arange(1, 1437, 1))
plt.hist(ToyotaCorolla.Age_08_04) #histogHP
plt.boxplot(ToyotaCorolla.Age_08_04) #boxplot

# PRICE
plt.bar(height = ToyotaCorolla.Price, x = np.arange(1, 1437, 1))
plt.hist(ToyotaCorolla.Price) #histogHP
plt.boxplot(ToyotaCorolla.Price) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=ToyotaCorolla['Age_08_04'], y=ToyotaCorolla['Price'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(ToyotaCorolla['Age_08_04'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(ToyotaCorolla.Price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histogHPs
import seaborn as sns
sns.pairplot(ToyotaCorolla.iloc[:, :])
                             
# Correlation matrix 
ToyotaCorolla.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = ToyotaCorolla).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index  1043 is showing high influence so we can exclude that entire row

ToyotaCorolla_new = ToyotaCorolla.drop(ToyotaCorolla.index[[798]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = ToyotaCorolla_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_ag = smf.ols('Age_08_04 ~ KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = ToyotaCorolla).fit().rsquared  
vif_ag = 1/(1 - rsq_ag) 

rsq_km = smf.ols('KM ~ Age_08_04 + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = ToyotaCorolla).fit().rsquared  
vif_km = 1/(1 - rsq_km)

rsq_hp = smf.ols('HP ~ Age_08_04 + KM + cc + Doors + Gears + Quarterly_Tax + Weight', data = ToyotaCorolla).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_cc = smf.ols('cc ~ Age_08_04 + KM + HP + Doors + Gears + Quarterly_Tax + Weight', data = ToyotaCorolla).fit().rsquared  
vif_cc = 1/(1 - rsq_cc) 

rsq_dr = smf.ols('Doors ~ Age_08_04 + KM + HP + cc + Gears + Quarterly_Tax + Weight', data = ToyotaCorolla).fit().rsquared  
vif_dr = 1/(1 - rsq_dr) 

rsq_gr = smf.ols('Gears ~ Age_08_04 + KM + HP + cc + Doors + Quarterly_Tax + Weight', data = ToyotaCorolla).fit().rsquared  
vif_gr = 1/(1 - rsq_gr) 

rsq_qt = smf.ols('Quarterly_Tax ~ Age_08_04 + KM + HP + cc + Gears + Doors + + Weight', data = ToyotaCorolla).fit().rsquared  
vif_qt = 1/(1 - rsq_qt) 

rsq_wt = smf.ols('Weight ~ Age_08_04 + KM + HP + cc + Gears + Doors + Quarterly_Tax + Weight', data = ToyotaCorolla).fit().rsquared  
vif_wt = 1/(1 - rsq_wt) 

# Storing vif values in a data fHPe
d1 = {'Variables':['Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight'], 'VIF':[vif_ag, vif_km, vif_hp, vif_cc, vif_dr, vif_gr, vif_qt, vif_wt]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As Quarterly_Tax is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Price ~ Age_08_04 + HP + cc + Doors + Gears + Weight', data = ToyotaCorolla).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(ToyotaCorolla)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = ToyotaCorolla.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
ToyotaCorolla_train, ToyotaCorolla_test = train_test_split(ToyotaCorolla, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight", data = ToyotaCorolla_train).fit()

# prediction on test data set 
test_pred = model_train.predict(ToyotaCorolla_test)

# test residual values 
test_resid = test_pred - ToyotaCorolla_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(ToyotaCorolla_train)

# train residual values 
train_resid  = train_pred - ToyotaCorolla_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
