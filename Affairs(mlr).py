import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
affairs = pd.read_csv("C:/data science/logistic regression/Datasets_LR/Affairs.csv", sep = ",")

#removing CASENUM

c1 = affairs
c1.drop('Sno', axis = 1,inplace=True)
c1.head(11)
c1.describe()
c1.isna().sum()

# To drop NaN values
df = c1.dropna()

c1['naffairs'] = np.where(c1['naffairs'] >0, 1, 0)
# Imputating the missing values           
# Mean Imputation - CLMAGE is a continuous data
c1.columns
c1.info()




# Alternate approach
########## Median Imputation for all the columns ############
c1.fillna(c1.median(), inplace=True)
c1.isna().sum()


#############################################################

for i in c1.columns:
    if (i!='naffairs'):        
        mean_ = c1[i].mean()
        print(mean_)
        c1[i] = c1[i].fillna(mean_)
        print(c1[i])       

   
    
# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + vryhap + antirel + notrel+ slghtrel+ smerel+ vryrel +yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr5+yrsmarr6', data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.naffairs, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
c1["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["naffairs"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + vryhap + antirel + notrel+ slghtrel+ smerel+ vryrel +yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr5+yrsmarr6', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix
a=confusion_matrix[0][0]
b=confusion_matrix[0][1]
c=confusion_matrix[1][0]
d=confusion_matrix[1][1]
accuracy_test = (a + d)/(a+b+c+d) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["naffairs"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

e=confusion_matrx[0][0]
f=confusion_matrx[0][1]
g=confusion_matrx[1][0]
h=confusion_matrx[1][1]

accuracy_train = (e+ h)/(e+f+g+h)
print(accuracy_train)
