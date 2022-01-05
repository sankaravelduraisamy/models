import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/data science/decision tree/DT_RF problem Statement/Company_Data.csv")

data.isnull().sum()
data.dropna()
data.columns
#data = data.drop(["phone"], axis = 1)

#converting into binary
lb = LabelEncoder()
data["Sales"] = lb.fit_transform(data["Sales"])
data["CompPrice"] = lb.fit_transform(data["CompPrice"])
data["Income"] = lb.fit_transform(data["Income"])
data["Advertising"] = lb.fit_transform(data["Advertising"])
data["Population"] = lb.fit_transform(data["Population"])
data["Price"] = lb.fit_transform(data["Price"])
data["ShelveLoc"] = lb.fit_transform(data["ShelveLoc"])
data["Age"] = lb.fit_transform(data["Age"])
data["Education"] = lb.fit_transform(data["Education"])
data["Urban"] = lb.fit_transform(data["Urban"])
data["US"] = lb.fit_transform(data["US"])

#data["default"]=lb.fit_transform(data["default"])

data['default'].unique()
data['default'].value_counts()
colnames = list(data.columns)

predictors = colnames[:10]
target = colnames[:1]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy
