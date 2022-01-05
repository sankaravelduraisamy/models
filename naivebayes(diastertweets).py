###################################### Solution of Q2 ######################################
import numpy as np
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
import pandas as pd

#import data file
NB_Car_Ad = pd.read_csv('C:/data science/naive bayes/Datasets_Naive Bayes (1)/Disaster_tweets_NB.csv')

#check data sets
NB_Car_Ad.head()

#X is the matrix of features, it contains independent variable number 2 and 3 which is Age,EstimatedSalary according to NB_Car_Ad
input_data = NB_Car_Ad.iloc[:,[2,3]].values
#Y contains dependent variable which is Purchased according to NB_Car_Ad and the column number is 4
output_data= NB_Car_Ad.iloc[:,4].values

#print independent variables
input_data
#print dependant variables
output_data

# Splitting the NB_Car_Ad into the Training set and Test set
X_train, X_test, y_train, y_test = tts(input_data,output_data,test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
classifier = GaussianNB()
classifier.fit(X_train , y_train)

prediction = classifier.predict(X_test)
	
accuracy_score(y_test, prediction, normalize = True)
# on training data
target_pred0 = classifier.predict(X_train)
	
accuracy_score(y_train, target_pred0, normalize = True)
