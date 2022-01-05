import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from time import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer()
from sklearn.decomposition import PCA


train=pd.read_csv('C:/datasets/mydata.csv')
test=pd.read_excel('C:/datasets/mydatate.xlsx')

sample=pd.read_excel('C:/data science/project/ETA/dataset/New folder/Sample_Submission.xlsx')

train['Restaurant']=le.fit_transform(train['Restaurant'])
train['Location']=le.fit_transform(train['Location'])
train['Minimum_Order']=pd.to_numeric(train['Minimum_Order'].str.replace('₹',' '))
train['Average_Cost']=pd.to_numeric(train['Average_Cost'].str.replace('[^0-9]',''))
train['Rating']=pd.to_numeric(train['Rating'].apply(lambda x : np.nan if x in ['Temporarily Closed','Opening Soon','-','NEW'] else x))
train['Votes']=pd.to_numeric(train['Votes'].apply(lambda x : np.nan if x=='-' else x))
train['Reviews']=pd.to_numeric(train['Reviews'].apply(lambda x : np.nan if x=='-' else x))
train['Delivery_Time']=pd.to_numeric(train['Delivery_Time'].str.replace('[^0-9]',''))
q1=train['Rating'].quantile(0.25)
q3=train['Rating'].quantile(0.75)
iqr=q3-q1
train['Rating']=train['Rating'].apply(lambda x: np.nan if x>q3+1.5*iqr or x<q1-1.5*iqr else x)
train['Rating']=train['Rating'].fillna(train['Rating'].median())


q1=train['Votes'].quantile(0.25)
q3=train['Votes'].quantile(0.75)
iqr=q3-q1
train['Votes']=train['Votes'].apply(lambda x: np.nan if x>(q3+1.5*iqr) or x<(q1-1.5*iqr) else x)
train['Votes']=train['Votes'].fillna(train['Votes'].mode()[0])


q1=train['Reviews'].quantile(0.25)
q3=train['Reviews'].quantile(0.75)
iqr=q3-q1
train['Reviews']=train['Reviews'].apply(lambda x: np.nan if x>(q3+1.5*iqr) or x<(q1-1.5*iqr) else x)
train['Reviews']=train['Reviews'].fillna(round(train['Reviews'].mean()))



q1=train['Average_Cost'].quantile(0.25)
q3=train['Average_Cost'].quantile(0.75)
iqr=q3-q1
train['Average_Cost']=train['Average_Cost'].apply(lambda x: np.nan if x>(q3+1.5*iqr) or x<(q1-1.5*iqr) else x)
train['Average_Cost']=train['Average_Cost'].fillna(round(train['Average_Cost'].mean()))
train.head()


train_01=train.copy()
train['Cuisines']=le.fit_transform(train['Cuisines'])
x=train.drop('Delivery_Time',axis=1)
y=train['Delivery_Time']
x=x.apply(zscore)
start_time=time()


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
rf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=42)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
accuracy_score(y_test,y_pred)


import pickle 
model_file=open("ETA-ML-final.pkl","wb")##to serialize
pickle.dump(rf.predict,model_file)
model_file.close()##always remember to close it

