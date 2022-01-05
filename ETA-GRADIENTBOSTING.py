import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


data = pd.read_excel('C:/data science/project/ETA/dataset/Data_Train.xlsx', index_col=0)

data_test = pd.read_excel('C:/data science/project/ETA/dataset/Data_Test.xlsx', index_col=0)

data=data.values[:,2:8]

print(data_test.values[:,2:8])

data_test=data_test.values[:,2:8]

#Data Cleaning and preprocessing

for i in range(data.shape[0]):
    data[i,0]=data[i,0].replace('₹', '')
    data[i,1]=data[i,1].replace('₹', '')
    data[i,2]=data[i,2].replace('-', '0')
    data[i,3]=data[i,3].replace('-', '0')
    data[i,4]=data[i,4].replace('-', '0')
    data[i,2]=data[i,2].replace('NEW', '0')
    data[i,5]=data[i,5].replace(' minutes', '')

    data[i,0]=data[i,0].replace(',', '')
    data[i,1]=data[i,1].replace(',', '')
    data[i,2]=data[i,2].replace(',', '')
    data[i,2]=data[i,2].replace('Opening Soon', '0')
    data[i,0]=data[i,0].replace('for', '0')
    data[i,2]=data[i,2].replace('Temporarily Closed', '0')

datat=data_test

for i in range(datat.shape[0]):
    datat[i,0]=datat[i,0].replace('₹', '')
    datat[i,1]=datat[i,1].replace('₹', '')
    datat[i,2]=datat[i,2].replace('-', '0')
    datat[i,3]=datat[i,3].replace('-', '0')
    datat[i,4]=datat[i,4].replace('-', '0')
    datat[i,2]=datat[i,2].replace('NEW', '0')
    #datat[i,5]=datat[i,5].replace(' minutes', '')

    datat[i,0]=datat[i,0].replace(',', '')
    datat[i,1]=datat[i,1].replace(',', '')
    datat[i,2]=datat[i,2].replace(',', '')
    datat[i,2]=datat[i,2].replace('Opening Soon', '0')
    datat[i,0]=datat[i,0].replace('for', '0')
    datat[i,2]=datat[i,2].replace('Temporarily Closed', '0')    
    


for i in range(data.shape[0]):
    
    for j in range(data.shape[1]):
        
        data[i,j] = float(data[i,j])
        
for i in range(datat.shape[0]):
    
    for j in range(datat.shape[1]):
        
        datat[i,j] = float(datat[i,j])
        

X = data[:,0:5]
Y = data[:,5]
Xt = datat[:,0:5]
print(X,"\n",Y,"\n",Xt)




#Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

Xt = sc.transform(Xt)
X_tt= Xt.reshape(-1, 1)

Y = Y.reshape((len(Y), 1)) 

Y = sc.fit_transform(Y)

Y = Y.ravel()
#Model Building
from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor( loss = 'huber',learning_rate=0.001,n_estimators=350, max_depth=6 , subsample=1,  verbose=False,random_state=126)  
w = gbr.fit(X, Y)

W = w.reshape(-1, 1)

#Prediction for test data 
y_pred_gbr = sc.inverse_transform(gbr.predict(X_tt))
y1=y_pred_gbr.astype(int) 
y1=y1.astype(int).astype(str) 
for i in range(len(y1)):
    y1[i] = y1[i]+" minutes"
    
    
    
#Sending output to Excel file
df = pd.DataFrame (y1)
print(df.head)
df.to_excel("GradientBoostingRegressor.xlsx", index = False)