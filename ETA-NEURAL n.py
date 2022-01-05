









import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('C:/data science/project/ETA/onlinedeliverydata.csv', index_col=0)
print(df.head)
df.columns

data=df.values[:,6:7]

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
    
print(data[0:10,:])

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i,j] = float(data[i,j])
        

X=data[:,0:5]
Y=data[:,5]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)


Y = Y.reshape((len(Y), 1)) 

Y = sc.fit_transform(Y)

Y = Y.ravel()



def larger_model():
	# create model
    model = Sequential()
    model.add(Dense(20, input_dim=5, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# evaluate model with standardized dataset


model = larger_model()
model.fit(X,Y,epochs=5,batch_size=20)

#model.fit(X,Y,nb_epoch=20, batch_size=20)


data_test = pd.read_excel('C:/data science/project/ETA/dataset/Data_Test.xlsx', index_col=0)


data_test=data_test.values[:,0:8]
data_test.columns

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

for i in range(datat.shape[0]):
    
    for j in range(datat.shape[1]):
        
        datat[i,j] = float(datat[i,j])

Xt = datat[:,1]
from sklearn import *
Xt = sc.transform(Xt)



yp = model.predict(Xt)
yp=abs(sc.inverse_transform(yp)) 
print(yp)


yp=yp.astype(int).astype(str) 
for i in range(yp.shape[0]):
    yp[i,0] = yp[i,0]+" minutes"
df = pd.DataFrame (yp)
df.to_excel("NNModel.xlsx", index = False)