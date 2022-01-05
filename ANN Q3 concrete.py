import numpy as np
import pandas as pd
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda
from sklearn.model_selection import train_test_split

# Reading data 
concrete = pd.read_csv("C:/data science/ANN/ANN/concrete.csv")

## Data Cleansing/Data Preparation/Exploratory Data Analysis
concrete.head()
concrete.columns

concrete.describe()

concrete.isna().sum()  # To check for NA values  

# We have to normalize the data to make it scale free

# Normalization
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
concrete_norm = scaler.fit_transform(concrete)

concrete_norm = pd.DataFrame(concrete_norm)
concrete_norm.describe()
concrete_norm.columns = ["cement", "slag", "ash", "water", "superplastic", "coarseagg", "fineagg",	"age",	"strength"]

concrete_norm.corr() # To know the correlation between the inputs we do a "correlation matrix" 


#splitting into train and test
train,test = train_test_split(concrete_norm,test_size = 0.3,random_state=42)
trainX = train.drop(["strength"],axis=1)
trainY = train["strength"]
testX = test.drop(["strength"],axis=1)
testY = test["strength"]

# Defining the ANN/MLP model
def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)


#Building the model
    
first_model = prep_model([8,50,40,20,1])
first_model.fit(np.array(trainX),np.array(trainY),epochs=20)

#checking train correlation
pred_train = first_model.predict(np.array(trainX))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-train["strength"])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,train["strength"],"bo")
np.corrcoef(pred_train,train["strength"]) # we got high correlation =0.9811

#Checking test correlation
pred_test = first_model.predict(np.array(testX))
pred_test = pd.Series([i[0] for i in pred_test])
rmse_value_test = np.sqrt(np.mean((pred_test-test["strength"])**2))
plt.plot(pred_test,test["strength"],"bo")
np.corrcoef(pred_test,test["strength"]) 