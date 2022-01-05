import numpy as np
import numpy as np 
import pandas as pd
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda
from sklearn.model_selection import train_test_split

# Reading data 
startup = pd.read_csv("C:/data science/ANN/ANN/50_Startups.csv")
startup.head()
startup.columns

## Data Cleansing/Data Preparation/Exploratory Data Analysis
#converting into binary
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
startup["State"]=lb.fit_transform(startup["State"]) # label encoding is done

startup.describe()

startup.isna().sum()  # To check for NA values  

startup.corr() # To know the correlation between the inputs we do a "correlation matrix" 


#splitting into train and test
train,test = train_test_split(startup,test_size = 0.3,random_state=42)
train_x = train.drop(["Profit"],axis=1)
train_y = train["Profit"]
test_x = test.drop(["Profit"],axis=1)
test_y = test["Profit"]

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
    
first_model = prep_model([4,50,1])
first_model.fit(np.array(train_x),np.array(train_y),epochs=10)

#checking train correlation
pred_train = first_model.predict(np.array(train_x))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-train["Profit"])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,train["Profit"],"bo")
np.corrcoef(pred_train,train["Profit"]) # we got high correlation =0.95

#Checking test correlation
pred_test = first_model.predict(np.array(test_x))
pred_test = pd.Series([i[0] for i in pred_test])
rmse_value_test = np.sqrt(np.mean((pred_test-test["Profit"])**2))
plt.plot(pred_test,test["Profit"],"bo")
np.corrcoef(pred_test,test["Profit"]) # we got high correlation =0.97

