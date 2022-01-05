# Business Problem : Predicting the burnt area of forest 			
# Business Objective : Minimise the loss of area due to forest fires 		
			
import numpy as np			
import pandas as pd			
import tensorflow			
from keras.models import Sequential			
from keras.layers import Dense, Activation,Layer,Lambda			
from sklearn.model_selection import train_test_split			
			
# Reading data 			
fire = pd.read_csv("C:/data science/ANN/ANN/fireforests.csv")			
			
## Data Cleansing/Data Preparation/Exploratory Data Analysis			
			
fire.head()			
fire.columns			
			
fire = fire.drop('month', 1) #Drop month			
fire = fire.drop('day', 1) #Drop day column			
			
# Normalization			
from sklearn import preprocessing			
scaler = preprocessing.MinMaxScaler()			
fire_norm = scaler.fit_transform(fire)			
			
fire_norm = pd.DataFrame(fire_norm)			
fire_norm.describe()			
fire_norm.columns = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area", "dayfri", "daymon", "daysat", "daysun", "daythu", "daytue", "daywed", "monthapr", "monthaug", "monthdec", "monthfeb", "monthjan", "monthjul", "monthjun", "monthmar",  "monthmay", "monthnov",	"monthoct",	"monthsep"]
			
			
#splitting into train and test			
train,test = train_test_split(fire_norm,test_size = 0.3,random_state=42)			
train_x = train.drop(["area"],axis=1)			
train_y = train["area"]			
test_x = test.drop(["area"],axis=1)			
test_y = test["area"]			
			
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
    			
first_model = prep_model([27,50,40,20,1])			
first_model.fit(np.array(train_x),np.array(train_y),epochs=5)			
			
#checking train correlation			
pred_train = first_model.predict(np.array(train_x))			
pred_train = pd.Series([i[0] for i in pred_train])			
rmse_value = np.sqrt(np.mean((pred_train-train["area"])**2))			
import matplotlib.pyplot as plt			
plt.plot(pred_train,train["area"],"bo")			
np.corrcoef(pred_train,train["area"]) # we got high correlation = 0.897			
			
#Checking test correlation			
pred_test = first_model.predict(np.array(test_x))			
pred_test = pd.Series([i[0] for i in pred_test])			
rmse_value_test = np.sqrt(np.mean((pred_test-test["area"])**2))			
plt.plot(pred_test,test["area"],"bo")			
np.corrcoef(pred_test,test["area"]) 			
