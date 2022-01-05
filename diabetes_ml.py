import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek ##For upsampling
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
import pickle #for serialization
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")



dataset=pd.read_csv("C:/datasets/diabetes.csv")##read our dataset
##Lets separate features into dependant and independent feature
X=dataset.drop("Outcome",axis=1)
y=dataset["Outcome"]
print(X.shape,y.shape)


y.value_counts()##returns count of unique class in that feature


smote=SMOTETomek(random_state=42,n_jobs=-1)##Library used to do upsampling
X_,Y_=smote.fit_resample(X,y)

print(X_.shape,Y_.shape)##Dimensions of data increased

Y_.value_counts()


x_train,x_test,y_train,y_test=train_test_split(X_,Y_,test_size=0.25,random_state=42)
print(x_train.shape,y_train.shape)


col=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for feature in col:
    x_train[feature].replace(0,x_train[feature].median(),inplace=True)
    
    
    
rf=RandomForestClassifier(n_jobs=-1)
rf.fit(x_train,y_train)

score=cross_val_score(rf,x_train,y_train,cv=10,n_jobs=-1)

score

score.shape

score.mean()

score_test=cross_val_score(rf,x_test,y_test,cv=10,n_jobs=-1)
score_test

score_test.mean()

print("The maximum accuracy that our model can get is {} and minimum accuracy the model can get is {}".format(np.round(score_test.max(),2),np.round(score_test.min(),2)))


pred=rf.predict(x_test)
confusion_matrix(y_test,pred)

print(classification_report(y_test,pred))

##hyperparmeters of random forest
RandomForestClassifier()



#hyperparametr tuning

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)


rs=RandomizedSearchCV(rf,random_grid,cv=10,verbose=2,n_jobs=-1)
rs.fit(x_train,y_train)


rs.best_params_

pred=rs.predict(x_test)
confusion_matrix(y_test,pred)


print(classification_report(pred,y_test))

metrics.plot_roc_curve(rs,x_test,y_test)

model_file=open("model.pkl","wb")##to serialize
pickle.dump(rs,model_file)
model_file.close()##always remember to close it


model=pickle.load(open("model.pkl","rb"))
pred=model.predict(X[:50])
confusion_matrix(y[:50],pred)

