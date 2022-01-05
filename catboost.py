import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.set_option('display.max_columns', 500)
#Import dataset and having birds eye view
df=pd.read_csv("C:/datasets/mydata.csv")
df.shape
df.head(10)
df.info()

#Check for missing value:
df.isnull().sum()
df.columns


#Data Pre-processing and Feature Engineering of train dataset:
# Creating unique feature for every unique observation in "Cuisines" variable (dummy variable):
df['Cuisines'] = df['Cuisines'].str.replace('Poké','Poke')
df['Cuisines'] = df['Cuisines'].str.replace('Coffee','Tea')
df['Cuisines'] = df['Cuisines'].str.replace('Hyderabadi','Biryani')

cuisines_list = df['Cuisines'].str.split(', ')

from collections import Counter
cuisines_counter = Counter(([a for b in cuisines_list.tolist() for a in b]))

for cuisine in cuisines_counter.keys():
    df[cuisine] = 0
    df.loc[df['Cuisines'].str.contains(cuisine), cuisine] = 1
#from sklearn.preprocessing import MultiLabelBinarizer
#mlb = MultiLabelBinarizer()
#df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('Cuisines')), columns=mlb.classes_,index=df.index))
#df['Restaurant'] = df['Restaurant'].str.replace('ID_', '')
df['Restaurant']=df['Restaurant'].astype('category')
df.head()
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#df['Location'] = df.fit_transform(df['Location'])
#For "Location" Varaible Removing whitespaces and turn it into lowercase:

df['Location']=df['Location'].str.strip().str.lower()
df['Location'].str.split(',',expand=True)

# Make a new feature "Locality" from "Location":
df['Locality']=df['Location'].str.split(',').str[0]
df['Location'].value_counts()

#Adding new feature "City" from "Location"variable:

df['City']=[i.replace('mico layout, stage 2, btm layout,bangalore',"Bangalore").replace('d-block, sector 63, noida',"Noida")
            .replace("sector 1, noida","Noida").replace("fti college, law college road, pune","Pune")
            .replace('delhi university-gtb nagar',"Delhi").replace('laxman vihar industrial area, sector 3a, gurgoan',"Gurgaon")
            .replace('sector 14, noida',"Noida").replace("delhi administration flats, timarpur","Delhi")
            .replace("mumbai central","Mumbai").replace("rmz centennial, i gate, whitefield","Bangalore")
            .replace("delhi high court, india gate","Delhi").replace("mg road, pune","Pune")
            .replace("nathan road, mangaldas road, pune","Pune").replace("sandhurst road, mumbai cst area","Mumbai")
            .replace("sector 3, marathalli","Bangalore").replace("majestic","Not Given").replace("delhi cantt.","Delhi")
            .replace("yerawada, pune, maharashtra","Pune").replace("dockyard road, mumbai cst area","Mumbai")
            .replace("babarpur, new delhi, delhi","Delhi").replace("pune university","Pune").replace("sector 63a,gurgaon","Gurgaon")
            .replace("moulali, kolkata",'Kolkata').replace("chandni chowk, kolkata","Kolkata").replace("tejas nagar colony, wadala west, mumbai","Mumbai")
            .replace("raja bazar, kolkata","Kolkata").replace("tiretti, kolkata","Kolkata").replace("hyderabad public school, begumpet","Hyderabad")
            .replace("gora bazar, rajbari, north dumdum, kolkata","Kolkata").replace("noorkhan bazaar, malakpet, hyderabad","Hyderabad")
            .replace("musi nagar, malakpet, hyderabad","Hyderabad").replace("panjetan colony, malakpet, hyderabad","Hyderabad")
            .replace("chatta bazaar, malakpet, hyderabad","Hyderabad")
            .replace("jaya nagar, saidabad, hyderabad","Hyderabad")
            .replace("btm layout 1, electronic city","Bangalore") for i in df['Location']]
#Percentage of order per city:

df["City"].value_counts()/len(df['City'])*100

#Import regex library and replace unwanted values ,symbols and remove whitespaces and convert it into numeric feature:
import re
df['Average_Cost'] = df['Average_Cost'].str.replace("[^0-9]","")
df['Average_Cost'] = df['Average_Cost'].str.strip()
df['Average_Cost']=pd.to_numeric(df['Average_Cost'])
#df.Average_Cost.fillna(0, inplace=True)
#Average cost in different city and locality:
df.groupby(['City','Locality'])['Average_Cost'].mean()
#Replacing missing value:
df['Average_Cost'].fillna(df.groupby(['City','Locality'])['Average_Cost'].transform('mean'), inplace=True)
df['Minimum_Order']=df['Minimum_Order'].str.replace("[^0-9]","")
df['Minimum_Order']=df['Minimum_Order'].str.strip()
df['Minimum_Order']=pd.to_numeric(df['Minimum_Order'])
#df.Minimum_Order.fillna(0, inplace=True)


#Minimum_order in different city and locality:
df.groupby(['City','Locality'])['Minimum_Order'].mean()
df['Minimum_Order'].fillna(df.groupby(['City','Locality'])['Minimum_Order'].transform('mean'), inplace=True)
df['Rating']=df['Rating'].replace('Temporarily Closed',np.nan).replace('Opening Soon',np.nan).replace("NEW",np.nan)
df['Rating'] = df['Rating'].replace("-",np.nan)
df['Rating']=df['Rating'].astype('float')
df['Rating']=pd.to_numeric(df['Rating'])
df["Rating"].fillna(0, inplace=True)
df['Rating_Category']=pd.qcut(df['Rating'],q=5,precision=0,labels=False)
df['Rating_Category'].value_counts()
df['Rating'] = df['Rating'].astype('category')
df["Votes"]=df["Votes"].replace("-",np.nan)
df["Votes"] = df["Votes"].astype('float')
df["Votes"].fillna(0, inplace=True)
df["Reviews"]=df["Reviews"].replace("-",np.nan)
df["Reviews"] = df["Reviews"].astype('float')
df["Reviews"].fillna(0, inplace=True)
df['Ratio_Min_Avg_Cost']=df['Minimum_Order']/df['Average_Cost']


#dropping column "Cuisines" and "Location":
df=df.drop(['Location','Cuisines','Indonesian','Bubble Tea','Tamil','Cantonese','Konkan','Bohri','Gujarati','Sri Lankan','Portuguese','Charcoal Chicken','Nepalese','Tex-Mex','South American','Israeli','Greek'
],axis=1)

#Creating dummy variables for "Locality" & "City" variable:
train_df=pd.get_dummies(df,columns=["Locality","City"],drop_first=True)
print(train_df.shape)
#Converting categorical data into numeric data:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['Delivery_Time'] = le.fit_transform(train_df['Delivery_Time'])
le.classes_
array(['10 minutes', '120 minutes', '20 minutes', '30 minutes',
       '45 minutes', '65 minutes', '80 minutes'], dtype=object)
train_df.head(10)

#Standardizationof columns as they have different metric/unit:
#col_names=['Average_Cost','Minimum_Order','Rating','Votes','Reviews']
#features=train_df[col_names]
#from sklearn.preprocessing import StandardScaler
#scaler=StandardScaler()
#scaler.fit(features.values)
#features = scaler.transform(features.values)
#scaled_features = pd.DataFrame(features, columns = col_names)
#New train dataset:
#train_new=pd.concat([train_df.drop(['Average_Cost','Minimum_Order','Rating','Votes','Reviews'],axis=1),scaled_features],axis=1)
#train_new.head()
train_df.head()

#train_new.shape
train_df.shape

#list(train_new.columns.values)
#Data Pre-processing and Feature Engineering of test dataset:
test=pd.read_excel("C:/datasets/mydatate.xlsx")
test.shape
test.dtypes
test.head(10)
test.isnull().sum()
#test['Restaurant'] = test['Restaurant'].str.replace(r'^ID_', '')
test['Restaurant']=test['Restaurant'].astype('category')
test['Location']=test['Location'].str.strip().str.lower()
test['Location'].str.split(',',expand=True)

# Make a new feature "Locality" from "Location":
test['Locality']=test['Location'].str.split(',').str[0]
#Adding new feature "City" from "Location"variable:

test['City']=[i.replace('mico layout, stage 2, btm layout,bangalore',"Bangalore").replace('d-block, sector 63, noida',"Noida")
            .replace("sector 1, noida","Noida").replace("fti college, law college road, pune","Pune")
            .replace('delhi university-gtb nagar',"Delhi").replace('laxman vihar industrial area, sector 3a, gurgoan',"Gurgaon")
            .replace('sector 14, noida',"Noida").replace("delhi administration flats, timarpur","Delhi")
            .replace("mumbai central","Mumbai").replace("rmz centennial, i gate, whitefield","Bangalore")
            .replace("delhi high court, india gate","Delhi").replace("mg road, pune","Pune")
            .replace("nathan road, mangaldas road, pune","Pune").replace("sandhurst road, mumbai cst area","Mumbai")
            .replace("sector 3, marathalli","Bangalore").replace("majestic","Not Given").replace("delhi cantt.","Delhi")
            .replace("yerawada, pune, maharashtra","Pune").replace("dockyard road, mumbai cst area","Mumbai")
            .replace("babarpur, new delhi, delhi","Delhi").replace("pune university","Pune").replace("sector 63a,gurgaon","Gurgaon")
            .replace("moulali, kolkata",'Kolkata').replace("chandni chowk, kolkata","Kolkata").replace("tejas nagar colony, wadala west, mumbai","Mumbai")
            .replace("raja bazar, kolkata","Kolkata").replace("tiretti, kolkata","Kolkata").replace("hyderabad public school, begumpet","Hyderabad")
            .replace("gora bazar, rajbari, north dumdum, kolkata","Kolkata").replace("noorkhan bazaar, malakpet, hyderabad","Hyderabad")
            .replace("musi nagar, malakpet, hyderabad","Hyderabad").replace("panjetan colony, malakpet, hyderabad","Hyderabad")
            .replace("chatta bazaar, malakpet, hyderabad","Hyderabad")
            .replace("jaya nagar, saidabad, hyderabad","Hyderabad")
            .replace("btm layout 1, electronic city","Bangalore") for i in test['Location']]
#test['Cuisines'] = test['Cuisines'].str.split(',', expand= False)
#mlb = MultiLabelBinarizer()
#test = test.join(pd.DataFrame(mlb.fit_transform(test.pop('Cuisines')),columns=mlb.classes_,index=test.index))
#Creating unique feature for every unique observation in "Cuisines" variable (dummy variable):

test['Cuisines'] = test['Cuisines'].str.replace('Poké','Poke')
test['Cuisines'] = test['Cuisines'].str.replace('Coffee','Tea')
test['Cuisines'] = test['Cuisines'].str.replace('Hyderabadi','Biryani')

cuisines_list_test = test['Cuisines'].str.split(', ')

from collections import Counter
cuisines_counter_test = Counter(([a for b in cuisines_list_test.tolist() for a in b]))

for cuisine in cuisines_counter_test.keys():
    test[cuisine] = 0
    test.loc[test['Cuisines'].str.contains(cuisine), cuisine] = 1
test['Average_Cost'] = test['Average_Cost'].str.replace("[^0-9]","")
test['Average_Cost'] = test['Average_Cost'].str.strip()
test['Average_Cost']=pd.to_numeric(test['Average_Cost'])
test['Average_Cost'].isnull().sum()

test['Minimum_Order']=test['Minimum_Order'].str.replace("[^0-9]","")
test['Minimum_Order']=test['Minimum_Order'].str.strip()
test['Minimum_Order']=pd.to_numeric(test['Minimum_Order'])
test['Minimum_Order'].isnull().sum()

test['Rating']=test['Rating'].replace('Opening Soon',np.nan).replace("NEW",np.nan)
test['Rating'] = test['Rating'].replace("-",np.nan)
test['Rating']=pd.to_numeric(test['Rating'])
test["Rating"].fillna(0, inplace=True)
test['Rating_Category']=pd.qcut(test['Rating'],q=5,precision=0,labels=False)
test['Rating_Category'].value_counts()

test['Rating'] = test['Rating'].astype('category')
test["Votes"]=test["Votes"].replace("-",np.nan)
test["Votes"] = test["Votes"].astype('float')
test["Votes"].fillna(0, inplace=True)
test["Reviews"]=test["Reviews"].replace("-",np.nan)
test["Reviews"] = test["Reviews"].astype('float')
test["Reviews"].fillna(0, inplace=True)
test['Ratio_Min_Avg_Cost']=test['Minimum_Order']/test['Average_Cost']

#dropping column "Cuisines" and "Location":
test1=test.drop(["Location","Cuisines"],axis=1)

#Creating dummy variables for "Locality" & "City" variable:
test_df=pd.get_dummies(test1,columns=["Locality","City"],drop_first=True)
print(test_df.shape)

#Standardizationof columns as they have different metric/unit:
#col_names=['Average_Cost','Minimum_Order','Rating','Votes','Reviews']
#features=test_df[col_names]
#from sklearn.preprocessing import StandardScaler
#scaler=StandardScaler()
#scaler.fit(features.values)
#features = scaler.transform(features.values)
#scaled_features_test = pd.DataFrame(features, columns = col_names)
#test_new=pd.concat([test_df.drop(['Average_Cost','Minimum_Order','Rating','Votes','Reviews'],axis=1),scaled_features_test],axis=1)
test_df.shape

#list(test_new.columns.values)
test_df.head()

# Save into new csv file:
train_df.to_csv("train_trained.csv",index=False)
test_df.to_csv("test_trained.csv",index=False)

#Import converted data set:
train_X=pd.read_csv("C:/datasets/train_trained.csv")
test_X=pd.read_csv("C:/datasets/train_trained.csv")

train_X.shape

holdout=test_X
holdout.shape

#Splittig data into train & test set:

#from sklearn.model_selection import train_test_split
X = train_X.drop(['Delivery_Time'],axis=1)
y = train_X['Delivery_Time']
##train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.20,random_state=0)
X.shape,y.shape

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf =SVC()
scoring = 'accuracy'
score = cross_val_score(clf,X ,y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print(score.mean())

# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
Text(0, 0.5, 'Cross-Validated Accuracy')


# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=1)
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

clf =RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(clf,X ,y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print(score.mean())

clf =DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf,X ,y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print(score.mean())

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
score = cross_val_score(clf,X ,y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print(score.mean())

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
score = cross_val_score(clf,X ,y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print(score.mean())

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
score = cross_val_score(clf,X ,y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print(score.mean())

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.20,random_state=0)
categorical_features_indices = np.where(train_X.dtypes == 'object')[0]
categorical_features_indices

from catboost import CatBoostClassifier
cat = CatBoostClassifier(loss_function='MultiClass', 
                         eval_metric='Accuracy', 
                         depth=6,
                         random_seed=42, 
                         iterations=1000, 
                         learning_rate=0.07,
                         leaf_estimation_iterations=1,
                         l2_leaf_reg=1,
                         bootstrap_type='Bayesian', 
                         bagging_temperature=1, 
                         random_strength=1,
                         od_type='Iter', 
                         od_wait=200)
cat.fit(train_X,train_y, verbose=50,
        use_best_model=True,
        cat_features=categorical_features_indices,
        eval_set=[(train_X, train_y),(test_X, test_y)],
        plot=False)

predictions = cat.predict(test_X)
print('accuracy:', accuracy_score(test_y, predictions))

feature_imp = pd.DataFrame(sorted(zip(cat.feature_importances_, X.columns), reverse=True)[:50], 
                           columns=['Value','Feature'])
plt.figure(figsize=(15,15))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Catboost Features')
plt.tight_layout()
plt.show()

err = []
y_pred_tot = []

fold = KFold(n_splits=10, shuffle=True, random_state=0)

for train_index, test_index in fold.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    cat = CatBoostClassifier(loss_function='MultiClass', 
                         eval_metric='Accuracy', 
                         depth=6,
                         random_seed=42, 
                         iterations=1000, 
                         learning_rate=0.07,
                         leaf_estimation_iterations=1,
                         l2_leaf_reg=1, 
                         bootstrap_type='Bayesian', 
                         bagging_temperature=1, 
                         random_strength=1,
                         od_type='Iter', 
                         od_wait=200)
    cat.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=200, cat_features=categorical_features_indices)

    y_pred_cat = cat.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test,y_pred_cat))

    err.append(accuracy_score(y_test,y_pred_cat))
    p = cat.predict(holdout)
    y_pred_tot.append(p)

np.mean(err,0)

cat_final = np.mean(y_pred_tot,0).round().astype(int)
cat_final

y_pred_class = le.inverse_transform(cat_final)
df_sub = pd.DataFrame(data=y_pred_class, columns=['Delivery_Time'])
df_sub.head()

df_sub['Delivery_Time'].value_counts()

writer = pd.ExcelWriter('submssion_new.xlsx', engine='xlsxwriter')
df_sub.to_excel(writer,sheet_name='Sheet1', index=False)
writer.save()
 
 
 
 
import pickle 
model_file=open("catecate.pkl","wb")##to serialize
pickle.dump(cat,model_file)
model_file.close()##always remember to close it
