##################### Solution of Q1 ##########################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
salary_data_train = pd.read_csv("C:/data science/naive bayes/Datasets_Naive Bayes (1)/SalaryData_Train.csv",encoding = "ISO-8859-1")
salary_data_test = pd.read_csv("C:/data science/naive bayes/Datasets_Naive Bayes (1)/SalaryData_Test.csv",encoding = "ISO-8859-1")

salary_data_train = pd.concat([salary_data_train,salary_data_test])


#preprocessing
for value in ['workclass', 'education',
          'maritalstatus', 'occupation',
          'relationship','race', 'sex',
          'native', 'Salary']:
            print (value,":", sum(salary_data_train[value] == '?'))
            
salary_data_train.describe()



##labelling one Hot EncodingPython
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
workclass_cat = le.fit_transform(salary_data_train.workclass)
education_cat = le.fit_transform(salary_data_train.education)
marital_cat   = le.fit_transform(salary_data_train.maritalstatus)
occupation_cat = le.fit_transform(salary_data_train.occupation)
relationship_cat = le.fit_transform(salary_data_train.relationship)
race_cat = le.fit_transform(salary_data_train.race)
sex_cat = le.fit_transform(salary_data_train.sex)
native_country_cat = le.fit_transform(salary_data_train.native)

#initialize the encoded categorical columns
salary_data_train['workclass_cat'] = workclass_cat
salary_data_train['education_cat'] = education_cat
salary_data_train['marital_cat'] = marital_cat
salary_data_train['occupation_cat'] = occupation_cat
salary_data_train['relationship_cat'] = relationship_cat
salary_data_train['race_cat'] = race_cat
salary_data_train['sex_cat'] = sex_cat
salary_data_train['native_country_cat'] = native_country_cat


#drop the old categorical columns from dataframe
dummy_fields = ['workclass', 'education', 'maritalstatus', 
                  'occupation', 'relationship', 'race',
                  'sex', 'native']
salary_data_train = salary_data_train.drop(dummy_fields, axis = 1)

salary_data_train
salary_data_train.describe()

salary_data_train.head(1)





num_features = ['age', 'workclass_cat', 'education_cat',
                'marital_cat', 'occupation_cat', 'relationship_cat', 'race_cat',
                'sex_cat', 'capitalgain', 'capitalloss', 'hoursperweek',
                ]
 
scaled_features = {}
for each in num_features:
    mean, std = salary_data_train[each].mean(), salary_data_train[each].std()
    scaled_features[each] = [mean, std]
    salary_data_train.loc[:, each] = (salary_data_train[each] - mean)/std
  
    

salary_data_train = salary_data_train[['age', 'educationno', 'capitalgain', 'capitalloss','hoursperweek','workclass_cat', 'education_cat',
                'marital_cat', 'occupation_cat', 'relationship_cat', 'race_cat',
                'sex_cat','native_country_cat','Salary']]

#feature scale
from sklearn.model_selection import train_test_split
features = salary_data_train.values[:,:13]
target = salary_data_train.values[:,13]
features_train, features_test, target_train, target_test = train_test_split(features,target, test_size = 0.2)


from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
clf = GaussianNB()
clf.fit(features_train, target_train)
#On test data
target_pred = clf.predict(features_test)
	
accuracy_score(target_test, target_pred, normalize = True)
# on training data
target_pred0 = clf.predict(features_train)
	
accuracy_score(target_train, target_pred0, normalize = True)
