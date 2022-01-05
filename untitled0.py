#pip install pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go
import scipy.stats as stats
import pylab
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer



# Read data into Python
data = pd.read_csv("C:/data science/notes/archive/insta.csv")
data.shape
data.columns
data.dtypes


#type casting
# Now we will convert 'float64' into 'int64' type. 
data.cyl = data.cyl.astype('float64') 
data.hp= data.hp.astype('float64') 
data.vs = data.vs.astype('float64') 
data.am = data.am.astype('float64')
data.gear = data.gear.astype('float64') 
data.carb = data.carb.astype('float64')
data.dtypes

#Identify duplicates records in the data
duplicate = data.duplicated()
duplicate
sum(duplicate)

#Removing Duplicates
data1 = data.drop_duplicates() 
sum(duplicate)


# drop emp_name column
data.drop(['mpg'],axis =1,inplace=True)
data.shape
data.columns


plt.bar(height = data.gmat, x = np.arange(1,774,1)) # initializing the parameter
plt.hist(data.gmat) #histogram
plt.boxplot(data.gmat) #boxplot


#Normal Quantile-Quantile Plot
# Checking Whether data is normally distributed
stats.probplot(data.gmat, dist="norm",plot=pylab)

stats.probplot(data.workex,dist="norm",plot=pylab)

#transformation to make workex variable normal
stats.probplot(np.log(data.workex),dist="norm",plot=pylab)

# z-distribution
# cdf => cumulative distributive function; # similar to pnorm in R
stats.norm.cdf(740,711,29)  # given a value, find the probability

# ppf => Percent point function; # similar to qnorm in R
stats.norm.ppf(0.975,0,1) # given probability, find the Z value

#t-distribution
stats.t.cdf(1.98,139) # given a value, find the probability; # similar to pt in R
stats.t.ppf(0.975, 139) # given probability, find the t value; # similar to qt in R


##############################################################
###### Data Preprocessing########################################

## import packages
##################  creating Dummy variables using dummies ###############
# we use ethinc diversity dataset  for this
df11 = pd.read_csv("C:/data science/1 finished assignments/others/Datasets_EDA/ethnic diversity.csv")
df11.dtypes
# Create dummy variables on categorcal columns
df11_new = pd.get_dummies(df11)

### we have created dummies for all categorical columns



#######lets us see using one hot encoding works
#df1 = pd.read_csv("C:/data science/1 finished assignments/others/Datasets_EDA/ethnic diversity.csv")
# creating instance of one-hot-encoder
enc = OneHotEncoder()
enc_df11 = pd.DataFrame(enc.fit_transform(df11).toarray())

############################################################
########/\/\/\/\/\/\\\/\/\/\//\/\\/\\\/\/\/\/\\\####################

#df2 = pd.read_csv('C:/data science/1 finished assignments/others/Datasets_EDA/ethnic diversity.csv')
#df2.drop(['Employee_Name', 'EmpID','Zip'], axis =1, inplace =True)
# creating instance of labelencoder
labelencoder = LabelEncoder()
X = df11.iloc[:, 0:9]
y = df11['Race']
y=df11.iloc[:,9:]

df11.columns

X['Sex']= labelencoder.fit_transform(X['Sex'])
X['MaritalDesc'] = labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc'] = labelencoder.fit_transform(X['CitizenDesc'])


########## label encode y

y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)

### we have to convert y to data frame so that we can use concatenate function

# concatenate X and y

data_new = pd.concat([X, y], axis =1)
## rename column name
data_new.columns
data_new = data_new.rename(columns={0:'Type'})
#################################################################################
## load data set

#ethnic= pd.read_csv("C:/data science/1 finished assignments/others/Datasets_EDA/ethnic diversity.csv")

#ethnic.columns


df11.isna().sum()
df11.isnull().sum()

scale()
# Normalization function using z std. all are continuous data.
def std_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df11_norm = std_func(df11.iloc[:,10:12])
df11_norm.describe()

##################################################################################
###################### Outlier Treatment ###################################

#df = pd.read_csv("C:/data science/1 finished assignments/others/Datasets_EDA/ethnic diversity.csv") # for doing modifications
#df.dtypes
#df.isna().sum()

# let's find outliers in RM
sns.boxplot(df11.Salaries)
plt.title('Boxplot')
plt.show() #

sns.boxplot(df11.age)
plt.title('Boxplot')
plt.show() #


# Detection of outliers (find limits for RM based on IQR)

IQR = df11['Salaries'].quantile(0.75) - df11['Salaries'].quantile(0.25)
lower_limit = df11['Salaries'].quantile(0.25) - (IQR * 1.5)
upper_limit = df11['Salaries'].quantile(0.75) + (IQR * 1.5)



############### 1. Remove (let's trimm the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set

outliers_df11 = np.where(df11['Salaries'] > upper_limit, True, np.where(df11['Salaries'] < lower_limit, True, False))
df11_trimmed = df11.loc[~(outliers_df11), ]
df11.shape,df11_trimmed.shape

# let's explore outliers in the trimmed dataset
sns.boxplot(df11_trimmed.Salaries);plt.title('Boxplot');plt.show()

#we see no outiers


####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
df11['data_replaced']= pd.DataFrame(np.where(df11['Salaries'] > upper_limit, upper_limit, np.where(df11['Salaries'] < lower_limit, lower_limit, data['Salaries'])))
sns.boxplot(df11.df11_replaced);plt.title('Boxplot');plt.show()




###################### 3. Winsorization #####################################

windsoriser = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Salaries'])
df11_t = windsoriser.fit_transform(data[['Salaries']])
#df_gmat = windsoriser.fit_transform(df[['Salaries']])

# we can inspect the minimum caps and maximum caps 
#windsoriser.left_tail_caps_, windsoriser.right_tail_caps_

# lets see boxplot
sns.boxplot(df11_t.Salaries);plt.title('Boxplot');plt.show()




###################################################################################
#################### Missing Values Imputation ##################################

# load the dataset
# use modified ethnic dataset
#df_raw = pd.read_csv('C:\\Users\\prakruthi\\Desktop\\dataset\\ethnic diversity.csv') # raw data without doing any modifications
#df = pd.read_csv('C:\\Users\\prakruthi\\modethnic.csv') # for doing modifications

# check for count of NA'sin each column
df11.isna().sum()

# There are 3 columns that have missing data ---Create an imputer object that fills 'Nan' values of SEX,MaritalDesc,Salaries
# Mean and Median imputer are used for numeric data (Salaries)
# mode is used for discrete data (SEX,MaritalDesc) 

# for Mean,Meadian,Mode imputation we can use Simple Imputer or df.fillna()

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df11["Salaries"] = pd.DataFrame(mean_imputer.fit_transform(df11[["Salaries"]]))
df11["Salaries"].isna().sum()  # all 2 records replaced by mean 


#df = pd.read_csv('C:\\Users\\prakruthi\\modethnic.csv')
df11["Salaries"].isna().sum() 
# Median Imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df11["Salaries"] = pd.DataFrame(median_imputer.fit_transform(df11[["Salaries"]]))
df11["Salaries"].isna().sum()  # all 2 records replaced by median 

df11.isna().sum()
# Mode Imputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df11["Sex"] = pd.DataFrame(mode_imputer.fit_transform(df11[["Sex"]]))
df11["MaritalDesc"] = pd.DataFrame(mode_imputer.fit_transform(df11[["MaritalDesc"]]))
df11.isnull().sum()  # all SEX,MaritalDesc,Salaries records replaced by mode

##############################################################################
################## Type casting###############################################

#data = pd.read_csv("C:\\Users\\prakruthi\\Desktop\\dataset\\ethnic diversity.csv")
#data.dtypes

##############################################################################
#################### String manipulation ######################################

word = "Keep Adapting"
print(word)

word

#Accessing

word = "Keep Adapting"

letter=word[4]

print(letter)

#length 
word = "Keep Adapting"

len(word)

letters = "wenf bwehfwfnewfje    "
len(letters)

#finding
'''
'''
word = "Keep Adapting"
print(word.count('p')) # count how many times p is in the string
print(word.find("keep")) # find the word t in the string
print(word.index("Adapting")) # find the letters Adapting in the string

s = "The world won't care about your self-esteem. The world will expect you to accomplish something BEFORE you feel good about yourself."

print(s.count('  '))

#Slicing
y="             "
print(y.count(' '))
word1 = "_$_the internet frees us from the responsibility of having to retain anything in our long-term memory@_."

print (word1[0])
print(word1[-1]) #get one char of the word
print (word1[0:1]) #get one char of the word (same as above)
print (word1[0:3]) #get the first three char
print (word1[:3]) #get the first three char
print (word1[-3:]) #get the last three char
print (word1[3:]) #get all but the three first char
print (word1[:-3]) #get all but the three last character
print (word1[3:-3]) #get all 


# spliting

word3 = 'Good health is not something we can buy. However, it can be an extremely valuable savings account.'

a =word3.split(' ')
a # Split on whitespace
['Good','health','is','not','something','we','can','buy.','However,','it','can','be','an','extremely','valuable','savings','account.']
type(a)
# Startswith / Endswith
word3 = 'Remain calm, because peace equals power.'
word3.startswith("R")
word3.endswith("e")
word3.endswith(".")

# repeat string 

print( " * "* 10 )# 


# replacing

word4 = "Live HapLive"

word4.replace("Live", "Lead Life")

dir(string)

# Reversing
string = "eert a tsniaga pu mih deit yehT .meht htiw noil eht koot dna tserof eht otni emac sretnuh wef a ,yad enO "

print (''.join(reversed(string)))

#Strip
#Python strings have the strip(), lstrip(), rstrip() methods for removing
#any character from both ends of a string.

#If the characters to be removed are not specified then white-space will be removed




