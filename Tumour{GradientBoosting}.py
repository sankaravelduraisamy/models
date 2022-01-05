import pandas as pd

df = pd.read_csv("C:/data science/pending/Ensemble_Techniques_Problem Statement/Datasets/Tumour.csv")

# Dummy variables
df.head()
df.info()

# n-1 dummy variables will be created for n categories
df = pd.get_dummies(df, columns = ["Radiation", "Gender"], drop_first = True)

df.head()


# Input and Output Split
predictors = df.loc[:, df.columns!="Concavity_se"]
type(predictors)

target = df["Concavity_se"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

# Refer to the links
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble

from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, boost_clf.predict(x_test))
accuracy_score(y_test, boost_clf.predict(x_test))


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
boost_clf2.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, boost_clf2.predict(x_test))
accuracy_score(y_test, boost_clf2.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, boost_clf2.predict(x_train))
