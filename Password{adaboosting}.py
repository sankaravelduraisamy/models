import pandas as pd

df = pd.read_csv("C:/data science/pending/Ensemble_Techniques_Problem Statement/Datasets/Password.csv")

# Dummy variables
df.head()
df.info()

# n-1 dummy variables will be created for n categories
df = pd.get_dummies(df, columns = ["Hack", "Genre"], drop_first = True)

df.head()


# Input and Output Split
predictors = df.loc[:, df.columns!="Characters"]
type(predictors)

target = df["Characters"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


# Refer to the links
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 500)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))
