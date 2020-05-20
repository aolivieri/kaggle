# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:29:59 2020

@author: Alexandre
"""


# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load our plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

#print(dataset_train.columns.tolist())

# Let's sort them in descending order!
#dataset_train.isnull().sum().sort_values(ascending = False)
# Let's calculate the percentages, even more informative!
#dataset_train.isnull().mean().sort_values(ascending = False)
# Excuse me, can we have a plot please?!
#sns.heatmap(dataset_train.isnull(), yticklabels = False, cbar = False, cmap = 'plasma')
#train.describe(include = 'all')
#sns.countplot(train['Pclass'])
#train.Name.value_counts()
#train['Age'].hist(bins = 50, color = 'blue')
# Countplot for 'Survived' variable
#sns.countplot(dataset_train['Survived'])

#Correlation with age to input the missing values
#sns.heatmap(dataset_train.corr(), annot = True)
#age_group = dataset_train.groupby("Pclass")["Age"]

#print(age_group.median())

#Fill in missing ages
dataset_train.loc[dataset_train.Age.isnull(), 'Age'] = dataset_train.groupby("Pclass").Age.transform('median')
dataset_test.loc[dataset_test.Age.isnull(), 'Age'] = dataset_test.groupby("Pclass").Age.transform('median')

#print(dataset_train["Age"].isnull().sum())

# We will get rid of 'Cabin' for now, for simplicity reasons, but cu at Part 3!
dataset_train.drop('Cabin', axis = 1, inplace = True)
dataset_test.drop('Cabin', axis = 1, inplace = True)

#plt.figure(figsize = (16, 8))
#sns.distplot(dataset_train["Age"])
#plt.title("Age Histogram")
#plt.xlabel("Age")
#plt.show()

# Let's impute 'Embarked' missing values with the mode, which happens to be "S"!

from statistics import mode
dataset_train["Embarked"] = dataset_train["Embarked"].fillna(mode(dataset_train["Embarked"]))
dataset_test["Embarked"] = dataset_test["Embarked"].fillna(mode(dataset_test["Embarked"]))

# Convert 'Sex' variable to integer form!
dataset_train["Sex"][dataset_train["Sex"] == "male"] = 0
dataset_train["Sex"][dataset_train["Sex"] == "female"] = 1
dataset_test["Sex"][dataset_test["Sex"] == "male"] = 0
dataset_test["Sex"][dataset_test["Sex"] == "female"] = 1

# Convert 'Embarked' variable to integer form!
dataset_train["Embarked"][dataset_train["Embarked"] == "S"] = 0
dataset_train["Embarked"][dataset_train["Embarked"] == "C"] = 1
dataset_train["Embarked"][dataset_train["Embarked"] == "Q"] = 2
dataset_test["Embarked"][dataset_test["Embarked"] == "S"] = 0
dataset_test["Embarked"][dataset_test["Embarked"] == "C"] = 1
dataset_test["Embarked"][dataset_test["Embarked"] == "Q"] = 2

# We'll drop the following features for now, but more to follow...
dataset_train.drop(['Name', 'Ticket'], axis = 1, inplace = True)
dataset_test.drop(['Name', 'Ticket'], axis = 1, inplace = True)

dataset_train.drop('Fare', axis = 1, inplace = True)
dataset_test.drop('Fare', axis = 1, inplace = True)

from sklearn.model_selection import train_test_split

# Here is out local validation scheme!
#X_train, X_test, y_train, y_test = train_test_split(dataset_train.drop(['Survived'], axis = 1), 
#                                                    dataset_train['Survived'], test_size = 0.2, 
#                                                    random_state = 2)

y_train2 = dataset_train['Survived']
X_train2 = dataset_train.drop('Survived',axis=1)

from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression(max_iter = 10000)
logisticRegression.fit(X_train2, y_train2)

# Predict!
X_test = dataset_test
dataset_test.isnull().sum().sort_values(ascending = False)
dataset_test['Survived'] = logisticRegression.predict(X_test)

#round(np.mean(predictions), 2)

from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test, predictions))


# OLD
#X_train = pd.get_dummies(dataset_train['Sex'])
#y_train = dataset_train['Survived']

# Fitting Simple Linear Regression to the Training set
# Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)

# Predicting the Test set results
#dataset_test = pd.read_csv('test.csv')
#X_test = pd.get_dummies(dataset_test['Sex'])
#dataset_test['Survived'] = classifier.predict(X_test)

dataset_test[['PassengerId', 'Survived']].to_csv('kaggle_submission2.csv', index = False)

my_submission = pd.read_csv('kaggle_submission2.csv')
print(my_submission.head())
print(my_submission.tail())


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()