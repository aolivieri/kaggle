# -*- coding: utf-8 -*-
"""
Created on Sun May 10 10:36:27 2020

@author: Alexandre
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load out plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Load our warnings library, ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Read train data
train = pd.read_csv('train.csv')

# Read test data
test = pd.read_csv('test.csv')

# A trick to keep 'Survived' in our dataset
test['Survived'] = np.nan

# Train + test concatenation
full = pd.concat([train, test])

full.head()

# Let's calculate percentages of missing values!
full.isnull().mean().sort_values(ascending = False)

from statistics import mode


# Convert 'Sex' variable to integer form!
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
full['Sex'] = le.fit_transform(full['Sex'])
#full["Sex"][full["Sex"] == "male"] = 0
#full["Sex"][full["Sex"] == "female"] = 1


# Let's impute 'Embarked' missing values with the mode, which happens to be "S"!
full["Embarked"] = full["Embarked"].fillna(mode(full["Embarked"]))
# Convert 'Embarked' variable to integer form!
#full["Embarked"] = le.fit_transform(full["Embarked"])
full["Embarked"][full["Embarked"] == "S"] = 0
full["Embarked"][full["Embarked"] == "C"] = 1
full["Embarked"][full["Embarked"] == "Q"] = 2

#sns.heatmap(full.corr(), annot = True)


# Grouping by Pclass and using a lambda to impute the Fare median
full['Fare'] = full.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
#map Fare values into groups of numerical values
#full['FareBand'] = pd.qcut(full['Fare'], 4, labels = [1, 2, 3, 4])
#full = full.drop('Fare', axis = 1)


# ****** CABIN *******
#full = full.drop(['Cabin'], axis=1)
# Replace missing values with 'U' for Cabin
full['Cabin'] = full['Cabin'].fillna('U')
full.isnull().mean().sort_values(ascending = False)
#full['Cabin'].unique().tolist()
# Let's import our regular expression matching operations module!
import re
# Extract (first) letter!
full['Cabin'] = full['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
#full['Cabin'].unique().tolist()
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}
full['Cabin'] = full['Cabin'].map(cabin_category)
#full['Cabin'].unique().tolist()



# Extract the salutation!
full['Title'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
full['Title'].unique().tolist()
# Look at salutations percentages
full['Title'].value_counts(normalize = True) * 100
# Bundle rare salutations: 'Other' category
full['Title'] = full['Title'].replace(['Rev', 'Dr', 'Col', 'Major', 
                                       'Capt', 'Dona', 'Jonkheer', 'Don'], 'Rare')
full['Title'] = full['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
full['Title'] = full['Title'].replace('Mlle', 'Miss')
full['Title'] = full['Title'].replace('Ms', 'Miss')
full['Title'] = full['Title'].replace('Mme', 'Mrs')
full['Title'].value_counts(normalize = True) * 100
title_category = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4,'Royal':5, 'Rare':6}
full[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
full['Title'] = full['Title'].map(title_category)
full['Title'].unique().tolist()

# Drop Ticket
full = full.drop('Ticket', axis = 1)

# Drop Name
full = full.drop('Name', axis = 1)


# Grouping by Pclass and using a lambda to impute the Age median
full['Age'] = full.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
#sort the ages into logical categories
#full["Age"] = full["Age"].fillna(-0.5)
#bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
#labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
#full['AgeGroup'] = pd.cut(full["Age"], bins, labels = labels)

#age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

#for x in range(len(full["AgeGroup"])):
#    if full['AgeGroup'][x] == "Unknown":
#        full['AgeGroup'][x] = age_title_mapping[full['Title'][x]]


# Engineer 'familySize' feature
full['familySize'] = full['SibSp'] + full['Parch'] + 1
full = full.drop(['SibSp', 'Parch'], axis = 1)
#full.familySize.replace(to_replace = [1], value = "single", inplace = True)
#full.familySize.replace(to_replace = [2,3], value = "small", inplace = True)
#full.familySize.replace(to_replace = [4,5], value = "medium", inplace = True)
#full.familySize.replace(to_replace = [6, 7, 8, 11], value = "large", inplace = True)


# Recover test dataset
test = full[full['Survived'].isna()].drop(['Survived'], axis = 1)
test.head()

# Recover train dataset
train = full[full['Survived'].notna()]
train.head()

# Cast 'Survived' back to integer
train['Survived'] = train['Survived'].astype(np.int8)

X = train.drop(['Survived', 'PassengerId'], axis = 1)
y = train['Survived']

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
print(X)
print(y)

# Import module for dataset splitting
from sklearn.model_selection import train_test_split

# Here is out local validation scheme!
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y.values, test_size = 0.2, 
                                                    random_state = 2)

import tensorflow as tf
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 1000)

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.65)

from sklearn.metrics import accuracy_score
# Calculate the accuracy for our powerful random forest!
print("accuracy is: ", round(accuracy_score(y_test, y_pred), 2))

# Predict!
sc_test = StandardScaler()
test_tmp = sc_test.fit_transform(test.drop(['PassengerId'], axis = 1))
y_pred2 = ann.predict(test_tmp)
test['Survived'] = (y_pred2 > 0.65)

# Cast 'Survived' back to integer
test['Survived'] = test['Survived'].astype(np.int8)

# Write test predictions for final submission
test[['PassengerId', 'Survived']].to_csv('kaggle_submission_titanic4_nn.csv', index = False)

my_submission = pd.read_csv('kaggle_submission_titanic4_nn.csv')
print(my_submission.head())
print(my_submission.tail())



