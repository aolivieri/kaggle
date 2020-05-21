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

# Let's impute 'Embarked' missing values with the mode, which happens to be "S"!
full["Embarked"] = full["Embarked"].fillna(mode(full["Embarked"]))

# Convert 'Sex' variable to integer form!
full["Sex"][full["Sex"] == "male"] = 0
full["Sex"][full["Sex"] == "female"] = 1

# Convert 'Embarked' variable to integer form!
full["Embarked"][full["Embarked"] == "S"] = 0
full["Embarked"][full["Embarked"] == "C"] = 1
full["Embarked"][full["Embarked"] == "Q"] = 2

#sns.heatmap(full.corr(), annot = True)

# Grouping by Pclass and using a lambda to impute the Age median
full['Age'] = full.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))

# Grouping by Pclass and using a lambda to impute the Fare median
full['Fare'] = full.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))

# Replace missing values with 'U' for Cabin
full['Cabin'] = full['Cabin'].fillna('U')

#full.isnull().mean().sort_values(ascending = False)

#full['Cabin'].unique().tolist()

# Let's import our regular expression matching operations module!
import re

# Extract (first) letter!
full['Cabin'] = full['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

full['Cabin'].unique().tolist()

cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}

full['Cabin'] = full['Cabin'].map(cabin_category)
full['Cabin'].unique().tolist()

# Extract the salutation!
full['Title'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
full['Title'].unique().tolist()

# Look at salutations percentages
full['Title'].value_counts(normalize = True) * 100

# Bundle rare salutations: 'Other' category
full['Title'] = full['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 
                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')


full['Title'].value_counts(normalize = True) * 100

title_category = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Other':5}

full['Title'] = full['Title'].map(title_category)
full['Title'].unique().tolist()

# Engineer 'familySize' feature
full['familySize'] = full['SibSp'] + full['Parch'] + 1

# Drop redundant features
full = full.drop(['Name', 'SibSp', 'Parch', 'Ticket'], axis = 1)

# Recover test dataset
test = full[full['Survived'].isna()].drop(['Survived'], axis = 1)
test.head()

# Recover train dataset
train = full[full['Survived'].notna()]
train.head()

# Cast 'Survived' back to integer
train['Survived'] = train['Survived'].astype(np.int8)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = train.drop(['Survived', 'PassengerId'], axis = 1)
X = sc.fit_transform(X)
y = train['Survived']

#Trying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)
#X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_


# Import module for dataset splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, test_size = 0.2, 
                                                    random_state = 2)


#Trying PCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#lda = LDA(n_components = 2)
#X_train = lda.fit_transform(X_train, y_train)
#X_test = lda.transform(X_test)


from sklearn.ensemble import RandomForestClassifier

# Define our optimal randomForest algo
randomForestFinalModel = RandomForestClassifier(random_state = 2, criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 300)

# Fit the model to the training set
randomForestFinalModel.fit(X_train, y_train)

# Predict!
predictions = randomForestFinalModel.predict(X_test)

from sklearn.metrics import accuracy_score

# Calculate the accuracy for our powerful random forest!
print("accuracy is: ", round(accuracy_score(y_test, predictions), 2))

# Predict!
test['Survived'] = randomForestFinalModel.predict(test.drop(['PassengerId'], axis = 1))

# Cast 'Survived' back to integer
test['Survived'] = test['Survived'].astype(np.int8)

# Write test predictions for final submission
test[['PassengerId', 'Survived']].to_csv('kaggle_submission_pca.csv', index = False)

my_submission = pd.read_csv('kaggle_submission_pca.csv')
print(my_submission.head())
print(my_submission.tail())



