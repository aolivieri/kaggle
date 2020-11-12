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
#full["Sex"][full["Sex"] == "male"] = 0
#full["Sex"][full["Sex"] == "female"] = 1
#full['Sex'] = pd.to_numeric(full['Sex'])

# Convert 'Embarked' variable to integer form!
#full["Embarked"][full["Embarked"] == "S"] = 0
#full["Embarked"][full["Embarked"] == "C"] = 1
#full["Embarked"][full["Embarked"] == "Q"] = 2
#full["Embarked"] = pd.to_numeric(full["Embarked"])

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

full = pd.get_dummies(full, columns=['Sex', 'Embarked', 'Cabin', 'Title'], drop_first=True)


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

# Import module for dataset splitting
from sklearn.model_selection import train_test_split

X = train.drop(['Survived', 'PassengerId'], axis = 1)
y = train['Survived']
# Here is out local validation scheme!
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, test_size = 0.2, 
                                                    random_state = 2)
#X_train = X
#y_train = y

from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate=0.02, 
                           n_estimators=600, 
                           objective='binary:logistic',
                           silent=True, 
                           nthread=1)

# Set our parameter grid
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Grid search
classifier_CV = GridSearchCV(estimator = classifier, 
                             param_grid = params, 
                             cv = 5)
classifier_CV.fit(X_train, y_train)
classifier_CV.best_params_


# Randomized Search
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(classifier, 
                                   param_distributions=params, 
                                   n_iter=5, 
                                   scoring='roc_auc', 
                                   n_jobs=4, 
                                   cv=skf.split(X_train,y_train), 
                                   verbose=3, 
                                   random_state=1001 )
random_search.fit(X_train, y_train)
random_search.best_params_


# Tune the final classifier based on the hyperparameters found before

classifier_final = XGBClassifier(learning_rate=0.02, 
                                 n_estimators=600, 
                                 objective='binary:logistic',
                                 silent=True, 
                                 nthread=1,
                                 colsample_bytree = 0.6,
                                 gamma = 2,
                                 min_child_weight = 1,
                                 subsample = 1.0,
                                 max_depth = 5
                                 )

classifier_final.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_final, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Predict!
predictions = classifier_final.predict(X_test)

from sklearn.metrics import accuracy_score

# Calculate the accuracy for our powerful random forest!
print("accuracy is: ", round(accuracy_score(y_test, predictions), 2))

# Predict!
test['Survived'] = classifier_final.predict(test.drop(['PassengerId'], axis = 1))

# Cast 'Survived' back to integer
test['Survived'] = test['Survived'].astype(np.int8)

# Write test predictions for final submission
test[['PassengerId', 'Survived']].to_csv('kaggle_submission_xgboost_fulldata2.csv', index = False)

my_submission = pd.read_csv('kaggle_submission_xgboost_fulldata.csv')
print(my_submission.head())
print(my_submission.tail())



