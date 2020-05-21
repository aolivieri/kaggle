# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:20:17 2020

@author: Alexandre
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load out plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

from statistics import mode

# Load our warnings library, ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Read train data
train = pd.read_csv('train.csv')
#X = train.iloc[:,:-1].values
#y = train.iloc[:,-1].values

# Read test data
test = pd.read_csv('test.csv')

# A trick to keep 'Survived' in our dataset
test['SaleCondition'] = np.nan

# Train + test concatenation
full = pd.concat([train, test])

#full.head()

# Let's calculate percentages of missing values!
is_null = full.isnull().mean().sort_values(ascending = False)

empty_variables = []
few_nans_variables = []
for index,value in is_null.items():
    if index != 'SalePrice':
        if value > 0.30 :
            empty_variables.append(index)
        elif value != 0:
            few_nans_variables.append(index)

# Drop features with lots of NaNs
full = full.drop(empty_variables, axis = 1)

# Learn more about the procedure below
#thresh = len(full) * .45
#full.dropna(thresh = thresh, axis = 1, inplace = True)


# Categorical variables
variable_types = full.dtypes
categorical_variables = []
for index, value in variable_types.items():
    if value == 'object':
        categorical_variables.append(index)


for few_nans_variable in few_nans_variables:
    full[few_nans_variable] = full[few_nans_variable].fillna(mode(full[few_nans_variable]))

full = pd.get_dummies(data=full, columns=categorical_variables)

# Drop other useless variables
full.drop(['Id'], axis = 1)

# Recover test dataset
test = full[full['SalePrice'].isna()].drop(['SalePrice'], axis = 1)
#test.head()

# Recover train dataset
train = full[full['SalePrice'].notna()]
#train.head()
X = train.drop(['SalePrice'], axis = 1)
y = train['SalePrice']

# Import module for dataset splitting
from sklearn.model_selection import train_test_split

# Here is out local validation scheme!
X_train, X_test, y_train, y_test = train_test_split(X , 
                                                    y, test_size = 0.2, 
                                                    random_state = 2)

# We'll use a logistic regression model again, but we'll go to something more fancy soon! 
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression(max_iter = 10000)
logisticRegression.fit(X_train, y_train)

# Predict!
predictions = logisticRegression.predict(X_test)

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,predictions))

from sklearn.metrics import mean_squared_log_error
np.sqrt(mean_squared_log_error(y_test,predictions))


test['SalePrice'] = logisticRegression.predict(test)


# Write test predictions for final submission
test[['Id', 'SalePrice']].to_csv('kaggle_submission.csv', index = False)

my_submission = pd.read_csv('kaggle_submission.csv')
print(my_submission.head())
print(my_submission.tail())





from sklearn.model_selection import KFold

# Set our robust cross-validation scheme!
kf = KFold(n_splits = 5, random_state = 2)

from sklearn.model_selection import cross_val_score

# Print our CV accuracy estimate:
print(cross_val_score(logisticRegression, X_test, y_test, cv = kf).mean())

from sklearn.ensemble import RandomForestClassifier

#Initialize randomForest
randomForest = RandomForestClassifier(random_state = 2)

# Set our parameter grid
param_grid = { 
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [100, 300, 500],
    'max_features': ['auto', 'log2'],
    'max_depth' : [3, 5, 7]    
}

from sklearn.model_selection import GridSearchCV

# Grid search
randomForest_CV = GridSearchCV(estimator = randomForest, param_grid = param_grid, cv = 5)
randomForest_CV.fit(X_train, y_train)

# Print best hyperparameters
randomForest_CV.best_params_

# Define our optimal randomForest algo
randomForestFinalModel = RandomForestClassifier(random_state = 2, 
                                                criterion = 'gini', 
                                                max_depth = 7, 
                                                max_features = 'auto', 
                                                n_estimators = 300)

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
test[['PassengerId', 'Survived']].to_csv('kaggle_submission3.csv', index = False)

my_submission = pd.read_csv('kaggle_submission3.csv')
print(my_submission.head())
print(my_submission.tail())



