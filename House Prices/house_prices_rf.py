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

is_null_train = train.isnull().mean().sort_values(ascending = False)

train['SalePrice'].describe()
#histogram
#sns.distplot(train['SalePrice']);

# Read test data
test = pd.read_csv('test.csv')
is_null_test = test.isnull().mean().sort_values(ascending = False)


# A trick to keep 'Survived' in our dataset
test['SaleCondition'] = np.nan

# Train + test concatenation
full = pd.concat([train, test])

#correlation matrix
corrmat = full.corr()
f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(full[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()

#missing data
total = full.isnull().sum().sort_values(ascending=False)
percent = (full.isnull().sum()/full.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data = missing_data.drop('SalePrice', axis = 0)
missing_data.head(20)

#dealing with missing data
full = full.drop((missing_data[missing_data['Total'] > 1]).index,1)
full = full.drop(full.loc[full['Electrical'].isnull()].index)
full = full.drop(full.loc[full['KitchenQual'].isnull()].index)
full = full.drop(full.loc[full['TotalBsmtSF'].isnull()].index)
full = full.drop(full.loc[full['BsmtUnfSF'].isnull()].index)
full = full.drop(full.loc[full['BsmtFinSF2'].isnull()].index)
full = full.drop(full.loc[full['BsmtFinSF1'].isnull()].index)
full = full.drop(full.loc[full['SaleType'].isnull()].index)
full = full.drop(full.loc[full['Exterior1st'].isnull()].index)
full = full.drop(full.loc[full['Exterior2nd'].isnull()].index)
full = full.drop(full.loc[full['GarageArea'].isnull()].index)
full = full.drop(full.loc[full['GarageCars'].isnull()].index)
full.isnull().sum().max() #just checking that there's no missing data missing...

# Let's calculate percentages of missing values!
#is_null = full.isnull().mean().sort_values(ascending = False)
#sns.heatmap(full.isnull(), yticklabel = False, cbar = False, cmap = 'plasma')

#full['FireplaceQu'] = full['FireplaceQu'].fillna('NA')


# Other utilities
#sns.countplot(train['XXX'])
#train.XXX.value_counts()
#train['XXX'].hist(bins = 50, color = 'blue')


#  lambda to impute the median
#full['Age'] = full.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))


full = pd.get_dummies(full)

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
#X_train = X
#y_train = y



from sklearn.model_selection import KFold

# Set our robust cross-validation scheme!
kf = KFold(n_splits = 5, random_state = 2)

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
randomForest_CV = GridSearchCV(estimator = randomForest, 
                               param_grid = param_grid, 
                               cv = 5)
randomForest_CV.fit(X_train, y_train)

# Print best hyperparameters
randomForest_CV.best_params_

# Define our optimal randomForest algo
randomForestFinalModel = RandomForestClassifier(random_state = 2, 
                                                criterion = 'entropy', 
                                                max_depth = 5, 
                                                max_features = 'auto', 
                                                n_estimators = 300)

# Fit the model to the training set
randomForestFinalModel.fit(X_train, y_train)

# Predict!
predictions = randomForestFinalModel.predict(X_test)

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(np.log(y_test),np.log(predictions)))

from sklearn.metrics import r2_score

# Calculate the accuracy for our powerful random forest!
print("accuracy is: ", round(r2_score(y_test, predictions), 2))

test['SalePrice'] = randomForestFinalModel.predict(test)


# Write test predictions for final submission
test[['Id', 'SalePrice']].to_csv('kaggle_submission_rf_another.csv', index = False)

my_submission = pd.read_csv('kaggle_submission_rf_another.csv')
print(my_submission.head())
print(my_submission.tail())



