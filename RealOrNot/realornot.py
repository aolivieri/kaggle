# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:29:09 2020

@author: Alexandre
"""


# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()


#from sklearn.linear_model import SGDClassifier - BEST SO FAR
#classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)


from sklearn.linear_model import RidgeClassifier
classifier = RidgeClassifier()

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


from sklearn.metrics import precision_recall_fscore_support
score = precision_recall_fscore_support(y_test, y_pred, average='macro')
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='macro') 


testset = pd.read_csv('test.csv')
corpus_test = []
for i in range(len(testset)):
    review = re.sub('[^a-zA-Z]', ' ', testset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_test.append(review)

# Creating the Bag of Words model
cv_test = CountVectorizer(max_features = 1500)
X_validation = cv_test.fit_transform(corpus_test).toarray()

testset = testset.drop(['keyword','location', 'text'], axis=1)

testset['target'] = classifier.predict(X_validation)

testset.to_csv('kaggle_submission_RidgeClassifier.csv', index = False)

my_submission = pd.read_csv('kaggle_submission_RidgeClassifier.csv')
print(my_submission.head())
print(my_submission.tail())