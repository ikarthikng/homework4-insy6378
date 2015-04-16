__author__ = 'Karthik'
'''
The dataset for Homework 4 has been posted. The target variable is called 'class', which may be "good' or "bad'.
The target represents the credit worthiness of the customer. The potential predictors are self-explanatory
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

#Read the CSV file using pandas
data = pd.read_csv("creditData.csv")
final_dict = {}
class_list = []
accuracy_list = []
precision_list = []
recall_list = []
fscore_list = []

'''
Preprocessing the data for classification
'''
#Get the values from dataframe without the header
values = data.values
#Convert the dataframe values to a matrix
matrix = data.transpose().to_dict().values()
#Create a DictVectorizer
dv = DictVectorizer(sparse=False)
#Fit the transform to the dict vectorizer
x = dv.fit_transform(matrix)
#get the column names from the dict vectorizer
names = dv.get_feature_names()
#prepare the final needed dataframe for classification
data_frame = pd.DataFrame(x, columns=names)
values = data_frame.values
#Get the target values -- class = bad as the label for the data
target = values[:,[6]]
#Drop the columns which is not required in the train data
data_frame = data_frame.drop('class=good',1)
data_frame = data_frame.drop('class=bad',1)
#train data is prepared
train_data = data_frame.values

'''
------Logistic Regression Classifier-----
'''
x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size=0.1, random_state=0)
#Prepare the classifier
clf = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=33)
#perform a fit
clf.fit(x_train, np.ravel(y_train))
#Run the prediction score for the test set
y_pred = clf.predict(x_test)

class_list.append("Logistic Classifier")
accuracy_list.append(metrics.accuracy_score(np.ravel(y_test), np.ravel(y_pred)))
precision_list.append(metrics.precision_score(np.ravel(y_test), np.ravel(y_pred)))
recall_list.append(metrics.recall_score(np.ravel(y_test), np.ravel(y_pred)))
fscore_list.append(metrics.f1_score(np.ravel(y_test), np.ravel(y_pred)))

'''
------Support Vector Machine Classifier-----
'''
x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size=0.1, random_state=0)
#specify a classifier
clf = SVC(kernel='rbf', probability=True, random_state=33)
clf.fit(x_train, np.ravel(y_train))
#let us use the trained classifier to predict the test set
y_pred = clf.predict(x_test)
#Test the accuracy of the classifier

class_list.append("SVM Classifier")
accuracy_list.append(metrics.accuracy_score(np.ravel(y_test), np.ravel(y_pred)))
precision_list.append(metrics.precision_score(np.ravel(y_test), np.ravel(y_pred)))
recall_list.append(metrics.recall_score(np.ravel(y_test), np.ravel(y_pred)))
fscore_list.append(metrics.f1_score(np.ravel(y_test), np.ravel(y_pred)))

'''
------Random Forest Classifier-----
'''
x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size=0.2, random_state=0)
#specify a classifier
clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
clf.fit(x_train, np.ravel(y_train))
#let us use the trained classifier to predict the test set
y_pred = clf.predict(x_test)
#Test the accuracy of the classifier

class_list.append("Random Forest Classifier")
accuracy_list.append(metrics.accuracy_score(np.ravel(y_test), np.ravel(y_pred)))
precision_list.append(metrics.precision_score(np.ravel(y_test), np.ravel(y_pred)))
recall_list.append(metrics.recall_score(np.ravel(y_test), np.ravel(y_pred)))
fscore_list.append(metrics.f1_score(np.ravel(y_test), np.ravel(y_pred)))

'''
------Extra Trees Classifier-----
'''
x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size=0.2, random_state=0)
#specify a classifier
clf = ExtraTreesClassifier(n_estimators=100, max_features='log2')
clf.fit(x_train, np.ravel(y_train))
#let us use the trained classifier to predict the test set
y_pred = clf.predict(x_test)
#Test the accuracy of the classifier

class_list.append("Extra Trees Classifier")
accuracy_list.append(metrics.accuracy_score(np.ravel(y_test), np.ravel(y_pred)))
precision_list.append(metrics.precision_score(np.ravel(y_test), np.ravel(y_pred)))
recall_list.append(metrics.recall_score(np.ravel(y_test), np.ravel(y_pred)))
fscore_list.append(metrics.f1_score(np.ravel(y_test), np.ravel(y_pred)))

final_dict['A Classifier Name'] = class_list
final_dict['Accuracy'] = accuracy_list
final_dict['Precision'] = precision_list
final_dict['Recall'] = recall_list
final_dict['F1 Score'] = fscore_list

final_df = pd.DataFrame(final_dict)
print final_df