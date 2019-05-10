# this file is used to plot ROC curves required in the 1A4 Imbalanced Task
import numpy as np
import csv
import data_processing_functions as dpf
from data_processing_functions import process_dict_reader
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn import neighbors

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

def plot_ROC(clf, x_test, y_test):
    # predict probabilities
    probs = clf.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    # calculate AUC
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    # show the plot
    pyplot.show()
#%%  use the feature of XDR_amount as the dataset
with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)
    dictlist = process_dict_reader(reader)

safe1, fraud1 = dpf.split_label(dictlist)
safe_set = dpf.get_column_set(safe1, 'xdr_amount')
fraud_set = dpf.get_column_set(fraud1, 'xdr_amount')
safe = dpf.get_column(safe1, 'xdr_amount')
fraud = dpf.get_column(fraud1, 'xdr_amount')

# safe set with label
safe_set = []
for line in safe:
    safe_set.append([int(line),0])
# fraud set with label
fraud_set = []
for line in fraud:
    fraud_set.append([int(line),1])

# combine and randomize the dataset for classifiers   
wholeset = safe_set + fraud_set
random.shuffle(wholeset)

# separate the label and feature sets
dataset = [[item[0]] for item in wholeset]
labelset = [item[1] for item in wholeset]

#%% get training and testing data(SMOTE and unSMOTEd)
x_array = np.array(dataset)
y_array = np.array(labelset)
usx = x_array.astype(np.float64)
usy = y_array.astype(np.float64)

x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size = 0.2)
sm = SMOTE()
x_res, y_res = sm.fit_resample(x_train, y_train)

#%% 1. Logistic Regression
# Test the original dataset
l_clf = LogisticRegression()
l_clf.fit(x_train,y_train)
l_prediction = l_clf.predict(x_test)
dpf.get_cl_result(l_prediction,y_test)
dpf.plot_ROC(l_clf, x_test, y_test)
# Test the SMOTED dataset
l_clf.fit(x_res,y_res)
l_prediction = l_clf.predict(x_test)
dpf.get_cl_result(l_prediction,y_test)
plot_ROC(l_clf, x_test, y_test)

#%% 2. Nearest Neighbors
# Test the original dataset
l_clf = neighbors.KNeighborsClassifier(algorithm = 'auto')
l_clf.fit(x_train,y_train)
l_prediction = l_clf.predict(x_test)
dpf.get_cl_result(l_prediction,y_test)
dpf.plot_ROC(l_clf, x_test, y_test)
# Test the SMOTED dataset
l_clf.fit(x_res,y_res)
l_prediction = l_clf.predict(x_test)
dpf.get_cl_result(l_prediction,y_test)
plot_ROC(l_clf, x_test, y_test)

#%% 3. Neural Network
# Test the original dataset
l_clf = MLPClassifier()
l_clf.fit(x_train,y_train)
l_prediction = l_clf.predict(x_test)
dpf.get_cl_result(l_prediction,y_test)
dpf.plot_ROC(l_clf, x_test, y_test)
# Test the SMOTED dataset
l_clf.fit(x_res,y_res)
l_prediction = l_clf.predict(x_test)
dpf.get_cl_result(l_prediction,y_test)
plot_ROC(l_clf, x_test, y_test)


# In this lab, we also put the whole feature dataset to train these
# 3 classifiers and plotted the ROC curves, but it takes too much
# time. Therefore, in this file, we only present the procedure
# by one feature training dataset.