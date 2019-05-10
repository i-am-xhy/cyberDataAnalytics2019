
import csv
import statistics
from data_processing_functions import process_dict_reader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import data_processing_functions as dpf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from numpy import array
import numpy
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)

accuracies = []
n = 5
for i in range(n):

    train_files = []
    validate_files = []


    labels = dpf.get_columns(dictlist, ['label'])
    dataset = dpf.get_columns(dictlist, ['xdr_amount'])

    traindata, testdata, trainlabels, testlabels = train_test_split(dataset, labels, test_size = 0.2)
    predictions = []
    for row in testdata:
        xdrAmount = row[0]
        if xdrAmount>13000 and xdrAmount<60000:
            predictions.append(1)
        else:
            predictions.append(0)

    _,_,_,_,accuracy = dpf.get_cl_result(predictions, testlabels)
    accuracies.append(accuracy)


print("average accuracy over {} runs: {}".format(n, statistics.mean(accuracies)))