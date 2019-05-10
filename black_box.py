
import csv
import math
import pandas
import statistics
import random
from data_processing_functions import process_dict_reader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
import data_processing_functions as dpf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from numpy import array
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)
    # print(dpf.get_distinct_in_column(dictlist, 'currencycode'))
    # for line in dictlist:
    #     print(line)

# dictlist = dpf.filter(dictlist, 'currencycode', 'AUD')
# safe, fraud = dpf.split_label(dictlist)
# safe = dpf.get_column(safe, 'xdr_amount')
# fraud = dpf.get_column(fraud, 'xdr_amount')
# safe.sort()
# fraud.sort()
# relative size of fraud versus safe, assuming fraud is much smaller
# rel_size = len(safe)/len(fraud)
# oversample fraud to create equal size
# original_fraud = fraud # for finding back original results
# oversampled_fraud = []
# for i in range(math.floor(rel_size)):
#     oversampled_fraud.extend(fraud)
accuracies = []
n = 1
for i in range(n):
    train_files = []
    validate_files = []

    datasets = []
    labels = dpf.get_columns(dictlist, ['label'])
    dataset = dpf.get_columns(dictlist, ['xdr_amount'])
    datasets.append([item[0] for item in dataset])
    dataset2 = dpf.get_columns(dictlist, ['shopperinteraction', 'txvariantcode'])
    dataset2 = dpf.convert_to_categorical_option_true(dataset2, ['Ecommerce', 'visadebit'])
    # datasets.append(dataset2)
    dataset3 = dpf.get_columns(dictlist, ['shopperinteraction', 'txvariantcode'])
    dataset3 = dpf.convert_to_categorical_option_true(dataset3, ['Ecommerce', 'mccredit'])
    datasets.append(dpf.row_has_value(1,0, dataset2, dataset3))
    datasets.append(dpf.previous_fraud_counts(dictlist, 'ip_id'))

    # print(dataset)
    dataset = list(zip(*datasets))
    print(dataset)

    # for row in dictlist:
    #     labels.append(row.pop('label'))
    #     dataset.append(row)

    traindata, testdata, trainlabels, testlabels = train_test_split(dataset, labels, test_size = 0.4)
    traindata = numpy.array(traindata).astype(numpy.float64)
    trainlabels = numpy.array(trainlabels).astype(numpy.float64)
    # classifier_cost_to_false_accusation = 5000
    # cost_to_false_accusation = 50
    # average_fraud_cost = 0
    # fraud_count = 0
    # for trainrow, trainlabel in zip(traindata, trainlabels):
    #     if trainlabel==1:
    #         average_fraud_cost+=trainrow[0]
    #         fraud_count += 1
    # average_fraud_cost = average_fraud_cost / fraud_count
    # # print("average fraud {}".format(average_fraud_cost))
    # # fraud_fraction = fraud_count/len(traindata)
    # non_fraud_count = len(testdata)-fraud_count
    # upsampling_ratio = non_fraud_count/fraud_count
    # # print("upsampling_ratio {}".format(upsampling_ratio))
    # # # max_customer_complaint_cost = (len(traindata)-fraud_count)*cost_to_false_accusation
    # upsampled_avg_fraud_cost = average_fraud_cost/upsampling_ratio
    # #
    # print(cost_to_false_accusation)
    # print(upsampled_avg_fraud_cost)
    #
    classWeights = {0: 1,
                    1: 1}

    # traindata = traindata.astype(np.float64)
    # trainlabels = trainlabels.astype(np.float64)
    sm = SMOTE()
    traindata, trainlabels = sm.fit_resample(array(traindata), array(trainlabels))
    # oversampler = RandomOverSampler()
    # traindata, trainlabels = oversampler.fit_resample(array(traindata), array(trainlabels))
    # print('done with smote ')

    classifier = LogisticRegression(class_weight=classWeights)
    for epoch in range(1):
        zipped = list(zip(traindata, trainlabels))

        random.shuffle(zipped)

        traindata, trainlabels = zip(*zipped)
        classifier.fit(traindata, trainlabels)
    predictions = classifier.predict(testdata)
    # print(predictions)
    _,_,_,_,accuracy = dpf.get_cl_result(predictions, testlabels)
    accuracies.append(accuracy)
# print(dpf.get_fraud(predictions, testlabels, testdata, cost_to_false_accusation))
print("average accuracy over {} runs: {}".format(n, statistics.mean(accuracies)))