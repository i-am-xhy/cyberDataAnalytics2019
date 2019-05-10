
import csv
import math
from data_processing_functions import process_dict_reader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE
import data_processing_functions as dpf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from numpy import array
import numpy
from sklearn.model_selection import train_test_split

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

train_files = []
validate_files = []


labels = dpf.get_columns(dictlist, ['label'])
dataset = dpf.get_columns(dictlist, ['xdr_amount'])
# for row in dictlist:
#     labels.append(row.pop('label'))
#     dataset.append(row)

traindata, testdata, trainlabels, testlabels = train_test_split(dataset, labels, test_size = 0.2)
# traindata = numpy.array(traindata).astype(numpy.float64)
# trainlabels = numpy.array(trainlabels).astype(numpy.float64)
# traindata = traindata.astype(np.float64)
# trainlabels = trainlabels.astype(np.float64)
# sm = SMOTE()
# traindata, trainlabels = sm.fit_resample(array(traindata), array(trainlabels))
# print('done with smote ')

# classifier = LogisticRegression()
# classifier.fit(traindata, trainlabels)
# predictions = classifier.predict(testdata)
print(testdata)
predictions = []
for row in testdata:
    xdrAmount = row[0]
    if xdrAmount>13000 and xdrAmount<60000:
        predictions.append(1)
    else:
        predictions.append(0)


print(predictions)
print(dpf.get_cl_result(predictions, testlabels))
print(dpf.get_fraud(predictions, testlabels, testdata))