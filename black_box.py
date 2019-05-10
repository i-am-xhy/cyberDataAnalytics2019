
import csv
import numpy
from data_processing_functions import process_dict_reader
from imblearn.over_sampling import SMOTE
import data_processing_functions as dpf
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


# get the data from the csv
with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)

# build the features used for training/predicting
datasets = []
labels = dpf.get_columns(dictlist, ['label'])
dataset = dpf.get_columns(dictlist, ['xdr_amount'])
datasets.append([item[0] for item in dataset])
dataset2 = dpf.get_columns(dictlist, ['shopperinteraction', 'txvariantcode'])
dataset2 = dpf.convert_to_categorical_option_true(dataset2, ['Ecommerce', 'visadebit'])
dataset3 = dpf.get_columns(dictlist, ['shopperinteraction', 'txvariantcode'])
dataset3 = dpf.convert_to_categorical_option_true(dataset3, ['Ecommerce', 'mccredit'])
datasets.append(dpf.row_has_value(1,0, dataset2, dataset3))
datasets.append(dpf.previous_fraud_counts(dictlist, 'card_id'))
dataset = list(zip(*datasets))

# split the dataset into training and test data
traindata, testdata, trainlabels, testlabels = train_test_split(dataset, labels, test_size = 0.5)
traindata = numpy.array(traindata).astype(numpy.float64)
trainlabels = numpy.array(trainlabels).astype(numpy.float64)

# apply smote
sm = SMOTE()
traindata, trainlabels = sm.fit_resample(array(traindata), array(trainlabels))

# define the classifier, tell it to crossfold 10 times
classifier = LogisticRegressionCV(cv=10)
# fit the training data
classifier.fit(traindata, trainlabels)

# do prediction on the testdata
predictions = classifier.predict(testdata)
_,_,_,_,accuracy = dpf.get_cl_result(predictions, testlabels)

print("accuracy on test data {}".format(accuracy))