
import csv
import statistics
from data_processing_functions import process_dict_reader
import data_processing_functions as dpf
from sklearn.model_selection import train_test_split

# get data from csv
with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)

#get labels and features
labels = dpf.get_columns(dictlist, ['label'])
dataset = dpf.get_columns(dictlist, ['xdr_amount'])

#apply randomized folds
accuracies = []
n = 10
for i in range(n):
    #split dataset into training and testdata
    traindata, testdata, trainlabels, testlabels = train_test_split(dataset, labels, test_size = 0.2)

    # do prediction
    predictions = []
    for row in testdata:
        xdrAmount = row[0]
        if xdrAmount>13000 and xdrAmount<60000:
            predictions.append(1)
        else:
            predictions.append(0)

    #store the accuracy for this fold
    _,_,_,_,accuracy = dpf.get_cl_result(predictions, testlabels)
    accuracies.append(accuracy)

# print the accuracy
print("average accuracy over {} runs: {}".format(n, statistics.mean(accuracies)))