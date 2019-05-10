import numpy
import datetime
from matplotlib import pyplot
from sklearn.metrics import roc_curve, roc_auc_score

def process_dict_reader(dictreader):
    dictlist = remove_refused(dictreader)
    dictlist = clean_formatting(dictlist)
    dictlist = add_imf_currency_conversion(dictlist)
    return dictlist


def remove_refused(dictlist):
    result = []
    for line in dictlist:
        if line['simple_journal'] == 'refused':
            continue
        if line['bin'] == 'na' or line['mail_id'] == 'na':
            continue

        result.append(line)

    return result


def clean_formatting(dictlist):
    for line in dictlist:
        # chargebacks can be assumed to be fraud, whereas the rest should be safe.
        if line['simple_journal'] == 'Chargeback':
            line['label'] = 1 # fraud
        else:
            line['label'] = 0 # safe

        # 0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
        line['cvcresponsecode'] = int(line['cvcresponsecode'])
        if line['cvcresponsecode'] > 2:
            line['cvcresponsecode'] = 3

        date = datetime.datetime.strptime(line['creationdate'], '%Y-%m-%d %H:%M:%S')
        line['year'] = date.year
        line['month'] = date.month
        line['day'] = date.day

    return dictlist
# how many of that currency is worth 1 XDR or imf special drawing right.
xdr_conversion_rates = {
    'SEK': 13.2510,
    'NZD': 2.09126,
    'GBP': 1.97804,
    'MXN': 26.4705,
    'AUD': 1.97804

}

def filter(dictlist, column, value):
    # returns all dictlist entries with column==value
    result = []
    for line in dictlist:
        if line[column] == value:
            result.append(line)
    return result

def add_imf_currency_conversion(dictlist):
    # adds an additional currency field in which the base currency is transferred to a uniform currency
    # uses imf special drawing rights to be more stable, might need to be expanded to take into account relative purchase power over time
    for line in dictlist:
        currency = line['currencycode']
        amount = float(line['amount'])

        line['xdr_amount'] = amount/xdr_conversion_rates[currency]
    return dictlist

def get_distinct_in_column(dictlist, column_name):
    # prints all distinct values in a given column
    result = set()
    for line in dictlist:
        result.add(line[column_name])
    return result

def split_label(dictlist):
    safe_result = []
    fraud_result = []
    for line in dictlist:
        if line['label'] == 0:
            safe_result.append(line)
        else:
            fraud_result.append(line)
    return safe_result, fraud_result

def get_column(dictlist, column_name):
    # returns a column as a list
    result = []
    for line in dictlist:
        result.append(line[column_name])
    return result

def get_columns(dictlist, column_names):
    # returns a column as a list
    result = []
    for line in dictlist:
        row_result = []
        for column_name in column_names:
            row_result.append(line[column_name])
        result.append(row_result)
    return result

def convert_to_categorical_option_true(columns, columnValues):
    result = []
    for row in columns:
        matchesColumnValues = 1
        for item, columnValue in zip(row,columnValues):
            if item!=columnValue:
                matchesColumnValues = 0
                break
        result.append(matchesColumnValues)
    return result

def row_has_value(value, falseValue, *args):
    result = []
    dataset = list(zip(*args))

    for row in dataset:
        rowHasTrue = falseValue
        for element in row:
            if element == value:
                rowHasTrue=value
                break
        result.append(rowHasTrue)
    return result

def previous_fraud_counts(dictlist, column):
    # for unique items in column counts the amount of preceding fraud counts
    # assumes ordering is chronological
    result = []
    # print(dictlist)
    previousCounts = {}
    for row in dictlist:
        value = row[column]
        if value not in previousCounts:
            previousCounts[value] = 0

        result.append(previousCounts[value])

        if row['label'] == 1:
            previousCounts[value] += 1

    return result


def get_combinatory_counts(dictlist, column1, column2, countColumn='label', countValue=1):
    # counts for all unique pairs of values in column1 and column2 the amount of times that countvalue is encountered in the countcolumn

    filtered_dictlist = filter(dictlist, countColumn, countValue)
    distinct1 = list(get_distinct_in_column(filtered_dictlist, column1))
    distinct2 = list(get_distinct_in_column(filtered_dictlist, column2))

    result = numpy.zeros((len(distinct2), len(distinct1)))

    for line in filtered_dictlist:
        x_index = distinct1.index(line[column1])
        y_index = distinct2.index(line[column2])

        result[y_index, x_index] += 1.0
    return result, distinct1, distinct2

def get_combinatory_pressure(dictlist, column1, column2):
    # counts for all unique pairs of values in column1 and column2 the amount of times that countvalue is encountered in the countcolumn

    filtered_dictlist = filter(dictlist, 'label', 1)
    distinct1 = list(get_distinct_in_column(filtered_dictlist, column1))
    distinct2 = list(get_distinct_in_column(filtered_dictlist, column2))

    result = numpy.zeros((len(distinct2), len(distinct1)))

    for line in dictlist:
        try:
            x_index = distinct1.index(line[column1])
            y_index = distinct2.index(line[column2])
        except:
            pass # out of range error, which is fine
        if line['label'] != 1:
            result[y_index, x_index] -= 1.0
            continue

        result[y_index, x_index] += 1.0
    return result, distinct1, distinct2

# def dict_of_dicts_to_matrix(dict_of_dicts):
#     result = []
#     for key, dicts in dict_of_dicts.items():
#         row_result = []
#         for other_key, value in dicts.items():
#             row_result.append(value)
#         result.append(row_result)
#     return result
def get_cl_result(predictions, trueLabels):
    TP, FP, FN, TN = 0, 0, 0, 0
    for prediction, trueLabel in zip(predictions, trueLabels):
        prediction = int(prediction)
        trueLabel = trueLabel[0]
        if trueLabel == 1 and prediction == 1:
            TP += 1
        if trueLabel == 0 and prediction == 1:
            FP += 1
        if trueLabel == 1 and prediction == 0:
            FN += 1
        if trueLabel == 0 and prediction == 0:
            TN += 1
    total_fraud = TP + FN
    total_non_fraud = TN + FP
    fraud_accuracy = (total_fraud-FN)/total_fraud
    non_fraud_accuracy = (total_non_fraud-FP)/total_non_fraud
    print ('TP: '+ str(TP))
    print ('FP: '+ str(FP))
    print ('FN: '+ str(FN))
    print ('TN: '+ str(TN))
    print('Accuracy: {} with fraud accuracy: {} and non-fraud accuracy: {}'.format((fraud_accuracy+non_fraud_accuracy)/2, fraud_accuracy, non_fraud_accuracy))
    return TP,FP,FN, TN, (fraud_accuracy+non_fraud_accuracy)/2


def get_fraud(predictions, trueLabels, testdata, cost_to_false_accusation):

    total_cost = 0
    total_possible_fraud = 0
    customer_complaints = 0
    fraud_missed = 0
    for prediction, trueLabel, row in zip(predictions, trueLabels, testdata):
        prediction = int(prediction)
        trueLabel = trueLabel[0]
        amount = row[0]
        # if trueLabel == 1 and prediction == 1:
        #     TP += 1
        if trueLabel == 0 and prediction == 1:
            total_cost += cost_to_false_accusation
            customer_complaints += 1
        if trueLabel == 1 and prediction == 0:
            total_cost += amount
            fraud_missed += amount
        if trueLabel == 1:
            total_possible_fraud += amount
        # if trueLabel == 0 and prediction == 0:
        #     TN += 1
    print("total cost of operation {} total amount of fraud catchable {} for a fraud reduction of {}".format(total_cost, total_possible_fraud, max((total_possible_fraud-total_cost) / total_possible_fraud, 0)))
    print("total cost of operation is constructed out of {} customers being falsely accused and {} XDR in fraud missed".format(customer_complaints, fraud_missed))
    return max((total_possible_fraud-total_cost) / total_possible_fraud, 0)

def get_column_set(dictlist, column_name):
    # returns a column as a list
    result = []
    for line in dictlist:
        result.append([line[column_name]])
    return result

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
