import numpy
import datetime


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
    print ('TP: '+ str(TP))
    print ('FP: '+ str(FP))
    print ('FN: '+ str(FN))
    print ('TN: '+ str(TN))
    return TP,FP,FN, TN

def get_fraud(predictions, trueLabels, testdata):
    cost_to_false_accusation = 50
    total_cost = 0
    total_possible_fraud = 0
    for prediction, trueLabel, row in zip(predictions, trueLabels, testdata):
        prediction = int(prediction)
        trueLabel = trueLabel[0]
        amount = row[0]
        # if trueLabel == 1 and prediction == 1:
        #     TP += 1
        if trueLabel == 0 and prediction == 1:
            total_cost += cost_to_false_accusation
        if trueLabel == 1 and prediction == 0:
            total_cost += amount
        if trueLabel == 1:
            total_possible_fraud += amount
        # if trueLabel == 0 and prediction == 0:
        #     TN += 1
    print("total cost of operation {} total amount of fraud catchable {} for a fraud reduction of {}".format(total_cost, total_possible_fraud, max((total_possible_fraud-total_cost) / total_possible_fraud, 0)))
    return max((total_possible_fraud-total_cost) / total_possible_fraud, 0)
