# -*- coding: utf-8 -*-
from operator import itemgetter
from itertools import groupby
import time

def process_dict_reader(dictreader):
    dictlist = remove_refused(dictreader)
    dictlist = clean_formatting(dictlist)
    dictlist = add_imf_currency_conversion(dictlist)
    return dictlist


def remove_refused(dictlist):
    result = []
    count = 0
    for line in dictlist:
        if line['simple_journal'] == 'Refused':
            count+=1
            continue
        if line['bin'] == 'na' or line['mail_id'] == 'na':
            count+=1
            continue
        result.append(line)
    print('deleted items:', count)
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

def get_cl_result(prediction, y_test):
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(prediction)):
        if y_test[i]==1 and prediction[i]==1:
            TP += 1
        if y_test[i]==0 and prediction[i]==1:
            FP += 1
        if y_test[i]==1 and prediction[i]==0:
            FN += 1
        if y_test[i]==0 and prediction[i]==0:
            TN += 1
    print ('TP: '+ str(TP))
    print ('FP: '+ str(FP))
    print ('FN: '+ str(FN))
    print ('TN: '+ str(TN))
    return TP,FP,FN, TN








#%% for the example code in brightspace
def aggregate(before_aggregate, aggregate_feature): #
    if aggregate_feature == 'day':
        after_aggregate = []
        before_aggregate.sort(key = itemgetter(9))#sort by timestamp
        temp = groupby(before_aggregate, itemgetter(9))
        print(before_aggregate)
        group_unit = []
        mean = []
        for i, item in temp:# i is group id
            for jtem in item:# unit in each group
                group_unit.append(jtem)
            #for feature_i in xrange(6):
            #    mean.append(zip(group_unit)[feature_i])
            #after_aggregate.append(group_unit)
            after_aggregate.append(mean)
            group_unit = []
        #print after_aggregate[0]
        #print before_aggregate[0]
    if aggregate_feature == 'client':
        after_aggregate = []
        pos_client = -3
        before_aggregate.sort(key = itemgetter(pos_client))#sort with cardID firstly，if sort with 2 feature, itemgetter(num1,num2)
        temp = groupby(before_aggregate, itemgetter(pos_client))#group
        group_unit = []
        for i, item in temp:# i is group id
            for jtem in item:# unit in each group
                group_unit.append(jtem)
            after_aggregate.append(group_unit)
            group_unit = []
    return after_aggregate

def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string, '%m/%d/%Y %H:%M')
    return time.mktime(time_stamp)

