# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import data_processing_functions1 as dpf
from data_processing_functions1 import process_dict_reader
import datetime
import random

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import RandomOverSampler
###############################################################################
#%% 1. Plot a histogram over the fraud and good transactions
with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)
    dictlist = process_dict_reader(reader)

safe1, fraud1 = dpf.split_label(dictlist)
safe = dpf.get_column(safe1, 'xdr_amount')
fraud = dpf.get_column(fraud1, 'xdr_amount')
safe.sort()
fraud.sort()

# relative size of fraud versus safe, assuming fraud is much smaller
rel_size = len(safe)/len(fraud)

# oversample fraud to create equal size
original_fraud = fraud # for finding back original results
oversampled_fraud = []
for i in range(math.floor(rel_size)):
    oversampled_fraud.extend(fraud)

# plot the histogrm of the oversampled fraud/non-fraud records
bins = np.linspace(0, 60000, 100)
plt.hist(safe, bins, alpha=0.5, color='green', label='safe')
plt.hist(oversampled_fraud, bins, alpha=0.5, color='red', label='fraud')
#plt.hist(original_fraud, bins, alpha=0.5, color='blue', label='fraud')
plt.legend(loc='upper right')
plt.xlabel('xdr amount')
plt.ylabel('occurences')
plt.show()
#%% 2. Use the histogram to build a simple classifier
fraud_count = 0   # predicted results
safe_count = 0

fraud_amount = 0  # real amount of frauded money 
fraud_amount_caught = 0  # predicted amount of frauded money

for amount in safe:
     if amount>15000 and amount < 60000:
         safe_count += 1
for amount in fraud:
     if amount>15000 and amount < 60000:
         fraud_count += 1
         fraud_amount_caught += amount
     fraud_amount += amount

 # safe_annoyed =100
cost_to_annoy =100
 # fraud_amount_caught = 10000
 # total = fraud_amount_caught - safe_annoyed*cost_to_annoy

print("at a cost of {}, we have a fraud detection accuracy of {} and {} of the total fraudulent cash flagged resulting in a performance of {}".format(
     safe_count/len(safe)*100,
     fraud_count/len(fraud)*100,
       fraud_amount_caught/fraud_amount*100,
     (fraud_amount_caught-safe_count*cost_to_annoy)/fraud_amount*100))

#################################################################################
#%% 3. Use sklearn kit to build classifiers on one feature only, xdr_amount
safe_set = dpf.get_column_set(safe1, 'xdr_amount')
fraud_set = dpf.get_column_set(fraud1, 'xdr_amount')

# safe set with label
safe_set = []
for line in safe:
    safe_set.append([int(line),0])
# fraud set with label
fraud_set = []
for line in fraud:
    fraud_set.append([int(line),1])
# oversampled fraud set with label
over_fraud_set = []
for line in oversampled_fraud:
    over_fraud_set.append([int(line),1])

# combine and randomize the dataset for classifiers   
wholeset = safe_set + fraud_set
over_wholeset = safe_set + over_fraud_set
random.shuffle(wholeset)
random.shuffle(over_wholeset)

# separate the label and feature sets
dataset = [[item[0]] for item in wholeset]
labelset = [item[1] for item in wholeset]
over_dataset = [[item[0]] for item in over_wholeset]
over_labelset = [item[1] for item in over_wholeset]

#%% Use LR classifier to test data with/without oversampling
# Test the original dataset
x_train, x_test, y_train, y_test = train_test_split(dataset, labelset, test_size = 0.2)
l_clf = LogisticRegression()
l_clf.fit(x_train,y_train)
l_prediction = l_clf.predict(x_test)
dpf.get_cl_result(l_prediction,y_test)

# Test the oversampled dataset
x_train, x_test, y_train, y_test = train_test_split(over_dataset, over_labelset, test_size = 0.2)
l_clf = LogisticRegression()
l_clf.fit(x_train,y_train)
l_prediction = l_clf.predict(x_test)
dpf.get_cl_result(l_prediction,y_test)



##################################################################################
#%% 4. Put all features/labels into a big datasets, basically pull from the example code in bright space
(issuercountry_set, txvariantcode_set, currencycode_set, shoppercountry_set, interaction_set,
verification_set, accountcode_set, mail_id_set, ip_id_set, card_id_set) = [set() for _ in range(10)]
src = 'data_for_student_case.csv'
ah = open(src, 'r')

data=[]     #All categorical+numerical features.
num_data=[] #only one numerical feature, amount.
label=[]
count_record = 0
ah.readline()#skip first line
for line_ah in ah:
    if line_ah.strip().split(',')[9]=='Refused':# remove the row with 'refused' label, since it's uncertain about fraud
        continue
    if 'na' in str(line_ah.strip().split(',')[14]).lower() or 'na' in str(line_ah.strip().split(',')[4].lower()):
        continue    
    count_record += 1
    # date reported flaud
    bookingdate = dpf.string_to_timestamp(line_ah.strip().split(',')[1])
    #country code
    issuercountry = line_ah.strip().split(',')[2]
    issuercountry_set.add(issuercountry)
    #type of card: visa/master
    txvariantcode = line_ah.strip().split(',')[3]
    txvariantcode_set.add(txvariantcode)
    #bin card issuer identifier
    issuer_id = float(line_ah.strip().split(',')[4])
    #transaction amount in minor units
    amount = float(line_ah.strip().split(',')[5])
    #the type of currentcy code
    currencycode = line_ah.strip().split(',')[6]
    currencycode_set.add(currencycode)
    #country code
    shoppercountry = line_ah.strip().split(',')[7]
    shoppercountry_set.add(shoppercountry)
    #online transaction or subscription
    interaction = line_ah.strip().split(',')[8]
    #define the fraud/legitible
    interaction_set.add(interaction)
    if line_ah.strip().split(',')[9] == 'Chargeback':
        label_temp = 1#label fraud
    else:
        label_temp = 0#label save
    #shopper provide CVC code or not    
    verification = line_ah.strip().split(',')[10]
    verification_set.add(verification)
    #0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
    cvcresponse = line_ah.strip().split(',')[11]
    if int(cvcresponse,10) > 2:
        cvcresponse = 3
    #Date of transaction     
    year_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%m/%d/%Y %H:%M').year
    month_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%m/%d/%Y %H:%M').month
    day_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%m/%d/%Y %H:%M').day
    creationdate = str(year_info)+'-'+str(month_info)+'-'+str(day_info)
    #Date of transaction-time stamp
    creationdate_stamp = dpf.string_to_timestamp(line_ah.strip().split(',')[12])
    #merchant’s webshop
    accountcode = line_ah.strip().split(',')[13]
    accountcode_set.add(accountcode)
    #mail
    mail_id = int(float(line_ah.strip().split(',')[14].replace('email','')))
    mail_id_set.add(mail_id)
    #ip
    ip_id = int(float(line_ah.strip().split(',')[15].replace('ip','')))
    ip_id_set.add(ip_id)
    #card
    card_id = int(float(line_ah.strip().split(',')[16].replace('card','')))
    card_id_set.add(card_id)
    
    data.append([issuercountry, txvariantcode, str(issuer_id), currencycode,
                shoppercountry, interaction, verification, cvcresponse, str(creationdate_stamp),
                 accountcode, str(mail_id), str(ip_id), str(card_id), creationdate, amount])# add the interested features here
    num_data.append([amount])
    label.append(label_temp)
print('the final num of records:', count_record)


#%% get a pure categorical feature dataset for classifier
data_np = np.asarray(data) # remove the last column of num element
cag_data = data_np[:,0:13]
cag_data = np.array(cag_data).tolist()

#%% encode the categorical features by onehorencoder
X = cag_data[0:500] # due to memory restriction, only select part to implement
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
data444 = enc.transform(X).toarray()
#%%









#
#
#
#x_train, x_test, y_train, y_test = train_test_split(data333, label, test_size = 0.2)
#clf = SVC(kernel='linear')
#clf.fit(x_train,y_train)
#clf_predictions = clf.predict(x_test)
#print("Accuracy: {}%".format(clf.score(x_test, y_test) * 100 ))


