import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import data_processing_functions as dpf
from data_processing_functions import process_dict_reader

with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)
    # print(dpf.get_distinct_in_column(dictlist, 'currencycode'))
    # for line in dictlist:
    #     print(line)

dictlist = dpf.filter(dictlist, 'currencycode', 'AUD')
safe, fraud = dpf.split_label(dictlist)
safe = dpf.get_column(safe, 'xdr_amount')
fraud = dpf.get_column(fraud, 'xdr_amount')
safe.sort()
fraud.sort()
# relative size of fraud versus safe, assuming fraud is much smaller
rel_size = len(safe)/len(fraud)
# oversample fraud to create equal size
original_fraud = fraud # for finding back original results
oversampled_fraud = []
for i in range(math.floor(rel_size)):
    oversampled_fraud.extend(fraud)


bins = np.linspace(0, 60000, 100)

plt.hist(safe, bins, alpha=0.5, color='green', label='safe')
plt.hist(oversampled_fraud, bins, alpha=0.5, color='red', label='fraud')
plt.legend(loc='upper right')
plt.xlabel('xdr amount')
plt.ylabel('occurences')
plt.show()

# fraud_count = 0
# fraud_amount_caught = 0
# fraud_amount = 0
# safe_count = 0
# for amount in safe:
#     if amount>15000 and amount < 60000:
#         safe_count += 1
# for amount in fraud:
#     if amount>15000 and amount < 60000:
#         fraud_count += 1
#
# for amount in fraud:
#     if amount>15000 and amount < 60000:
#         fraud_amount_caught += amount
#
#     fraud_amount += amount
#
# # safe_annoyed =100
# cost_to_annoy =100
# # fraud_amount_caught = 10000
# # total = fraud_amount_caught - safe_annoyed*cost_to_annoy
#
# print("at a cost of {}, we have a fraud detection accuracy of {} and {} of the total fraudulent cash flagged resulting in a performance of {}".format(
#     safe_count/len(safe)*100,
#     fraud_count/len(fraud)*100,
#       fraud_amount_caught/fraud_amount*100,
#     (fraud_amount_caught-safe_count*cost_to_annoy)/fraud_amount*100))
# # a = np.random.random((16, 16))
# # plt.imshow(a, cmap='hot', interpolation='nearest')
# #
