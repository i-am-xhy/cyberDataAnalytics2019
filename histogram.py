import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import data_processing_functions as dpf
from data_processing_functions import process_dict_reader

with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)


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

# show the histogram visualization
bins = np.linspace(0, 50000, 50)
plt.hist(safe, bins, alpha=0.5, color='green', label='safe')
plt.hist(oversampled_fraud, bins, alpha=0.5, color='red', label='fraud')
plt.legend(loc='upper right')
plt.title('Transaction type counts per amount of IMF special drawing rights (XDR)')
plt.xlabel('XDR amount')
plt.ylabel('Occurences')
plt.show()