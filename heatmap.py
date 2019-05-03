import matplotlib.pyplot as plt
import numpy as np
import csv
import data_processing_functions as dpf
from data_processing_functions import process_dict_reader

with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)
    # print(dpf.get_distinct_in_column(dictlist, 'currencycode'))
    # for line in dictlist:
    #     print(line)

safe, fraud = dpf.split_label(dictlist)
plt.plot(dpf.get_column(safe, 'xdr_amount'), alpha=0.5, color='green', label='safe')
plt.plot(dpf.get_column(fraud, 'xdr_amount'), alpha=0.5, color='red', label='fraud')
plt.legend(loc='upper right')
print(dpf.get_column(fraud, 'xdr_amount'))
# a = np.random.random((16, 16))
# plt.imshow(a, cmap='hot', interpolation='nearest')
#
plt.show()