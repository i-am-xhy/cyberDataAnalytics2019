import matplotlib.pyplot as plt
import csv
from itertools import groupby
import data_processing_functions as dpf
from data_processing_functions import process_dict_reader

with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)


# plot the card id repeat fraud
safe, fraud = dpf.split_label(dictlist)
fraud = dpf.get_column(fraud, 'card_id')

repeatFraudCounts = [len(list(group)) for key, group in groupby(fraud)]
plt.hist(repeatFraudCounts)
plt.show()

