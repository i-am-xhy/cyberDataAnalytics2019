import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import data_processing_functions as dpf
from data_processing_functions import process_dict_reader

with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)

# a = np.random.random((16, 16))
# for line in dictlist:
#     print(line)
# filtered_dictlist = dpf.filter(dictlist, 'label', 1)
# for line in dictlist:
#     print(line)
# print(filtered_dictlist)
# based on https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

def get_annotated_categorical_heatmap(dictlist, column1, column2, xlabel='', ylabel='', title='', labelValue=1):
    fig, ax = plt.subplots()
    combinatory_counts, x_labels, y_labels = dpf.get_combinatory_counts(dictlist, column1, column2, countValue=labelValue)
    # combinatory_counts = dpf.dict_of_dicts_to_matrix(combinatory_counts)
    # print(combinatory_counts)
    plt.imshow(combinatory_counts, cmap='hot', interpolation='nearest')

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            if combinatory_counts[j, i] != 0:
                text = ax.text(i, j, int(combinatory_counts[j, i]),
                           ha="center", va="center", color="b")
    # plt.set_title("")
    fig.tight_layout()
    plt.show()

# column1, column2 = 'day', 'month'
# get_annotated_categorical_heatmap(dictlist, column1, column2, 'day', 'month', 'Fraud by day and month')


# column1, column2 = 'bin', 'shopperinteraction'
# get_annotated_categorical_heatmap(dictlist, column1, column2, 'Card issuer', 'Shopper interaction type', 'Fraud by card issuer by shopper interaction type')
# # get_annotated_categorical_heatmap(dictlist, column1, column2, countvalue=0)

column1, column2 = 'mail_id', 'card_id'
get_annotated_categorical_heatmap(dictlist, column1, column2, 'Card type', column1, column2)
# get_annotated_categorical_heatmap(dictlist, column1, column2, 'Card type', 'Shopper interaction type', 'Fraud pressure by card issuer by shopper interaction type', labelValue=0)
# get_annotated_categorical_heatmap(dictlist, column1, column2, countvalue=0)