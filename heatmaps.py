import matplotlib.pyplot as plt
import numpy as np
import csv
import data_processing_functions as dpf
from data_processing_functions import process_dict_reader

# extract data from csv
with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    dictlist = process_dict_reader(reader)

#fraud counts function
def get_pressurized_annotated_categorical_heatmap(dictlist, column1, column2, xlabel='', ylabel='', title=''):
    fig, ax = plt.subplots()
    combinatory_counts, x_labels, y_labels = dpf.get_combinatory_pressure(dictlist, column1, column2)

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

    fig.tight_layout()
    plt.show()

#fraud pressure function
def get_annotated_categorical_heatmap(dictlist, column1, column2, xlabel='', ylabel='', title=''):
    fig, ax = plt.subplots()
    combinatory_counts, x_labels, y_labels = dpf.get_combinatory_counts(dictlist, column1, column2)

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

    fig.tight_layout()
    plt.show()

# day month fraud visualization
column1, column2 = 'day', 'month'
get_annotated_categorical_heatmap(dictlist, column1, column2, 'day', 'month', 'Fraud by day and month')


# fraud pressure visualization
column1, column2 = 'txvariantcode', 'shopperinteraction'
get_pressurized_annotated_categorical_heatmap(dictlist, column1, column2, 'Card type', 'Shopper interaction type', 'Fraud pressure by card issuer by shopper interaction type')
