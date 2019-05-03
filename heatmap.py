import matplotlib.pyplot as plt
import numpy as np
import csv

with open('data_for_student_case.csv') as file:
    reader = csv.DictReader(file)

    for line in reader:
        print(line)

a = np.random.random((16, 16))
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()