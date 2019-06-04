from sklearn.decomposition import PCA
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# with open('data/BATADAL_dataset03.csv', 'r') as training_data_file:
#csv.DictReader(training_data_file)
training_data = pd.read_csv('data/BATADAL_dataset03.csv')
training_data.drop('DATETIME', inplace=True, axis='columns') # DATETIME isnt immediatly useable. so ignore for now
# with open('data/BATADAL_test_dataset', 'r') as test_data_file:
#     test_data = csv.DictReader(test_data_file)

def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss

# def scatterPlot(xDF, yDF, algoName):
#     tempDF = pd.DataFrame(data=xDF.loc[:,0:1], index=xDF.index)
#     tempDF = pd.concat((tempDF,yDF), axis=1, join="inner")
#     tempDF.columns = ["First Vector", "Second Vector", "Label"]
#     sns.lmplot(x="First Vector", y="Second Vector", hue="Label", \
#                data=tempDF, fit_reg=False)
#     ax = plt.gca()
#     ax.set_title("Separation of Observations using "+algoName)

# def plotResults(trueLabels, anomalyScores, returnPreds = False):
#     preds = pd.concat([trueLabels, anomalyScores], axis=1)
#     preds.columns = ['trueLabel', 'anomalyScore']
#     precision, recall, thresholds = \
#         precision_recall_curve(preds['trueLabel'],preds['anomalyScore'])
#     average_precision = \
#         average_precision_score(preds['trueLabel'],preds['anomalyScore'])
#
#     plt.step(recall, precision, color='k', alpha=0.7, where='post')
#     plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')
#
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#
#     plt.title('Precision-Recall curve: Average Precision = \
#     {0:0.2f}'.format(average_precision))
#
#     fpr, tpr, thresholds = roc_curve(preds['trueLabel'], \
#                                      preds['anomalyScore'])
#     areaUnderROC = auc(fpr, tpr)
#
#     plt.figure()
#     plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
#     plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic: \
#     Area under the curve = {0:0.2f}'.format(areaUnderROC))
#     plt.legend(loc="lower right")
#     plt.show()
#
#     if returnPreds==True:
#         return preds

def chi_squared(pca):
    means = pca.mean(axis=0, skipna = True)
    # print(means[1])
    print(means)
    # print(len(pca[abs(pca[0]) < 0.01]))
    result = pd.DataFrame(columns=['chi_squared'])
    result['chi_squared'] = [0.0 for _ in range(len(pca))]
    # print(result)

    for columnIndex, column in pca.iteritems():
        for rowIndex, cell in column.iteritems():
            # if rowIndex not in result['chi_squared']:
            #     result['chi_squared'][rowIndex] = 0
            # print(columnIndex)
            # print(rowIndex)
            # print(cell)

            #todo fix chi squared
            # result['chi_squared'][rowIndex] += cell**2/means[columnIndex]
            result['chi_squared'][rowIndex] += cell**2/means[columnIndex]

    return result


def chi_squared_to_inlier_outlier(chi_squared_pca, threshold):
    inliers = chi_squared_pca[chi_squared_pca['chi_squared'] <= threshold]
    outliers = chi_squared_pca[chi_squared_pca['chi_squared'] > threshold]
    return inliers, outliers


def plot_dataframe_singals(dataframe):
    for columnIndex, column in dataframe.iteritems():
        plt.plot(column, label=columnIndex)
    plt.legend(loc='best')
    plt.show()

def normalize(dataframe):
    x = dataframe.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler((0, 1))
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=dataframe.columns)

n_components = 0.95
random_state = 2018
threshold = 10.3

pca = PCA(n_components=n_components, random_state=random_state)

training_data = normalize(training_data)
# min_max_scaler = preprocessing.MinMaxScaler((0,1))
# x_scaled = min_max_scaler.fit_transform(x)
# training_data = pd.DataFrame(x_scaled, columns=training_data.columns)
# print(training_data)

training_data_PCA = pca.fit_transform(training_data)
training_data_PCA = pd.DataFrame(data=training_data_PCA, index=training_data.index)
training_data_PCA = normalize(training_data_PCA)
# print(training_data_PCA)
# plot_PCA(training_data_PCA)
chi_squared_pca = chi_squared(training_data_PCA)
# print(chi_squared_pca)

inliers, outliers = chi_squared_to_inlier_outlier(chi_squared_pca, threshold)
print(training_data_PCA.ix[outliers.index])
print(training_data.columns)
data_to_visualize = training_data.ix[outliers.index]
data_to_visualize = data_to_visualize[['L_T2', 'L_T3', 'L_T5', 'F_PU2', 'S_PU2','P_J269', 'F_PU3', 'F_PU7', 'S_PU4', 'F_PU4']]

plot_dataframe_singals(data_to_visualize)
print("amount of components needed to reach a maintained variance of {}% is {}".format(n_components*100, len(training_data_PCA.columns)))
print("the fraction of false outliers is: {}%".format(len(outliers)/(len(inliers)+len(outliers))*100))



# training_data_PCA_inverse = pca.inverse_transform(training_data_PCA)
# training_data_PCA_inverse = pd.DataFrame(data=training_data_PCA_inverse, index=training_data.index)
#
# # scatterPlot(training_data_PCA, y_train, "PCA")
#
# anomalyScoresPCA = anomalyScores(training_data, training_data_PCA_inverse)
# print(anomalyScoresPCA)
# preds = plotResults(training_data, anomalyScoresPCA, True)