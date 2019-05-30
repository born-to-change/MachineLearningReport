from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
import time
from sklearn.metrics import accuracy_score


raw_data = datasets.load_wine()
#print(raw_data)


data_train, data_test, label_train, label_test = train_test_split(raw_data['data'], raw_data['target'], test_size=0.3)
print(len(data_train), ' samples in training data\n', len(data_test), ' samples in test data\n', )

# from sklearn.cluster import KMeans
# from sklearn import decomposition
# from sklearn import preprocessing
# from sklearn.mixture import GaussianMixture
#
# pca = decomposition.PCA(n_components=6)
# data_train = pca.fit_transform(data_train)
#
#
#
# acc_list=[]
# epoch_list=[]
# for epoch in np.arange(1, 300, 10):
#     model = KMeans(n_clusters=3, max_iter=epoch)
#     model.fit(data_train)
#     data_test = pca.fit_transform(data_test)
#     predict = model.predict(data_test)
#     # print(predict)
#     # print(label_test)
#
#     total = len(predict)
#     correct = 0
#     for i in np.arange(len(predict)):
#
#         if predict[i] == label_test[i]:
#             correct += 1
#     acc = correct / total
#     acc_list.append(acc)
#     epoch_list.append(epoch)
#     #print('acc:', )
#
# plt.figure()
# plt.title('epoch-acc')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# print(acc_list)
# plt.plot(epoch_list, acc_list)
# plt.show()












