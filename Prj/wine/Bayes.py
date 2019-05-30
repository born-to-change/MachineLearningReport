import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set_style('whitegrid')

from scipy.linalg import fractional_matrix_power


df = pd.read_csv('../../data/wine.data', names=['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                        'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                        'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline '])
print(df.head())

n_class1 = df['Class'][df['Class'] == 1].count()
n_class2 = df['Class'][df['Class'] == 2].count()
n_class3 = df['Class'][df['Class'] == 3].count()

total_class = df['Class'].count()
print('n_classq:{},n_class2:{},n_class3:{}'.format(n_class1, n_class2,n_class3))
P_class1 = n_class1 / total_class
P_class2 = n_class2 / total_class
P_class3 = n_class3 / total_class

print('Probabilitas prior')
print('Class 1 : ', P_class1)
print('Class 2 : ', P_class2)
print('Class 3 : ', P_class3)

meanX1 = df[df['Class'] == 1].groupby('Class', as_index=False).mean().drop('Class', axis=1).values
meanX2 = df[df['Class'] == 2].groupby('Class', as_index=False).mean().drop('Class', axis=1).values
meanX3 = df[df['Class'] == 3].groupby('Class', as_index=False).mean().drop('Class', axis=1).values
# drop(self, labels=None, axis=0, index=None, columns=None,level=None, inplace=False, errors='raise') 默认行维度
# axis=1表示指定丢弃列
X = df.drop('Class', axis=1).values
X1 = df[df['Class'] == 1].drop('Class', axis=1).values
X2 = df[df['Class'] == 2].drop('Class', axis=1).values
X3 = df[df['Class'] == 3].drop('Class', axis=1).values
Y=[]
for i in np.arange(178):
    if i<n_class1:
        Y.append(1)
    elif i>n_class1 and i<n_class2:
        Y.append(2)
    else:
        Y.append(3)

print(X.shape) # (178, 13)

print(len(Y))  # 178
#print(X1)  # (59, 13)

#决策树
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = \
    train_test_split(raw_data['data'],raw_data['target'],
                     test_size=0.2)
print(len(data_train),' samples in training data\n',
      len(data_test),' samples in test data\n', )



# globalMean = X.mean().values()
# meanCorrectedX1 = X1 = globalMean
# meanCorrectedX2 = X2 = globalMean
# meanCorrectedX3 = X3 = globalMean
#
# convMatX1 = ((meanCorrectedX1.T).dot(meanCorrectedX1)/len(X1))  # dot矩阵乘
# convMatX2 = ((meanCorrectedX2.T).dot(meanCorrectedX2)/len(X2))
# convMatX3 = ((meanCorrectedX3.T).dot(meanCorrectedX3)/len(X3))

# def likelihood(y, convMat, meanX):
#     pi = np.power(2*np.pi, 2)
#     detConvMat = np.sqrt(np.absolute(np.linalg.det(convMat)))  # 计算行列式
#     data_mean = y - meanX
#     exp = (-0.5*(data_mean.T)).dot(np.linalg.inv(convMat)).dot(data_mean)
#     result = ((1/math.pow(2*pi, 13/2) * detConvMat * np.exp(exp)))
#     return result
#
# def posterior(likelihood, prior):
#     return likelihood * prior


