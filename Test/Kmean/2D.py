import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('../data/ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
print(data2.shape)  # (300,2)

def combine_data_C(data, C):
    data_with_c  = data.copy()
    data_with_c['C'] = C
    return data_with_c

def random_init(data, k):
    # 随机采3个样本
    return data.sample(k).as_matrix()  # (3,2)

def _find_your_cluster(x, centroids):
    """find the right cluster for x with respect to shortest distance
       Args:
           x: ndarray (n, ) -> n features
           centroids: ndarray (k, n)
       Returns:
           k: int
       """

    #  Apply a function to 1-D slices along the given axis.
    distance = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=centroids-x)  # 对每一行元素的（centroids-x）求矩阵2范数
    return np.argmin(distance)

def assign_cluster(data, centroids):
    """assign cluster for each node in data
       return C ndarray
       """
    return np.apply_along_axis(lambda x: _find_your_cluster(x, centroids),
                               axis=1, arr=data.as_matrix())

def new_centroids(data, C):
    data_with_c = combine_data_C(data, C)
    return data_with_c.groupby('C', as_index=False).mean().sort_values(by='C').drop('C', axis=1).as_matrix()

def cost(data, centroids, C):
    m = data.shape[0]
    expand_C_with_centroids = centroids[C]
    distances = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=data.as_matrix()-expand_C_with_centroids)

    return distances.sum()/m

def _k_means_iter(data, k, epoch=100, tol=0.0001):

    centroids = random_init(data, k)   # 三个聚类中心(3,2)
    cost_progress = []

    for i in range(epoch):
        print('running epoch{}'.format(i))

        C = assign_cluster(data, centroids)
        centroids = new_centroids(data, C)
        cost_progress.append(cost(data, centroids, C))

        if len(cost_progress) > 1:
            if (np.abs(cost_progress[-1] - cost_progress[-2])) / cost_progress[-1] < tol:
                break

    return C, centroids, cost_progress[-1]

def k_means(data, k, epoch=100, n_init=10):

    tries = np.array([_k_means_iter(data, k, epoch)] for _ in range(n_init))
    least_cost_idx = np.argmin(tries[:, -1])
    return tries[least_cost_idx]

init_centroids = random_init(data2, 3)
print(init_centroids.shape)

x = np.array([1,1])

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x=init_centroids[:, 0], y=init_centroids[:, 1])

for i, node in enumerate(init_centroids):
    ax.annotate('{}:({},{})'.format(i, node[0], node[1]), node)

# ax.scatter(x[0], x[1], marker='x', s=200)
# plt.show()
final_C, findal_centroid, _  = _k_means_iter(data2, 3)
data_with_c = combine_data_C(data2, final_C)

from sklearn.cluster import KMeans
import seaborn as sns
sk_kmeans = KMeans(n_clusters=3)
sk_kmeans.fit(data2)

sk_C = sk_kmeans.predict(data2)
data_with_c = combine_data_C(data2, sk_C)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)

plt.show()


# x = np.array([
#     [0, 3, 4],
#     [1, 6, 4]])
# # 默认参数ord=None，axis=None，keepdims=False
# print("默认参数(矩阵整体元素平方和开根号，不保留矩阵二维特性)：", np.linalg.norm(x))
# print("矩阵整体元素平方和开根号，保留矩阵二维特性：", np.linalg.norm(x, keepdims=True))
#
# print("矩阵每个行向量求向量的2范数：", np.linalg.norm(x, axis=1, keepdims=True))
# print("矩阵每个列向量求向量的2范数：", np.linalg.norm(x, axis=0, keepdims=True))
#
# print("矩阵1范数：", np.linalg.norm(x, ord=1, keepdims=True))   # 列模和 最大的
# print("矩阵2范数：", np.linalg.norm(x, ord=2, keepdims=True))
# print("矩阵∞范数：", np.linalg.norm(x, ord=np.inf, keepdims=True))   # 行模和 最大的
#
# print("矩阵每个行向量求向量的1范数：", np.linalg.norm(x, ord=1, axis=1, keepdims=True))