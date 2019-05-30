import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio
import numpy as np




def plot_n_image(X, n):

    pic_szie = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size, sharey=True, sharex=True, figsize=(8,8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_szie, pic_szie)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


def covariance_matrix(X):
    m = X.shape[0]
    return np.dot(X.T, X)/m

def normalize(X):
    X_copy = X.copy()
    m, n = X_copy.shape

    for col in range(n):
        X_copy[:, col] = (X_copy[:, col] - X_copy[:, col].mean()) / X_copy[:, col].std()
    return X_copy

def pca(X):

    X_norm = normalize(X)

    Sigma  = covariance_matrix(X_norm)  # (2,2)

    U,S,V = np.linalg.svd(Sigma)

    return U,S,V

def project_data(X,U,k):
    m, n = X.shape
    if k>n:
        raise ValueError('k should be lower dimension of n')
    return np.dot(X,U[:, :k])

def recover_data(Z, U):
    m, n = Z.shape   # (50,1) (2,2)

    if n >= U.shape[0]:
        raise ValueError('Z dimension is >= U, you should recover from lower dimension to higher')
    return np.dot(Z, U[:, :n].T)



mat = sio.loadmat('../data/ex7data1.mat')
print(mat.keys())

# data1 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
# #print(data1)
#
# sns.set(context='notebook', style='white')
# sns.lmplot('X1', 'X2', data=data1, fit_reg=False)
# plt.show()
X = mat.get('X')
X_norm = normalize(X) # (50,2)
# # print(X)
# # print(X_norm)
#
# sns.lmplot('X1', 'X2', data=pd.DataFrame(X_norm, columns=['X1', 'X2']), fit_reg=False)
#
# plt.show()

U,S,V = pca(X_norm)  # S:对角线特征值
print(covariance_matrix(X_norm))
print(U)  # (2, 2) (2,) (2, 2)
print(S)

Z = project_data(X_norm, U, 1)
print(Z.shape)  # (50, 1)
print(Z[:10])
X_recover = recover_data(Z, U)  # (50,1) (2,2)

U, S, V = np.linalg.svd(np.array([[1,2],[0,0],[0,0]]))
print(S) #  [2.23606798,  0.]
#
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
# sns.regplot('X1', 'X2', data=pd.DataFrame(X_norm, columns=['X1', 'X2']), fit_reg=False, ax=ax1)
#
# ax1.set_title('Original dimension')
#
# sns.rugplot(Z, ax=ax2)
# ax2.set_xlabel('Z')
# ax2.set_title('Z dimension')
#plt.show()

# SKlearn PCA

from sklearn.decomposition import PCA


mat = sio.loadmat('../data/ex7faces.mat')
X = np.array([x.reshape((32, 32)).T.reshape(1024) for x in mat.get('X')])
plot_n_image(X, 10)

print(X.shape)  # (5000, 1024)
sk_pca = PCA(n_components=100)
Z = sk_pca.fit_transform(X)
print(Z.shape)    #  (5000, 100)
plot_n_image(Z, 64)

plt.show()
