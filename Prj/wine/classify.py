from IPython.core.display import display
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


data_train, data_test, label_train, label_test = train_test_split(raw_data['data'], raw_data['target'], test_size=0.2)
print(len(data_train), ' samples in training data\n', len(data_test), ' samples in test data\n', )







from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier

dict_classifiers = {
    "Logistic Regression":
        {'classifier': LogisticRegression(),
         'params': [
             {
                 'penalty': ['l1', 'l2'],
                 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
             }
         ]
         },
    "Nearest Neighbors":
        {'classifier': KNeighborsClassifier(),
         'params': [
             {
                 'n_neighbors': [1, 3, 5, 10],
                 'leaf_size': [3, 30]
             }
         ]
         },

    "Linear SVM":
        {'classifier': SVC(),
         'params': [
             {
                 'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['linear']
             }
         ]
         },
    "Gradient Boosting Classifier":
        {'classifier': GradientBoostingClassifier(),
         'params': [
             {
                 'learning_rate': [0.05, 0.1],
                 'n_estimators': [50, 100, 200],
                 'max_depth': [3, None]
             }
         ]
         },
    "Decision Tree":
        {'classifier': tree.DecisionTreeClassifier(),
         'params': [
             {
                 'max_depth': [3, None]
             }
         ]
         },
    "Random Forest":
        {'classifier': RandomForestClassifier(),
         'params': {}
         },
    "Naive Bayes":
        {'classifier': GaussianNB(),
         'params': {}
         }
}

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.6, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

num_classifiers = len(dict_classifiers.keys())

def  batch_classify(X_train, Y_train, X_test, Y_test, verbose = True):
    df_results = pd.DataFrame(
        data=np.zeros(shape=(num_classifiers,4)),
        columns=['classifier',
                 'train_score',
                 'test_score',
                 'training_time'])
    count = 0
    for key, classifier in dict_classifiers.items():
        t_start = time.clock()

        # GridSearchCV模块，能够在指定的范围内自动搜索具有不同超参数的不同模型组合，有效解放注意力。
        grid = GridSearchCV(classifier['classifier'],  # estimator: 所使用的分类器
                            classifier['params'],  # 字典或列表，即需要最优化的参数的取值
                            refit=True,
                            cv=10,  # cv 交叉验证参数: k折交叉验证。对于分类任务，使用StratifiedKFold（类别平衡，每类的训练集占比一样多
                            scoring='accuracy',
        # 准确评价标准，默认为None（使用estimator的误差估计函数），这时需要使用score函数；或者如scoring='roc_auc'，
                            # 根据所选模型不同，评价准则不同。
                            n_jobs=-1)
        estimator = grid.fit(X_train,  # grid.fit：运行网络搜索
                             Y_train)
        t_end = time.clock()
        t_diff = t_end - t_start
        train_score = estimator.score(X_train,
                                      Y_train)
        test_score = estimator.score(X_test,
                                     Y_test)
        df_results.loc[count, 'classifier'] = key
        df_results.loc[count, 'train_score'] = train_score
        df_results.loc[count, 'test_score'] = test_score
        df_results.loc[count, 'training_time'] = t_diff
        if verbose:
            print("trained {c} in {f: .2f} s".format(c=key, f=t_diff))
        count += 1
        plot_learning_curve(estimator,
                            "{}".format(key),
                            X_train,
                            Y_train,
                            ylim=(0.75, 1.0),
                            cv=10)
    return df_results


df_results = batch_classify(data_train, label_train, data_test, label_test)
display(df_results.sort_values(by='test_score', ascending=False))

