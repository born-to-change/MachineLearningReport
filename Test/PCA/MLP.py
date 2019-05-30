from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

raw_data = datasets.load_wine()
#print(raw_data)
data_train, data_test, label_train, label_test = train_test_split(raw_data['data'], raw_data['target'], test_size=0.3)
print(len(data_train), ' samples in training data\n', len(data_test), ' samples in test data\n', )
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_optput):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_optput)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.out(x)
        return x

net = Net(n_feature=13, n_hidden=40, n_optput=3)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
loss_func = torch.nn.CrossEntropyLoss()
results = {}
for epoch in range(1000):
    inputs = torch.Tensor(data_train).type(torch.FloatTensor)
    #inputs = torch.from_numpy(data_train)
    label = torch.Tensor(label_train).type(torch.LongTensor)
    out = net(inputs)   # (124, 3)
    #out = torch.argmax(out, dim=1)  # (124)
    loss = loss_func(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    data_test = torch.Tensor(data_test).type(torch.FloatTensor)
    predict = net(data_test)
    if epoch % 10 == 0:
        correct = 0
        total = len(predict)
        for i in np.arange(total):
            predicts = torch.argmax(predict, dim=1).numpy()

            if predicts[i] == label_test[i]:
                correct += 1
        results[epoch] = correct / total



plt.figure()
plt.title('epoch-acc')
plt.xlabel('epoch')
plt.ylabel('acc')
print(results)
plt.plot(results.keys(), results.values())
plt.show()
