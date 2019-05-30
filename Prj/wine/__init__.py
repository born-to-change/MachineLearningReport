import numpy as np

a = np.arange(12).reshape(1,3,4)
b = np.ones((1,3,4))

c = np.concatenate((a, b), axis=1)
print('a:', a)
print('b:', b)
print('c:', c)