import pandas as pd
import numpy as np

# dates = pd.date_range('20130101', periods=6)
# print(dates)
#
# df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
# print(df)
# print(df.drop('A', axis=1))
#
# s = pd.Series(np.random.randint(0,7,size=10))
# print(s)


b=np.arange(6).reshape(2,3)
copy_b=b.copy()
a=np.arange(6).reshape(3,2)
print('a:')
print(a)
print('b:')
print(b)
print('copy_b*b:')
print(copy_b*b)
print('multiply(copy_b,b):')
print(np.multiply(copy_b,b))
print('np.dot(b,a):')
print(np.dot(b,a))

