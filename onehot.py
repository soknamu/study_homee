from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

y = np.array([2,4,6,8,10])
print(y.shape)

# y = to_categorical(y)
# print(y)
# print(y.shape)
# print(type(y))

# # (5,)
# # [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# # (5, 11)
# # <class 'numpy.ndarray'>

# y=np.delete(y, 0, axis=1)
# print(y)

# # <class 'numpy.ndarray'>
# # [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]

# y=np.delete(y, 0, axis=1)
# print(y)

# # [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 1.]]

# y=np.delete(y, (1,3,5,7), axis=1)
# print(y)

# # [[1. 0. 0. 0. 0.]
# #  [0. 1. 0. 0. 0.]
# #  [0. 0. 1. 0. 0.]
# #  [0. 0. 0. 1. 0.]
# #  [0. 0. 0. 0. 1.]]




# y = pd.get_dummies(y)
# print(y)
# print(y.shape)
# print(type(y))
# y = np.array(y)
# print(type(y))

# # (5,)
# #    2   4   6   8   10
# # 0   1   0   0   0   0
# # 1   0   1   0   0   0
# # 2   0   0   1   0   0
# # 3   0   0   0   1   0
# # 4   0   0   0   0   1
# # (5, 5)
# # <class 'pandas.core.frame.DataFrame'>
# # <class 'numpy.ndarray'>





# ohe = OneHotEncoder()
# y = y.reshape(-1, 1)
# y = ohe.fit_transform(y).toarray()
# print(y)
# print(y.shape)

# (5,)
# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
# (5, 5)


le = LabelEncoder()
y = le.fit_transform(y)
print(y)
print(y.shape)

# [0 1 2 3 4]
# (5,)