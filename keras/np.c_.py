import numpy as np
a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [10,11,12]])

b = [1,2,3,4]

c = np.c_[a,b]
print(c)




x = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [10,11,12]])

y = [[1,2,3]]

z = np.r_[x,y]
print(z)


# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/call/'
path_save = './_save/call/'

import pandas as pd

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
train_csv['주간통화시간'] = train_csv['주간통화시간'].astype('Float32')
print(train_csv.dtype())

a = train_csv['주간통화시간'] + train_csv['저녁통화시간'] + train_csv['밤통화시간']
b = train_csv['주간통화횟수'] + train_csv['저녁통화횟수'] + train_csv['밤통화횟수']

print(a)
print(b)