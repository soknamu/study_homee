import numpy as np
import pandas as pd

a = [[1,2,3], [4,5,6]] #2행3열

b = np.array(a)
print(b) # [[1 2 3]
#           [4 5 6]]

c = [[1,2,3], [4,5]]
print(c) #[[1, 2, 3], [4, 5]]
d = np.array(c)
print(d) #[list([1, 2, 3]) list([4, 5])]
# 1. 리스트는 크기가 달라도 상관이 없다.

########################################
e = [[1,2,3], ["바보", "맹구", 5, 6]]
print(e) #[[1, 2, 3], ['바보', '맹구', 5, 6]]

#2. 리스트에는 다른 자료형(숫자, 글자)을 넣어도 상관이 없다.

f = np.array(e)
print(f) #[list([1, 2, 3]) list(['바보', '맹구', 5, 6])] 와꾸만 맞으면 상관없음.

#print(e.shape) #AttributeError: 'list' object has no attribute 'shape'
#리스트 자체에는 shape가 존재하지 않는다. 그래서length로 수치를 확인하는 방법 밖에 없다.