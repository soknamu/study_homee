# input 3 output 1

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x = np.array(
  [[1,2,3,4,5,6,7,8,9,10],    #항상 괄호 다음에는 쉼표를 찍기!
   [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
   [9,8,7,6,5,4,3,2,1,0]]
   ) 

x= x.transpose() # 괄호 넣기

y = np.array([11,12,13,14,15,16,17,18,19,20])

#[실습]
#예측 [[10,1.4,0]]

model = Sequential() #괄호 넣기
model.add(Dense(5, input_dim = 3)) # 3행이니깐 인풋레이어가 3
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))  # y가 1행이니깐 아웃풋레이어가 1

model.compile(loss = 'mse',  optimizer = 'adam')
model.fit(x, y, epochs = 20,  batch_size = 3)

loss = model.evaluate(x,y)
print("loss : " , loss)

result = model.predict([[10,1.4,0]])
print("[10,1.4,0]의 예측값은 : ", result)

