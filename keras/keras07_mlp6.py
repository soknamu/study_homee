# input 3 output 3

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x = np.array([range(10), range(21,31), range(201, 211)]) #3행 10열 range ex) 10 -> 0~9 영부터 카운트

x = x.T #10행 3열

y = np.array([[1,2,3,4,5,6,7,8,9,10],
               [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 ],
               [9,8,7,6,5,4,3,2,1,0]]) # 3행 10열

y = y.T   #10행 3열

#예측 [[9,30,210]] -> [[10,1.9,0]]

#모델링

model = Sequential()

model.add(Dense(5, input_dim =3))
model.add(Dense(3))  # alt + shift + ↓ 누르면 복사 아랫열 복사
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))

model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x,y, epochs = 1998, batch_size = 1)

loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([[9,30,210]])
print('[9,30,210]의 예측값은 : ', result)

# 1/1 [==============================] - 0s 75ms/step
# [9,30,210]의 예측값은 :  [[10.000002   1.9000003]] epochs = 1000, batch = 2 'mse'

