# input 3 output 1 (range 사용)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201, 211)]) #3행 10열 range ex) 10 -> 0~9 영부터 카운트

x = x.T #10행 3열

y = np.array([[1,2,3,4,5,6,7,8,9,10]]) # 1행 10열

y = y.T #10행 1열


#2. 모델링
model = Sequential()
model.add(Dense(5, input_dim = 3)) #인풋레이어가 3열이기때문에 디멘션3
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1)) # 이 마지막 Y값이기 때문에 아웃레이어가 1임


#3. 컴파일, 훈련
model.compile(loss = 'mse',  optimizer = 'adam')
model.fit(x, y, epochs = 25000,  batch_size = 1)


#4. 평가,예측
loss = model.evaluate(x,y)
print("loss : " , loss)

result = model.predict([[9,30,210]])
print("[9,30,210]의 예측값은 : ", result)


#1/1 [==============================] - 0s 105ms/step
#[9,30,210]의 예측값은 :  [[9.999701]]

# 1/1 [==============================] - 0s 89ms/step
# [9,30,210]의 예측값은 :  [[10.000035]]

# 1/1 [==============================] - 0s 103ms/step
# [9,30,210]의 예측값은 :  [[9.999841]]

# 1/1 [==============================] - 0s 104ms/step
# [9,30,210]의 예측값은 :  [[10.000002]] -> 25000 epochs, batch 1
