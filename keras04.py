import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x= np.array([1,2,3,4,5])
y= np.array([1,2,3,5,4])

# [실습] 만든다 [6]을 예측한다.

#2. 모델구성(그림으로 이해)
model = Sequential()
model.add(Dense(2, input_dim=1)) #여기서부터 히든레이어
model.add(Dense(26))
model.add(Dense(21))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1)) #아웃풋레이어


#3. 컴파일, 훈련 compile : 편집하다.
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x,y, epochs=999)


#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

result= model.predict([6])
print('[6]의 예측값은 : ', result)


# 6.0027885
#1/1 [==============================] - 0s 78ms/step
#[6]의 예측값은 :  [[6.002547]]
