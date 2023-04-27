#1. 데이터
import numpy as np 
x = np.array([1,2,3]) 
y = np.array([1,2,3])

#2. 모델구성(그림으로 이해)
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential() 
model.add(Dense(3, input_dim=1)) 
model.add(Dense(18)) 
model.add(Dense(17))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(1))

#3. 컴파일, 훈련 compile 편집하다.
model.compile(loss='mae', optimizer= 'adam')
model.fit(x,y, epochs=10)

#4 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([4]) #위에도 저렇게 표기해서 []를 표기
print("[4]의 예측값 : " , result)

#4.103033 3.9751494 3.9816754 3.9914048 3.9999995 4.