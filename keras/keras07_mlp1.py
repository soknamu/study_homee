#원본

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

# ex 환율 유가    행무시, 열우선-> 열은 하나의 특성을 가짐(세로)

x = np.array(
   [[1,1],
    [2,1],
    [3,1],
    [4,1],
    [5,2],
    [6, 1.3],
    [7, 1.4],
    [8, 1.5],
    [9, 1.6],
    [10, 1.4]]
   ) #10행 2열 -> 2개의 특성을 가진 10개의 데이터
y = np.array([11,12,13,14,15,16,17,18,19,20]) # -> 삼성전자의 주가

#괄호 안에 많이 들어간 애부터 숫자세기 



print(x.shape) #10행 2열 -> 2개의 특성을 가진 10개의 데이터
print(y.shape) # (10,)


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=2)) #3차원이면 input이 3이 되야됨
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#4. 컴파일, 훈련

model.compile(loss= 'mse', optimizer='adam')
model.fit(x, y, epochs=3000 , batch_size=3)

#4 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([[10, 1.4]]) #열의 개수랑 동일하게 해야됨 오류난이유: 벡터를 적었기때문에
print('[[10, 1.4]]의 예측값 : ', result)

