'''
회귀(남자or여자or외계인) 분류?(yes or no) 2개이상분류(다중분류)


'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.data
x= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y= np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state= 124)  # -> 첫번째 x가 x_train이랑 x_test로 먼저감
 #train_size 보통 60~80%로 변경
#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(7))
model.add(Dense(16))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mae', optimizer = 'adam') 
model.fit(x_train, y_train, epochs = 500, batch_size =5)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)      # x를 넣었을 때 y예측값나옴 (x * m = y의값)
                                       # -> x대신 x_test로 수정(이유: 이미 훈련된 값도 포함되어있기 때문)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # -> y대신 y_test로 수정(이유: 이미 훈련된 값도 포함되어있기 때문)
print('r2스코어 : ', r2)
# R squared : 회귀분석의 성능 평가 척도 중 하나로, 결정력(결정계수)라고도 합니다.
#  1에 가까울수록 독립변수가 종속변수를 잘 설명할 수 있다는 뜻입니다.

'''
1/1 [==============================] - 0s 80ms/step
r2스코어 :  0.9779169389582473
'''