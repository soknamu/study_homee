from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.data
x= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y= np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state= 1234)  # -> 첫번째 x가 x_train이랑 x_test로 먼저감

#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(14))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 99, batch_size =2)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


y_predict = model.predict(x)      # x를 넣었을 때 y예측값나옴 (x * m = y의값)

#5. 시각화
import matplotlib.pyplot as plt   # -> 그림그리는거
plt.scatter(x,y)                  # -> 현재 자료에있는 점들 표시
#plt.scatter(x,y_predict)         # ->점으로 찍는다.
plt.plot(x,y_predict, color = 'green')  # -> 선, 색깔변경
plt.show()                        # -> 그림실행 명령어


#-> x의 예측값이 y의 예측값
# result = model.predict([4])
# print('[4]의 예측값  : ', result)

'''
회귀(남자or여자or외계인) 분류?(yes or no) 2개이상분류(다중분류)


'''