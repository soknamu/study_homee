'''*****(굉장히 중요)분리를 하되, 전체범위에서 일부값을 랜덤으로 뽑아서 테스트에 넣든, 트레이닝에 넣어서 할 것.*****

백만개의 데이터를 셔플(섞는다) 70만개 트레이닝 30만개 테스트
30만개를 아무리빼봤자 범위가 바뀌지 않는다.(중간중간에 이빨이 많이 빠진것으로 보임)
훈련을 시킬때 랜덤으로 뽑는다. 나머지 30만개로 평가 데이터 모양도 바뀌지않고, 어느정도 유지됨
간격이 틀어지는 일 없음.
'''


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10,])
y = np.array([10,9,8,7,6,5,4,3,2,1,])

#[실습] 넘파이 리스트의 슬라이싱 7:3으로 잘라라

x_train = x[0:7]     #  0:7 : 1부터 7까지 or :7 도 가능 ->1,2,3,4,5,6,7
x_test = x[7:10]     #  7:10 : 8부터 10까지 or 7: 도 가능
y_train = y[:7]      # ->10,9,8,7,6,5,4 역수로 나오는건 당연 문제X
y_test = y[7:]
print(y_train)

# print(x_train.shape, x_test.shape) #(7,) (3,)
# print(y_train.shape, y_test.shape) #(7,) (3,)

#2. modeling

model = Sequential()
model.add(Dense(10, input_dim = 1 ))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. compile, train

model.compile(loss = 'mse' , optimizer = 'adam') #adam은 최적화
model.fit(x_train, y_train, epochs = 300, batch_size =2) #one F5에 4번돔(train 1~7이기때문에 2 2 2 1)

#4. evaluate, predict

loss = model.evaluate(x_test ,y_test) #x_test ,y_test가 적어서 오차가 적게나옴
print('loss : ', loss)

result = model.predict([5])
print('[5]의 예측값은 : ', result)

