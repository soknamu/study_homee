import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10,])    #맨뒤에 있는 곳에 ,(쉼표)를 찍어도 문제가 생기지 않는다.
y = np.array([10,9,8,7,6,5,4,3,2,1,])
# print(x)
# print(y)

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7]) #대문자로 하든 소문자로 해도 상관없다.
x_test = np.array([8,9,10])               #실무에서는 엄청많다.
y_test = np.array([8,9,10])


print(x_train.shape, x_test.shape) #(7,) (3,)
print(y_train.shape, y_test.shape) #(7,) (3,)

#2. modeling

model = Sequential()
model.add(Dense(10, input_dim = 1 ))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. compile, train

model.compile(loss = 'mse' , optimizer = 'adam') #adam은 최적화
model.fit(x_train, y_train, epochs = 333, batch_size =2) #one F5에 4번돔(train 1~7이기때문에 2 2 2 1)

#4. evaluate, predict

loss = model.evaluate(x_test ,y_test) #x_test ,y_test가 적어서 오차가 적게나옴
print('loss : ', loss)

result = model.predict([5])
print('[11]의 예측값은 : ', result)


# 1/1 [==============================] - 0s 87ms/step
# [11]의 예측값은 :  [[11.000003]]

# 1/1 [==============================] - 0s 70ms/step
# [11]의 예측값은 :  [[11.]]

# 1/1 [==============================] - 0s 74ms/step
# [11]의 예측값은 :  [[10.999998]]

# 1/1 [==============================] - 0s 84ms/step
# [11]의 예측값은 :  [[11.000001]]
