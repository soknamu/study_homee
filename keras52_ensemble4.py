# 1. 데이터
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

x1_datasets = np.array([range(100), range(301, 401)])       # 삼성, 아모레
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)])     
x3_datasets = np.array([range(201, 301), range(511, 611), range(1300, 1400)])     

x1 = np.transpose(x1_datasets)
# x2 = x2_datasets.T
# x3 = x3_datasets.T

y1 = np.array(range(2001, 2101))     # 환율
y2 = np.array(range(1001, 1101))

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, train_size=0.7, random_state=333)

# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 2.1 모델1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='stock1')(input1)
dense2 = Dense(200, activation='relu', name='stock2')(dense1)
dense3 = Dense(300, activation='relu', name='stock3')(dense2)
output1 = Dense(110, activation='relu', name='output1')(dense3)

# # 2.2 모델2
# input2 = Input(shape=(3,))
# dense11 = Dense(100, name='weather1')(input2)
# dense12 = Dense(100, name='weather2')(dense11)
# dense13 = Dense(100, name='weather3')(dense12)
# dense14 = Dense(100, name='weather4')(dense13)
# output2 = Dense(110, name='output2')(dense14)

# # 2.3 모델3
# input3 = Input(shape=(3,))
# dense21 = Dense(100, name='1')(input3)
# dense22 = Dense(100, name='2')(dense21)
# dense23 = Dense(100, name='3')(dense22)
# dense24 = Dense(100, name='4')(dense23)
# output3 = Dense(110, name='output3')(dense24)


# from tensorflow.keras.layers import concatenate, Concatenate
# # 소문자는 함수, 대문자는 클래스

# # 2.4 머지
# merge1 = concatenate([output1, output2, output3], name='mg1')
# merge2 = Dense(200, activation='relu', name='mg2')(merge1)
# merge3 = Dense(300, activation='relu', name='mg3')(merge2)
# hidden_output = Dense(100, name='last')(merge3)

# 2.5 분기1
bungi1 = Dense(10, activation='selu', name='bg1')(output1)
bungi2 = Dense(10, name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)

# 2.6 분기2
last_output2 = Dense(1, activation='linear', name='last2')(output1)

# a = Dense(1, name='a')(last_output)
# b = Dense(1, name='b')(last_output)

model = Model(inputs=input1, outputs=[last_output1, last_output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode='auto', patience=50, restore_best_weights=True)
model.fit(x1_train, [y1_train, y2_train], epochs=1000, batch_size=8, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x1_test, [y1_test, y2_test])
print('loss : ', loss)      # ( 전체 loss , loss1, loss2 )
y_predict = model.predict(x1_test)

r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])

print("r2 : ", (r2_1 + r2_2)/2) 