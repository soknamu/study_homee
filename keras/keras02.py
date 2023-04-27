# #1. 데이터
import numpy as np       # 넘파이를 약자로 부른다.
x = np.array([1,2,3])    # array = 배열 
y = np.array([1,2,3])    #

# #2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential #모델스 폴더안에 시퀀셜이라는 폴더를 땡겨온다.
from tensorflow.keras.layers import Dense # layers 폴더안에 Dense이라는 폴더를 땡겨온다.

model = Sequential()   
model.add(Dense(1, input_dim=1))   

# #3. 컴퓨터, 훈련
model.compile(loss='mse', optimizer= 'adam')   # 최저값= optimizer 'mse= mean squared error평균제곱오차(손실함수)'
model.fit(x,y, epochs=99)   # fit = 훈련, x,y데이터를 epochs= 훈련횟수
