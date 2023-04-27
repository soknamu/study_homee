#1. 데이터
import numpy as np #넘파이가 행렬부분을 사람과 유사하게 해준다.
x = np.array([1,2,3]) # np는 넘파이의 array 배열하는 숫자
y = np.array([1,2,3])

#2. 모델구성(그림으로 이해)
import tensorflow as tf
from tensorflow.keras.models import Sequential 
# -> 텐서플로 케라스 모델 폴더 안에 있는Sequential(순서)이라는 기능을 불러온다. = 순차적 모델(클래스)를 쓴다.
from tensorflow.keras.layers import Dense
# -> layers(층)   Dense-> 댄서 모델을 가져오겠다.(함수모델 y= mx+b)

model = Sequential() # Sequential을 모델이라고 지정함
model.add(Dense(3, input_dim=1)) # 3을 붙인이유 첫부분에서 뻗어나가는 부분(input) ->이곳이 input layers 틀렸음 
 #input_dim가 첫번째 레이어이고, 앞에 있는 3은 히든 레이어여서 마음대로 수정가능
model.add(Dense(28)) #위에 인풋이 입력되어 있어서 인풋을 생략
model.add(Dense(32))
model.add(Dense(28))
model.add(Dense(23))
model.add(Dense(21))  #개수니깐 소수는 x
model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(18))  

#3. 컴파일, 훈련 compile 편집하다.
model.compile(loss='mse', optimizer= 'adam')
model.fit(x,y, epochs=100)

#0.0010