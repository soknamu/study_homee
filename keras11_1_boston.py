from sklearn.datasets import load_boston #load boston 을 가져오겠다.

#1. 데이터
datasets = load_boston() # 로드 보스턴을 데이터셋으로 부르겠다.
x = datasets.data      # 13개
y = datasets.target    # 1개

# print(x)
# print(y)
#FutureWarning: Function load_boston is deprecated; 
# `load_boston` is deprecated in 1.0 and will be removed in 1.2
# 우리는 로드보스턴 을 사용하지 않을 것이다. 1.0에서
# 4.9800e+00 정규화된 수 ex)1조 *1조는 오버클럭이 생김 그래서 수를 최댓값으로 나눠서 1을 못넘게함
# print(datasets)


#print(datasets.feature_names)

#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] -> inputdim=13


# print(datasets.DESCR)

# instance : 506 예시
# attraribute 13개 y는 1000달러의 한개의 컬럼

# print(x.shape,y.shape)    #(506, 13) (506,) 스칼라 506 백터1개

###############[실습]##############
#1.train 0.7
#2. R2 0.8이상
###################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state= 39471848)
 
#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim=13))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(13))
model.add(Dense(18))
model.add(Dense(18))
model.add(Dense(21))
model.add(Dense(35))
model.add(Dense(21))
model.add(Dense(43))
model.add(Dense(37))
model.add(Dense(30))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(6))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mse', optimizer = 'adam') 
model.fit(x_train, y_train, epochs = 1000, batch_size =6)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)      
                                       

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


#5/5 [==============================] - 0s 754us/step
#r2스코어 :  0.7133451960164805