from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2 #num_words 를 크게 줘야지 표현을 크게 할 수 있음.
)

'''
print(x_train)

print(y_train) #[ 3  4  3 ... 25  3 25]
print(x_train.shape, y_train.shape) #(8982,) (8982,)
print(x_test.shape, y_test.shape) #(2246,) (2246,) -> 0.2 로 했을 때 총11228개 0.1도 마찬가지. test_split로 test,train으로 나뉨.

print(len(x_train[0]), len(x_train[1])) #87 56 길이가 다른 리스트.
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 
# 4579, 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]
print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45] 46개의 클래스
#input_dim data = 10000개 데이터. output_dim 하고싶은대로. input_len

print(type(x_train), type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'>

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) #길이를 계속 뽑을 꺼다. 최대값이 나올때까지.
print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/ len(x_train)) #길이를 계속 뽑을 꺼다. 최대값이 나올때까지.
#뉴스기사의 최대길이 :  2376
#뉴스기사의 평균길이 :  145.5398574927633
'''

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100,
                        truncating= 'pre' #어디를 짜를 것이냐(100이상일 경우)
                        )
x_test = pad_sequences(x_test, padding='pre', maxlen=100,
                        truncating= 'pre' #어디를 짜를 것이냐(100이상일 경우)
                        )
print(x_train.shape) #(8982, 100)


#onehotencording
def onehot(y_train, y_test):
    y=np.concatenate((y_train,y_test))
    y=pd.get_dummies(y,prefix='number')
    return y[:len(y_train)],y[len(y_train):]
y_train,y_test = onehot(y_train, y_test)

#나머지 전처리하고
print(y_train.shape)
#모델 구성
#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, Embedding
#rnn으로 reshape (14,5,1)로 바꿔줌.
#고정된 크기로 변환 Embedding(one hot encording)
model = Sequential()
model.add(Embedding(10000, 32,input_length=100)) 
model.add(Reshape(target_shape=(100,32), input_shape =(100,)))
model.add(LSTM(128)) 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation= 'softmax'))

# 시작
model.summary()
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
es = EarlyStopping(monitor='acc', mode='max', patience=100, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 256, validation_data = (x_test, y_test), callbacks=[es])

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1] #0번쨰째가
print('acc: ', acc)

# acc:  0.5707924962043762