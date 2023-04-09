from tensorflow.keras.datasets import imdb # 영화평론 데이터
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000, 
)
'''
print(x_train)
print(y_train) #[1 0 0 ... 0 1 0] 긍정 부정
print(x_train.shape, y_train.shape) #(25000,) (25000,)
print(np.unique(y_train, return_counts=True)) #[0 1] 판다스에서는 value_counts
#(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
print(pd.value_counts(y_train)) 
1    12500
0    12500
dtype: int64
'''
print("영화평론의 최대길이 : ", max(len(i) for i in x_train)) #뉴스기사의 최대길이 :  2494
print("영화평론의 평균길이 : ", sum(map(len, x_train))/ len(x_train)) #뉴스기사의 평균길이 :  238.71364

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=239,
                        truncating= 'pre' #어디를 짜를 것이냐(100이상일 경우)
                        )
x_test = pad_sequences(x_test, padding='pre', maxlen=239,
                        truncating= 'pre' #어디를 짜를 것이냐(100이상일 경우)
                        )
#print(x_train.shape)


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
model.add(Embedding(10000, 32,input_length=239)) 
model.add(Reshape(target_shape=(239,32), input_shape =(239,)))
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

