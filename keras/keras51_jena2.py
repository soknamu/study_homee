import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_jena/'
path_save = './_save/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

x = datasets.drop(['T (degC)'], axis=1)
y = datasets['T (degC)']

scaler = MinMaxScaler()
x_train, x_test = train_test_split(x, train_size=0.7, shuffle=False)
scaler.fit(x_train)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False)
x_test, x_predict, y_test, y_predict = train_test_split(x_test, y_test, train_size=2/3, shuffle=False)

def split_x(datasets, timesteps):
    a = []
    for i in range(len(datasets)-timesteps-1):
        b = datasets[i:(i+timesteps)]
        a.append(b)
    return np.array(a)

timesteps = 10

x_train_split = split_x(x_train, timesteps)
x_test_split = split_x(x_test, timesteps)
x_predict_split = split_x(x_predict, timesteps)

y_train_split = y_train[(timesteps+1):]
y_test_split = y_test[timesteps+1:]
y_predict_split = y_predict[11:]

# 2. 모델구성
model = Sequential()
model.add(LSTM(16, input_shape=(10, 13)))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', restore_best_weights=True)
model.fit(x_train_split, y_train_split, epochs=1, batch_size=128, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test_split, y_test_split, verbose=1, batch_size=64)
print('loss : ', loss)

predict = model.predict(x_predict_split)
r2 = r2_score(predict, y_predict_split)
print('r2 : ', r2)

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

rmse = RMSE(predict, y_predict_split)
print('rmse : ', rmse)

print(predict[-2:])