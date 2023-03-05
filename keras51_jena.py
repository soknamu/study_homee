import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, Flatten, Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(datasets)
print(len(datasets))
print(datasets.shape)
print(datasets.columns)
print(datasets.info())
print(datasets.describe())
print(type(datasets))

print(datasets['T (degC)'].values)      # 판다스를 넘파이로
print(datasets['T (degC)'].to_numpy)    # 판다스를 넘파이로

x = datasets.drop(['T (degC)'], axis=1)
y = datasets['T (degC)']

x_train, x_test, _ , _ = train_test_split(x, y, train_size=0.7, shuffle=False)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x=scaler.transform(x)

def split_x(a, b):
    aaa=[]
    for i in range(len(a) - b):
        subset = a[i : (i + b)]
        aaa.append(subset)
    return np.array(aaa)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False)
x_test, x_predict, y_test, y_predict = train_test_split(x_test, y_test, train_size=2/3, shuffle=False)

timesteps = 10
x_train_split = split_x(x_train, timesteps)
x_test_split = split_x(x_test, timesteps)
x_predict_split = split_x(x_predict, timesteps)

y_train_split = y_train[timesteps:]
y_test_split = y_test[timesteps:]
y_predict_split = y_predict[timesteps:]


# 2. 모델구성
model = Sequential()
model.add(Conv1D(1, 2, input_shape=(10, 13)))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train_split, y_train_split, epochs=100, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test_split, y_test_split)
print('loss : ', loss)

predict = model.predict(x_predict_split)

r2 = r2_score(y_predict_split, predict)
print('r2 : ', r2)

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

rmse = RMSE(y_predict_split, predict)
print('rmse : ', rmse)