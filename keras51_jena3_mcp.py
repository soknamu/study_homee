import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

# 1. 데이터
path = './_data/kaggle_jena/'
filepath = './_save/MCP/kaggle_jena/'
filename = '{epoch:04d}-{val_loss:.2f}.hdf5'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

x = datasets.drop(['T (degC)'], axis=1)
y = datasets['T (degC)']

def split_x(dt, st):
    a = []
    for i in range(len(dt)-st):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

timesteps = 10

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, shuffle=False)
x_train, x_predict, y_train, y_predict = train_test_split(x_train, y_train, train_size=2/3, shuffle=False)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)

x_train_split = split_x(x_train, timesteps)
x_test_split = split_x(x_test, timesteps)
x_predict_split = split_x(x_predict, timesteps)

y_train_split = y_train[timesteps:]
y_test_split = y_test[timesteps:]
y_predict_split = y_predict[timesteps:]

# 2. 모델구성
model = Sequential()
model.add(LSTM(32, input_shape=(10, 13)))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics='mae')
es = EarlyStopping(monitor='loss', mode='min', patience=10, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='loss', save_best_only=True, filepath="".join([filepath, 'kaggle_jena_', date, '_', filename]))
hist = model.fit(x_train_split, y_train_split, epochs=10, batch_size=128, callbacks=[es, mcp], validation_split=0.2)

model.save('./_save/MCP/kaggle_jena/kaggle_jena_' + date + '.h5')

# 4. 평가, 예측
loss = model.evaluate(x_test_split, y_test_split)
print('loss : ', loss)

predict_result = model.predict(x_predict_split)
print('r2 :', r2_score(predict_result, y_predict_split))

def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))
rmse = RMSE(predict_result, y_predict_split)

print('RMSE :', rmse)
print(predict_result[-1])

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.show()