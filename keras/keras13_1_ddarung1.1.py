import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
# print(train_csv.shape, test_csv.shape)
# print(train_csv.columns)
# print(train_csv.info())
# print(train_csv.describe())
# print(type(train_csv))

# 1.3 결측치 제거
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape)      #(1328, 10)

# 1.4 x와 y분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

# 1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=12412)

# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=9))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=0)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print('rmse : ', rmse)

# 4.1 내보내기
y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'submission.csv', index_col=0)
submission['count'] = y_submit

submission.to_csv(path + 'submit_0306_0712.csv')
