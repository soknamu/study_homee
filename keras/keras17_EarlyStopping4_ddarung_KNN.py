import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))

# 결측치 처리 1. 제거
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)

print(train_csv.isnull().sum())
filled_train = imputer.fit_transform(train_csv)      ### KNNimputer 를 이용한 결측지 제거 ###
print(filled_train.shape)      #(1328, 10) [dropna()] -> (1459, 10) [KNNimputer] : 값 채워넣기로 행 수가 유지됨

print(filled_train)
print(type(filled_train))      # <class 'numpy.ndarray'> : 다시 데이터 프레임으로 변환 필요

filled_train = pd.DataFrame(filled_train, columns=train_csv.columns)
print(type(filled_train))      # <class 'pandas.core.frame.DataFrame'>

# 1.4 x, y 분리
x = filled_train.drop(['count'], axis=1)
print(x)

y = filled_train['count']
print(y)

# 1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=1000, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])

# 4. 평가, 예측
filled_test = imputer.fit_transform(test_csv)      ### KNNimputer 를 이용한 결측지 제거 ###

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

"""
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('따릉이')
plt.legend()
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.show()
"""
# 4.1 내보내기
filled_test = imputer.fit_transform(test_csv)      ### KNNimputer 를 이용한 결측지 제거 ###

submission = pd.read_csv(path + 'submission.csv', index_col=0)
y_submit = model.predict(filled_test)
submission['count'] = y_submit

submission.to_csv(path_save + 'submit_ES+KNN_0310_0743.csv')
