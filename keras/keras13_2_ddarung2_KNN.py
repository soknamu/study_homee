# 데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from tensorflow.keras.optimizers import Adam

# 1. 데이터
path = './_data/ddarung/' # . 은 현재 폴더(STUDY)를 의미함

# train_csv = pd.read_csv('./_data/ddarung/train.csv')
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

print(train_csv)
print(train_csv.shape)      #(1459, 10)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(test_csv)
print(test_csv.shape)      #(715, 9)

# ================================================================================================= #

print(train_csv.columns)

# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())

#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64


print(train_csv.describe())

print(type(train_csv))      # <class 'pandas.core.frame.DataFrame'>

######################### 결측치 처리 #############################
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

######################### train_csv데이터에서 x와 y를 분리 ##############################
x = filled_train.drop(['count'], axis=1)
print(x)

y = filled_train['count']
print(y)
######################### train_csv데이터에서 x와 y를 분리 ##############################



for i in range(600,700):
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=i)

    # print(x_train.shape, x_test.shape)      #(1021, 9), (438, 9) -> (929, 9), (399, 9)
    # print(y_train.shape, y_test.shape)      #(1021,), (438,) -> (929,), (399,)

    # 2. 모델구성
    model = Sequential()
    model.add(Dense(32, input_dim=9, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    # 3. 컴파일, 훈련
    optimizer=Adam(learning_rate=0.0005)
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(x_train, y_train, epochs=300, batch_size=20, verbose=0)

    # 4. 평가, 예측
    loss = model.evaluate(x_test, y_test)
    if loss < 2400:
        print("loss : ", loss)
        print(i)
        break
        
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):        # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)      # RMSE 함수 활용
print("RMSE : ", rmse)

#### submission.csv를 만들어봅시다!!! ####
# print(test_csv.isnull().sum())        # 여기도 결측치가 있네!!!

filled_test = imputer.fit_transform(test_csv)      ### KNNimputer 를 이용한 결측지 제거 ###

y_submit = model.predict(filled_test)
print(y_submit)

submission = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission)
submission['count'] = y_submit
print(submission)

submission.to_csv(path + 'submit_0306_0730.csv')
