import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBClassifier

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape)      #(10886, 11)
print(test_csv.shape)       #(6493, 8)

# print(train_csv.info())

# print(train_csv.describe())

print(train_csv.columns)
#Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')

print(test_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed'],
#       dtype='object')

# print(type(train_csv))      #<class 'pandas.core.frame.DataFrame'>

# 1.3 결측치 제거
# print(train_csv.isnull().sum()) # 원래 결측치가 없는 데이터
train_csv = train_csv.dropna()      # 생략 가능

# 1.4 x, y 분리
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

# 1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=1)

# 2. 모델구성
model = XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=4, random_state = 1)

# 3. 컴파일, 훈련
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
model.fit(x_train, y_train)

# 4. 평가, 예측
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# 4.1 내보내기
y_submit = model.predict(test_csv)
submission= pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit

submission.to_csv(path_save + 'submit_0310_0726.csv')