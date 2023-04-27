import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from xgboost import XGBClassifier

# 1. 데이터
# 1.1 경로
path = './_data/ddarung/'
path_save = './_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
# print(train_csv.shape)
# print(train_csv.info())
# print(train_csv.describe())
# print(train_csv.columns)
# print(type(train_csv))

# 1.3 결측지 제거

# 1.4 x, y분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']



# 1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=82)

# 2. 모델구성
model = XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=4, random_state = 1)

# 3. 컴파일, 훈련
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print(rmse)

# 4.1 내보내기
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv', index_col=0)
submission['count'] = y_submit

submission.to_csv(path_save + 'submit_xgb_0307_0410.csv')