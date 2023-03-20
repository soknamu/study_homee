import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder, LabelEncoder
from tensorflow.python.keras.models import Model, load_model, sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from xgboost import XGBClassifier

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/wine/'
path_save = './_save/wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape)      # (5497, 13), (1000, 12)
# print(train_csv.columns, test_csv.columns)
# print(train_csv.info(), test_csv.info())
# print(train_csv.describe(), test_csv.describe())
# print(type(train_csv), type(test_csv))

# 1.3 결측지 제거
# print(train_csv.isnull().sum())     # 결측지 없음

# 1.4 라벨인코딩
le = LabelEncoder()
le.fit(train_csv['type'])
train_csv['type']=le.transform(train_csv['type'])
test_csv['type']=le.transform(test_csv['type'])


# 1.5 x, y 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

print(x.shape)      # (5497, 11)
print(test_csv.shape)       # (5497, 11)

# 1.6 원핫인코딩
# print(np.unique(y))     # [3 4 5 6 7 8 9]
# print(type(y))
# y = y - np.min(y)
# print(y)
# print(np.unique(y))     # [0 1 2 3 4 5 6] 
y = pd.get_dummies(y)
y = np.array(y)

# y = to_categorical(y)
# print(y)
# print(y.shape) 
# y = y[:,3:]
# print(y.shape)
# print(y)


# 1.7 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, stratify=y)

# 1.8 Scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# 2. 모델구성
# input1 = Input(shape=(12,))
# dense1 = Dense(32, activation='relu')(input1)
# drop1 = Dropout(0.2)(dense1)
# dense2 = Dense(64, activation='relu')(drop1)
# drop2 = Dropout(0.2)(dense2)
# dense3 = Dense(32, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# output1 = Dense(7, activation='softmax')(drop3)
# model = Model(inputs=input1, outputs=output1)

model = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=20, random_state=42)
model.fit(x_train, y_train)

importances = model.feature_importances_
for i, v in enumerate(importances):
    print('Feature: %0d, Score: %.5f' % (i, v))
    
# 3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=50, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
# result = model.evaluate(x_test, y_test)
# print('result', result)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=-1)
y_test = np.argmax(y_test, axis=-1)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

# import matplotlib.pyplot as plt
# plt.plot(hist.history['val_acc'], label ='val_acc')
# plt.plot(hist.history['acc'], label = 'acc')
# plt.legend()
# plt.show()

# 4.1 내보내기
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
y_submit = model.predict(test_csv)

# print(y_submit)
# print(y_submit.shape)
y_submit = np.argmax(y_submit, axis=1)
# print(y_submit)
# print(y_submit.shape)

y_submit += 3

# print(y_submit)

submission['quality'] = y_submit
import datetime
date = datetime.datetime.now()
print(date)
date = date.strftime("%m%d_%H%M%S")

submission.to_csv(path_save + 'submit_' + date + '.csv')
