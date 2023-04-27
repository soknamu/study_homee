#내꺼

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input,LeakyReLU
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
#1. 데이터

path = './_data/kaggle_house/'
path_save = './_save/kaggle_house/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)                                                                  
non_numeric_cols = train_csv.select_dtypes(exclude=np.number).columns.tolist()
train_csv = train_csv.drop(non_numeric_cols, axis = 1)
test_csv = test_csv.drop(non_numeric_cols, axis = 1)

le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
print(len(train_csv.columns))
# print(train_csv.info())
train_csv=train_csv.dropna()
print(train_csv.shape)

# train_csv = train_csv.fillna(train_csv.median())

# print(train_csv.isnull().sum())

print(train_csv.shape) #(1460, 80)


x = train_csv.drop(['SalePrice'], axis= 1)
y = train_csv['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 850, train_size= 0.7)

print(x_train.shape, x_test.shape) #(1021, 79) (439, 79)
print(y_train.shape, y_test.shape) #(1021,) (439,)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


#2.모델구성

input1 = Input(shape = (79,))
dense1 = Dense(60,activation= 'relu')(input1)
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(50,activation= 'relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense4 = Dense(40,activation= 'relu')(drop2)
drop3 = Dropout(0.4)(dense4)
dense5 = Dense(30,activation= 'relu')(drop3)
dense6 = Dense(20,activation= 'relu')(dense5)
drop4 = Dropout(0.5)(dense5)
output1 = Dense(1,activation= 'linear')(drop4)
model = Model(inputs = input1, outputs = output1)

#3.컴파일

es = EarlyStopping(monitor= 'val_loss', patience= 1000, mode = 'min',
                   restore_best_weights= True,
                   verbose= 1)

model.compile(loss = 'mse',
              optimizer = 'adam',
             )

#날짜입력.
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
print(date)

filepath = './_save/MCP/dacon_wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor = 'val_acc', mode = 'auto',
#                        verbose = 1, 
#                        save_best_only= True,
#              filepath = "".join([filepath, 'wine_', date, '_', filename]))

hist = model.fit(x_train, y_train, epochs = 50, 
          batch_size = 850, verbose = 1,
          validation_split= 0.2,
          callbacks = [es,
                       #mcp
                       ])

#4. 평가, 예측

results = model.evaluate(x_test,y_test)
print('results :', results)
y_predict = model.predict(x_test)

#파일저장

y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)

submission['SalePrice'] = y_submit

submission.to_csv(path_save + 'kaggle_house_' + date +'.csv')

