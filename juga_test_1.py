import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))

def split_x(dt, st):
    a = []
    for i in range(len(dt)-st):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/juga/'
path_save = './_save/juga/'

datasets = pd.read_csv(path + '', index_col=        )

train_csv = 
test_csv = 

# 1.2 결측치가 일정 수 이상인 열 삭제

# 1.3 train, test 데이터 합치기 ( 결측치 제거와 원핫을 위해 )

# 1.4 범주형 결측치 처리 (SimpleImputer)

# 1.5 숫자형 결측치 처리 (KNNImputer)

# 1.6 원핫인코딩

# 1.7 (원본) train, test 분리

# 1.8 원본의 train 데이터에서 x, y 분리

# 1.9 train, test 분리

# 1.10 Scaler

# 2. 모델구성


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=   )
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
hist = model.fit()

model.save('./_save/juga/' + date + '.h5')


# 4. 평가, 예측
loss = model.evaluate()
print('loss : ', loss)

rmse = RMSE()
r2 = r2_score()

print('RMSE : ', rmse)
print('r2 : ', r2)

# 4.1 내보내기
