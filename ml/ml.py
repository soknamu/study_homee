from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import X
#1. 데이터

x,y = load_breast_cancer(return_X_y=True)

x_train,x_test, y_train,y_test = train_test_split(x,y,
    shuffle=True, stratify=y, train_size=0.7,random_state=369)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=369)

parameters = {'n_estimators' :10000,  # epochs 역할
              'learning_rate' : 0.3, # 학습률의 크기 너무 크면 최적의 로스값을 못잡고 너무 작으면 최소점에 가지도못하고 학습이끝남.
              'max_depth': 3,        #tree계열일때 깊이를 3개까지만 가겠다.
              'gamma': 0,
              'min_child_weight': 1, #최소의 
              'subsample': 0.5,      # dropout과 비슷한 개념.
              'colsample_bytree': 1,
              'colsample_bylevel': 1., #xgboost.core.XGBoostError: Invalid Parameter format for colsample_bylevel expect float but value='[1]' 리스트형태하지마라
              'colsample_bynode': 1,
              'reg_alpha': 1,        #규제
              'reg_lambda': 1,
              'random_state': 369,
}

#2.모델
model = randomforest