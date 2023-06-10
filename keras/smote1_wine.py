import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score #
from imblearn.over_sampling import SMOTE #데이터 증폭
from sklearn.ensemble import RandomForestClassifier
#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) #(178, 13) (178,)

#print(np.unique(y, return_counts=True))
#(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
#print(pd.Series(y).value_counts().sort_index())
# 1    71            0   59
# 0    59    ->      1   71
# 2    48            2   48

#print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x = x[:-25] #뒷자리 -25를 짜르겠다. 
y = y[:-25]
print(x.shape, y.shape) #(153, 13) (153,)


x_train, x_test, y_train, y_test = train_test_split(
x,y, shuffle=True, train_size=0.75, random_state=3377, 
stratify=y
)

#2. 모델

model = RandomForestClassifier(random_state=3377)

#3. 훈련
model.fit(x_train,y_train)

#4.평가, 예측
score = model.score(x_test, y_test)

y_predict = model.predict(x_test)

print("model_score :", score)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("f1_score(micro) :", f1_score(y_test, y_predict, average = 'micro')) #가중치 평균을 내주는것, 원래f1_score는 이중분류.
print("f1_score(macro) :", f1_score(y_test, y_predict, average = 'macro'))

# model_score : 0.9487179487179487
# accuracy_score : 0.9487179487179487
# f1_score(micro) : 0.9487179487179487
# f1_score(macro) : 0.9439984430496765

print("=================SMOTE 적용후 ============================")
smote = SMOTE(random_state=337,k_neighbors= 8) #최근접 이웃 방식. K: n개와 같은느낌. (단점: 생성시간이 엄청 오래걸림.)
x_train, y_train = smote.fit_resample(x_train, y_train) #값이 쏠리는 것을 막기위해 사용.
print(x_train.shape,y_train.shape) #(159, 13) (159,)
print(pd.Series(y_train).value_counts().sort_index()) #웬만하면 y_test는 건드리지 않기.
# 0    53
# 1    53
# 2    53

#2. 모델

model = RandomForestClassifier(random_state=37)

#3. 훈련
model.fit(x_train,y_train)

#4.평가, 예측
score = model.score(x_test, y_test)

y_predict = model.predict(x_test)

print("model_score :", score)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("f1_score(micro) :", f1_score(y_test, y_predict, average = 'micro')) #가중치 평균을 내주는것, 원래f1_score는 이중분류.
print("f1_score(macro) :", f1_score(y_test, y_predict, average = 'macro'))