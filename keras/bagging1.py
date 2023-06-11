# stacking -> 약한 모델들 합쳐서 강한 모델로 다시돌림.

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
x,y  = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, train_size=0.8, random_state=1030
)

scaler = StandardScaler()
x_train =  scaler.fit_transform(x_train)
x_test =  scaler.fit_transform(x_test)

#2. 모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
# model = RandomForestClassifier
model = BaggingClassifier(DecisionTreeClassifier(),
                          n_estimators=5,
                          n_jobs=-1,
                          bootstrap=False, #디폴트 true
                          random_state=1030)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test,y_test))
print("acc : ", accuracy_score(y_test,y_pred))

#디시젼트리
# model.score :  0.9122807017543859
# acc :  0.9122807017543859

#randomforest
# model.score :  0.9473684210526315
# acc :  0.9473684210526315

#BaggingClassifier 100
# model.score :  0.9385964912280702
# acc :  0.9385964912280702

# 5번
# model.score :  0.956140350877193
# acc :  0.956140350877193

#배깅/10번/부스트트랩 false