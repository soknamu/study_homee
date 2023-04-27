# 데이콘 따릉이 문제풀이

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error #mse 까지불러옴
import pandas as pd

#1. 데이터

path = './_data/ddarung/'  #점(.) 한개 :현재폴더, / : 현재폴더의 밑의폴더 -> 폴더위치 #변수 만들기


# train_csv = pd.read_csv('./_data/ddarung/train_csv') -> 원래는 이렇게 적어야되는데 path라는 변수를 넣어서 줄임
train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
# 데이터는 아니지만 사람이 볼때는 잘보이는데 컴퓨터는 그러지를 못하기 때문에 index를 사용
# 연산을 할때는 컬럼명(헤더)와 인덱스는 연산하지 않는다.
#index_col = 0 이면 맨앞줄id를 연산하지 않는다. /1이면 hour를 연산하지 않는다.

test_csv = pd.read_csv(path + 'test.csv', index_col= 0) 
# 자료(따릉이 자료를 부른 것)

print(train_csv)
print(train_csv.shape) #(1459, 11) 인덱스 추가후 #[1459 rows x 10 columns] #(1459, 10)

print(test_csv)
print(test_csv.shape) #[715 rows x 9 columns] (715, 9)

#==============================================================================

print(train_csv.columns) #-> 칼럼 정보를 확인
# #Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object') 카운트까지 10개

print(train_csv.info()) #결집치 데이터들을 볼 수 있음. -> 정보확인 숫자왼쪽 빨간점: 찍으면 중간점(여기까지만 실행됨)

print(train_csv.describe()) # 데이터의 mean(평균), std(표준편차), min(최소값), max(최대값) 등을 보여줌

#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
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

#행이 중복되어 있을 수 있기 때문에 최대 117개 지워짐

print(type(train_csv)) # <class 'pandas.core.frame.DataFrame'>

#####################결측치처리#######################
# 결측치 처리 1. 제거

print(train_csv.isnull().sum()) #이거는 결측치를 제거하기전 
train_csv = train_csv.dropna() #dropna 결측치 삭제
print(train_csv.isnull().sum()) #isnull : nan(결측치)이(가) 몇개있는지 확인 ,+sum 난의 계수 합계, (이거는 결측치를 제거한 후)
print(train_csv.info())
print(train_csv.shape)          #(1328,10)


########################train_csv 데이터에서 x와 y를 분리 ############################

#완벽하게 암기 판다스가 데이터 프라임으로 만들어짐
x = train_csv.drop(['count'],axis=1) # #aixs=0(index)은 행, aixs=1(index)은 열(세로)
                                     # []를 list라고 함. 두개 이상은 리스트(list)
                                     # 뜻 :트레인_csv에있는 카운트 열에 있는 것을 뺀다. axis는 드랍할때 같이 나옴.

#여기까지는 count를 뺀값.

print(x)

y = train_csv['count']

print(y)

x_train, x_test , y_train, y_test = train_test_split(
x,y, shuffle= True, random_state= 777, train_size= 0.7
)
                                                           #결측치 제거후
print(x_train.shape, x_test.shape) #(1021, 9) (438, 9) -> (929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(1021,) (438,) -> (929,) (399,)

#얘는 데이터를 빼고 결측치도 뺀값.


# 트레인과 테스트가 이미 파일이 존재하는데 train에서 왜 또다시 train과 test를 나누는 가?
# 실제에서는 test 따로 train 따로 훈련하고 테스트나눠서 훈련시키는데 
# 현재 우리가 하는 것은 train에 훈련과 테스트를 같이 하고 있기 때문에 그럼


#2. 모델구성


model = Sequential() # 두가지가 있는데 하나는 시퀀셜, 또하나는 함수

model.add(Dense(5, input_dim= 9))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련

model.compile(loss = 'mse', optimizer ='adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 32, verbose=1) 

            #verbose = 0이면 훈련과정 생략
            # 1과 auto가 들어가면 원래 그대로의 값(디폴트값)
            # 2면 프로그래스바(진행바)가 나오지 않음
            # 0과 1과 2가 아니면 epochs만나옴 ->이것을 조절함으로서 속도가 향상됨(매우빨라지지는 않음)
            # # 원데이터는 건드리지 말기 스페이스바 한번이라도 누르면 바뀜.

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss) # nan은 값이 없기때문에 nan이라고 적음.
#결측치처리 할때 첫번째 솔루션 0을 넣는다. 하지만 온도 같은경우에는 0이면 확달라지기 때문에 x 

#loss = 2838이면 여기다가 루트를 씌우면 대략 40정도

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

def RMSE(y_test, y_predict) : #함수의 약자 RMSE 임의의 값. 함수는 y= f(x)라는 식으로 계속 이용함.
   return np.sqrt(mean_squared_error(y_test,y_predict))   #리턴(나오는 값)해주면 된다. 함수를 정의 한 것. 

#np.sqrt로 루트를 씌운다.
# 여기까지는 실행이 안되고, 정의만 한 것

rmse = RMSE(y_test, y_predict)  #실행 코드 RMSE 함수 사용

print("RMSE : ",rmse)