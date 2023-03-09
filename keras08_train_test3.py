import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10,])
#y = np.array([10,9,8,7,6,5,4,3,2,1,])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3, 
    random_state=51, 
    shuffle=True
    )
#test_size 가능 train_size 도 가능
#random_state 랜덤시드표(군대암호표) 처럼 1이면 123 789 이렇게 뽑아라 라는 표가있음(고정되있음)
# 랜덤값을 1~10까지 뽑아내지만, 훈련을 시킬때 데이터외의 나머지의 부분을 튜닝, 
# 데이터값이 바뀌면 훈련시키면 모델을 잘만들었는지 못만들었는지 알 수 없음
#랜덤이 되더라도 데이터값이 고정되어야됨 고정되지 않으면 모델의 성능을 알수가없음.
# 그래서 나온게 랜덤시드다.
print(x_train, x_test)
print(y_train, y_test)



# [검색] train 과 test를 섞어서 7:3으로 찾을 수 있는 방법!
# 힌트 사이킷런

# test_size : 전체데이터에서 테스트 데이터 세트의 크기를 얼마나 샘플링할 것인가
# ex) 0.3이면 train 70%, test 30%
# random_state 호출할때마다 동일한 학습/테스터용 데이터 세트를 생성하기 위해 주어지는 난수 값
# shuffle = true 셔플로 할것이다. 보통은 디폴트값으로 주석처리해도됨 ,false로 하면 섞이지 않게됨 

#2. modeling

model = Sequential()
model.add(Dense(10, input_dim = 1 ))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. compile, train

model.compile(loss = 'mse' , optimizer = 'adam') #adam은 최적화
model.fit(x_train, y_train, epochs = 2700, batch_size =1) #one F5에 4번돔(train 1~7이기때문에 2 2 2 1)

# -> 위에 test를 적어놔도 train해도 실행가능


#4. evaluate, predict

loss = model.evaluate(x_test ,y_test) #x_test ,y_test가 적어서 오차가 적게나옴
print('loss : ', loss)

result = model.predict([4])
print('[4]의 예측값은 : ', result)




