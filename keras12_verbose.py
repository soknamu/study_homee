

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_boston

#1.데이터

datasets = load_boston()  #괄호 안써서 틀림

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(    
x,y, train_size= 0.7, shuffle= True, random_state= 30)

#2.모델구성

model = Sequential()

model.add(Dense(5, input_dim= 13)) #input_dim 제대로 확인 안해서 틀림
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
model.fit(x_train, y_train, epochs = 300, batch_size = 32, verbose=0) 

            #verbose = 0이면 훈련과정 생략
            # 1과 auto가 들어가면 원래 그대로의 값(디폴트값)
            # 2면 프로그래스바(진행바)가 나오지 않음
            # 0과 1과 2가 아니면 epochs만나옴 ->이것을 조절함으로서 속도가 향상됨(매우빨라지지는 않음)

'''
#4. 평가

loss = model.evaluate(x_test, y_test, verbose=0)
print('loss : ', loss)


# import matplotlib.pyplot as plt

# plt.scatter(x_test,y_test)

# plt.show()   # ValueError: x and y must be the same size x와y의 값이 같지 않아서 오류가 뜸

'''
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score

r2 = r2_score(y_predict, y_test)

print('r2 score : ',r2)
