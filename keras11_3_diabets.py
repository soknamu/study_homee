#[실습]
# R2 0.62 이상 데이터를 더 정확해야지 할수 있음

from sklearn.datasets import load_diabetes

#1.데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442, 10) (442,) input_dim = 10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.9, shuffle=True, random_state=650874)
#650874
#2. 모델구성

model = Sequential()
#model.add(Dense(12,input_dim=10))
model.add(Dense(10,input_dim=10))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mse', optimizer = 'adam') 
model.fit(x_train, y_train, epochs = 500, batch_size =8)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)      
                                       

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#2/2 [==============================] - 0s 3ms/step
#r2스코어 :  0.7238750746771716


