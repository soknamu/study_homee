from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(7, (2,2), padding='same', input_shape=(8,8,1)))        # 출력 : (N, 7, 7, 7)
# (batch_size, rows, columns, channels)

# (2*2)*1*7 +7 =35

model.add(Conv2D(filters=4, kernel_size=(3,), padding='same', activation='relu'))         # 출력 : (N, 5, 5, 4)
# (3*3)*7*4+4 =256

model.add(Conv2D(10, (2,2)))         # 출력 : (N, 4, 4, 10)
# (2*2)*4*10 + 10 =170

model.add(Flatten())            # 출력 : (N, 160)

model.add(Dense(32, activation='relu'))
# (160+1)*32

model.add(Dense(10, activation='relu'))
# (32+1)*10

model.add(Dense(3, activation='softmax'))
# (10+1)*3
model.summary()