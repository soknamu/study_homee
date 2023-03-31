import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #이미지 전처리 데이터

#1. 데이터

path = 'd:/study_data/_save/_npy/'
# np.save(path + 'keras55_1_x_train.npy', arr = xy_train[0][0])
# np.save(path + 'keras55_1_x_test.npy', arr = xy_test[0][0])
# np.save(path + 'keras55_1_y_train.npy', arr = xy_train[0][1])
# np.save(path + 'keras55_1_y_test.npy', arr = xy_test[0][1])


x_train = np.load(path + 'keras55_1_x_train.npy')
x_test = np.load(path + 'keras55_1_x_test.npy')
y_train = np.load(path + 'keras55_1_y_train.npy')
y_test = np.load(path + 'keras55_1_y_test.npy')

print(x_train)
print(x_train.shape, x_test.shape) #(160, 100, 100, 1) (120, 100, 100, 1)
print(y_train.shape, y_test.shape) #(160,) (120,)


#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape= (100,100,1), activation= 'relu'))
model.add(Conv2D(64, (3,3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(16, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

hist = model.fit(x_train, y_train, epochs=1000, 
                    steps_per_epoch= 32,
                    validation_data= (x_test,y_test),
                    validation_steps=24,
                    )

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print(acc)

print('val_loss : ', val_loss[-1])
print('loss : ', loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

#1. 그림그리기!
from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.plot(loss,label= 'loss')
plt.plot(val_loss,label= 'val_loss')

plt.subplot(1,2,2)
plt.plot(acc,label= 'acc')
plt.plot(val_acc,label= 'val_acc')

plt.show()
