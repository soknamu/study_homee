import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #이미지 전처리 데이터

train_datagen = ImageDataGenerator(
    rescale= 1./255, #스케일링 하겠다.(. 붙히는 이유 : 부동소수점으로 연산, 정규화(nonaliazation))
    # horizontal_flip=True, #(가로뒤집기)
    # vertical_flip= True,  #좌우 반전
    # width_shift_range= 0.1, #소수점만큼의 사진을 넓이를 이동시킨다.(증폭)
    # height_shift_range= 0.1, # 소수점만큼의 사진을 높이를 이동시킨다.(증폭)
    # rotation_range= 5,
    # zoom_range= 1.2, #확대
    # shear_range= 0.7, #찌그러 트리는거
    # fill_mode= 'nearest',  # 이동된자리가 짤림 그걸방지하는 것이 있음. nearest는 옮겨진 값을 근처값으로 넣어줌.
    # -> 지금까지 기능은 다 증폭기능. 넣거나 더 빼도됨.
)

test_datagen = ImageDataGenerator(
    rescale= 1./255,   #평가 데이터를 증폭시키는 것은 데이터조작이 될 수 있어서 증폭시키지 않음.
)

xy_train = train_datagen.flow_from_directory(
    'd:study_data/_data/brain/train/', #ad normal 폴더로 들어가지 않는 이유. 라벨값이 정해져 있기 때문에. 그래서 상위폴더로 지정해줌.
    target_size=(100, 100), #모은 사진들을 확대 또는 축소를 해서 크기를 무조건 200 * 200으로 만듬. (크든 작든)
    batch_size= 5,  #5장씩 쓰겠다. #전체데이터를 쓸라면 160(데이터의 최대의 개수만큼(그이상도 가능.))넣어라!
    class_mode= 'binary',#데이터가 2개밖에 없기때문에 (수치화 되서 만들어줌) 카테고리컬은
    color_mode= 'grayscale',#흑백칼라 
    #color_mode= 'rgba',#빨초파(투명도)
    shuffle= True,
) #directory = folder

xy_test = test_datagen.flow_from_directory(
    'd:study_data/_data/brain/test/',
    target_size=(100, 100),
    batch_size= 5,
    class_mode= 'binary',
    color_mode= 'grayscale',
    shuffle= True,
)

print(xy_train)
print(xy_train[0])
#print(xy_train.shape) #error
print(len(xy_train))    #32
print(len(xy_train[0])) #2 -> x와 y
print(xy_train[0][0]) #엑스가 다섯개 들어가있다. 
print(xy_train[0][1]) #[1. 0. 0. 0. 1.]

print(xy_train[0][0].shape) # (160, 100, 100,  1)  -> 맨앞에 있는 5가 배치사이즈임. (train size 의 배치사이즈를 건드림.)
print(xy_train[0][1].shape) # (160,) 

print("===================================================================================")
print(type(xy_train))    # -> 'keras.preprocessing.image.DirectoryIterator'
print(type(xy_train[0])) # -> turple
print(type(xy_train[0][0])) # numpy.ndarray  
print(type(xy_train[0][1])) # numpy.ndarray


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

hist = model.fit(xy_train, epochs=1000, 
                    steps_per_epoch= 32,
                    validation_data= xy_test,
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
