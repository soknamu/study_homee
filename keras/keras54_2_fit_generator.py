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

print(xy_train[0][0].shape) # (5, 200, 200,  1)  -> 맨앞에 있는 5가 배치사이즈임. (train size 의 배치사이즈를 건드림.)
print(xy_train[0][1].shape) # (5,) 

print("===================================================================================")
print(type(xy_train))    # -> 'keras.preprocessing.image.DirectoryIterator'
print(type(xy_train[0])) # -> turple
print(type(xy_train[0][0])) # numpy.ndarray  
print(type(xy_train[0][1])) # numpy.ndarray


# 현재 (5, 200, 200 ,1 ) 짜리 데이터가 32덩어리, y도 배치사이의 영향을 받아 5개.

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

#model.fit(xy_train[0][0]) #로 하면 배치사이즈 영향을 받아서 5개밖에 안들어감.
#model.fit(xy_train[:][0], xy_train[:][1], epochs=10)
# -> TypeError: '>=' not supported between instances of 'slice' and 'int'
# 슬라이스하지말고 숫자로 넣어라.

#model.fit(xy_train[0][0], xy_train[0][1], epochs=10) #이거는 가능. Total params: 39,759,105 통배치가능
hist = model.fit_generator(xy_train, epochs=1000, #xy_train에 배치사이즈와 데이터가 들어가있음.
                    steps_per_epoch= 32,  #-> epoch를 나눈것 만큼 짤라서씀. => 전체데이터크기/batch 160/5 = 32 그 이상으로 주면 맛탱이감.
                    validation_data= xy_test,
                    validation_steps=24, # 발리데이터/ batch => 120/5 = 24
                    ) #x데이터 y의 데이터 배치사이즈가 다들어감.

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print(acc) #[0.5687500238418579, 0.737500011920929, 0.856249988079071, 0.9437500238418579, 0.949999988079071, 0.96875, 0.9937499761581421, 0.9750000238418579, 
#0.9937499761581421, 1.0, 
#0.9937499761581421, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9937499761581421, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

print('val_loss : ', val_loss[-1])
print('loss : ', loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])
# -> 마지막 값들만 나옴.

#1. 그림그리기!
from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.plot(loss,label= 'loss')
plt.plot(val_loss,label= 'val_loss')

plt.subplot(1,2,2)
plt.plot(acc,label= 'acc')
plt.plot(val_acc,label= 'val_acc')

plt.show()
