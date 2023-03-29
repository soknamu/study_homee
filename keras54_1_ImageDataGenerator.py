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
    target_size=(200, 200), #모은 사진들을 확대 또는 축소를 해서 크기를 무조건 200 * 200으로 만듬. (크든 작든)
    batch_size= 5, #5장씩 쓰겠다.
    class_mode= 'binary',#데이터가 2개밖에 없기때문에 (수치화 되서 만들어줌) 카테고리컬은
    color_mode= 'grayscale',#흑백칼라 
    shuffle= True,
) #directory = folder

xy_test = test_datagen.flow_from_directory(
    'd:study_data/_data/brain/test/',
    target_size=(200, 200),
    batch_size= 5,
    class_mode= 'binary',
    color_mode= 'grayscale',
    shuffle= True,
)

print(xy_train)
# train : Found 160 images belonging to 2 classes. (class 2 : 0과 1)

# x의 값은 (160, 200, 200, 1) -> (개수, 가로, 세로, 색깔) 4차원shape @@증폭이 되지않은 상태.@@
# y의 값은 (160,)

# test : Found 120 images belonging to 2 classes.
# x의 값은 (120, 200, 200, 1)
# y의 값은 (120,)
#rgb 는 3, 흑백은 1

#pandas : value_counts
#numpy : np.unique

#<keras.preprocessing.image.DirectoryIterator object at 0x0000014E6D4C7B50> Iterator(반복자=list)
print(xy_train[0])
#print(xy_train.shape) #error

print(len(xy_train))    #32
print(len(xy_train[0])) #2개 -> x와 y 이렇게 하나씩 들어있어서.
print(xy_train[0][0]) #엑스가 다섯개 들어가있다. 
print(xy_train[0][1]) #[1. 0. 0. 0. 1.]

print(xy_train[0][0].shape) # (5, 200, 200,  1)  -> 맨앞에 있는 5가 배치사이즈임. (train size 의 배치사이즈를 건드림.)
print(xy_train[0][1].shape) # (5,) 

print("===================================================================================")
print(type(xy_train))    # -> 'keras.preprocessing.image.DirectoryIterator'
print(type(xy_train[0])) # -> turple
print(type(xy_train[0][0])) # numpy.ndarray  
print(type(xy_train[0][1])) # numpy.ndarray

#결론 : 배치크기 대로 짤려있고, 0번째에 0번째는 x 0번째에 1번째에 y가 들어가있다.
#very good 0부터 시작하기 때문에 32가 아니라 31까지임.