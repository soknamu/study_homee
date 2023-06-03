import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #이미지 전처리 데이터
from sklearn.model_selection import train_test_split
path = 'c:/study/_data/cat_dog/PetImages/'
save_path = 'c:/study/_save/cat_dog/'
datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip=True,
    vertical_flip= True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range= 1.2,
    shear_range= 0.7,
    fill_mode= 'nearest', 
)

test_datagen2 = ImageDataGenerator(rescale= 1./1)

augment_size = 2500


cat_dog = datagen.flow_from_directory(path,
            target_size=(144,144),
            batch_size=24998,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

cat_dog_x = cat_dog[0][0]
cat_dog_y = cat_dog[0][1]

cat_dog_x_train, cat_dog_x_test, cat_dog_y_train, cat_dog_y_test = train_test_split(
    cat_dog_x, cat_dog_y, train_size= 0.7, shuffle= True, random_state=1557
)

randidx = np.random.randint(cat_dog_x_train.shape[0], size = augment_size)

x_augmented = cat_dog_x_train[randidx].copy() # x_augmented 에 4만개가 들어감. copy를 통해서 x_train데이터가 덮어씌어지지 않음.
y_augmented = cat_dog_y_train[randidx].copy()

x_augmented = datagen.flow(
    x_augmented, y_augmented,
batch_size=augment_size, shuffle= False).next()[0]

# print(np.max(x_train), np.min(x_train)) #255 0
# print(np.max(x_augmented), np.min(x_augmented)) #1.0 0.0

x_train = np.concatenate((cat_dog_x_train/255., x_augmented), axis=0)
y_train = np.concatenate((cat_dog_y_train, y_augmented), axis=0) #y스케일링하면 큰일남.
x_test = cat_dog_x_test/255.

print(cat_dog_x_train.shape) #(630, 150, 150, 3)
print(cat_dog_x_test.shape)  #(270, 150, 150, 3)
print(cat_dog_y_train.shape) #(630, 2)
print(cat_dog_y_test.shape)  #(270, 2)

np.save(save_path + 'keras58_cat_dog_x_train.npy', arr = cat_dog_x_train)
np.save(save_path + 'keras58_cat_dog_x_test.npy', arr = cat_dog_x_test)
np.save(save_path + 'keras58_cat_dog_y_train.npy', arr = cat_dog_y_train)
np.save(save_path + 'keras58_cat_dog_y_test.npy', arr = cat_dog_y_test)