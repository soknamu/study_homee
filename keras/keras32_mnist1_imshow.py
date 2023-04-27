import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) =mnist.load_data() #다운로드됨. # 텐서플로우는 트레인과 분리되어있음.

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(x_train)
print(y_train)
print(x_train[0]) #60000만장 데이터
print(y_train[0]) # 숫자 5사진.
print(np.unique(y_train, return_counts = True)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 아웃풋10
import matplotlib.pyplot as plt
plt.imshow(x_train[333], 'gray') #이런 비슷한 사진이사진이 6만장.
plt.show()