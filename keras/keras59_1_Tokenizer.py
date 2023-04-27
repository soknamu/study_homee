from tensorflow.keras.preprocessing.text import Tokenizer #전처리 개념.
import numpy as np
text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

#Tokenizer : 단어별로 짜르겠다, 수치화를 지정 해야됨.

token =  Tokenizer()
token.fit_on_texts([text]) #text를 트레인 시킨다. 문장이 여러개가 있을수 있으니 리스트모양으로 만들어줌.

'''
print(token.word_index) 
#{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
#마구가 가장 많아서 1번으로 감. 두번째는 매우2개여서 2번째,

print(token.word_counts)
#OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])
# 단어 뒤에 나온 숫자는 단어의 개수.
'''

x = token.texts_to_sequences([text])
#print(x) 
#print(type(x)) <class 'list'>
#[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.' ->(1, 11) 1행 11열


#그냥 계산을 하면 숫자의 수치에 가치가 있다고 판단해서 원핫 인코딩을 해줘야됨.

# ###################1. to_categorical ##################
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# # [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# print(x.shape) # (1, 11, 9)
# #결국 0을 지우고 (11,1)로 reshape


###################2. pandas ################## 1차원으로 받아드려야됨.
import pandas as pd

#x = np.array(pd.get_dummies(x[0])) #[0]을 안넣으면 주소값의 주소값을 불러오는 것이기 때문에. [0] 사용.
#x = [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] x[0]의 의미가 [3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]
#따라서 [3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]을 원핫해줘야되기 때문에 x[0] 을 사용.
#x = np.array(pd.get_dummies(x[0]))
#x = pd.get_dummies(np.array(x).reshape(11,))
x = pd.get_dummies(np.array(x).ravel()) #flatten이랑 동일.
#3개 다 동일한 코드

# [[0 0 1 0 0 0 0 0]
#  [0 0 0 1 0 0 0 0]
#  [0 1 0 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0]
#  [0 0 0 0 1 0 0 0]
#  [0 0 0 0 0 1 0 0]
#  [0 0 0 0 0 0 1 0]
#  [1 0 0 0 0 0 0 0]
#  [1 0 0 0 0 0 0 0]
#  [1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 1]]

print(x) #TypeError: unhashable type: 'list'
print(x.shape) #(11, 8)
# 오류 : 첫번째 리스트를 넘파이로 바꿔야 된다.
#        두번째 그러면 리스트는 왜 안될까?
# 함수가 pandas Series 또는 DataFrame 개체를 입력으로 예상하기 때문에 
# 따라서 list은 이 함수에 유효한 입력 유형이 아닙니다.

'''
######### 3.사이킷런 onehot #############
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) 
x = np.array(x)
x = x.reshape(-1,1)
x = ohe.fit_transform(x)
print(x.shape)
print(x)
'''

# ######### 3.사이킷런 onehot #############  // 2차원에서 먹힘.
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder() 
# x = ohe.fit_transform(np.array(x).reshape(11,1)).toarray()
# print(x)
# print(x.shape) #(11, 8)

##############지금까지 한것. 문자를 숫자로 만들었고, 숫자들의 순서를 세우기 위해서 원핫 인코더를 함.#########################
