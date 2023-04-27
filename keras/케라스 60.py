from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화 입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요'
        ]

#긍정 1, 부정 0
label =  np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])

token =  Tokenizer()
token.fit_on_texts(docs)

print(token.word_index) 

# {'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밌어요': 5, '최고에요': 6, '만든': 7, '영화예요': 8, '추천하고': 9, '싶은': 10, '영화': 11, '입니다': 12, '한': 13, '번': 14, 
#  '더': 15, '보고': 16, '싶네요': 17, '글쎄요': 18, '별로에요': 19, '생각보다': 20, '지루해요': 21, 
# '연기가': 22, '어색해요': 23, '재미없어요': 24, '재미없다': 25, '재밌네요': 26, '생기긴': 27, '했어요': 28, '안해요': 29}

print(token.word_counts)
# OrderedDict([('너무', 2), ('재밌어요', 1), ('참', 3), ('최고에요', 1), ('잘', 2), ('만든', 1), ('영화예요', 1), ('추천하고', 1), 
# ('싶은', 1), ('영화', 1), ('입니다', 1), ('한', 1), ('번', 1), ('더', 1), ('보고', 1), ('싶네요', 1), ('글세요', 1), 
# ('별로에요', 1), ('생각보다', 1), ('지루해요', 1), ('연기가', 1), ('어색해요', 1), ('재미없어요', 1), ('재미없다', 1), ('재밌네요', 1), 
# ('환희가', 2), ('생기긴', 1), ('했어요', 1), ('안해요', 1)])

x = token.texts_to_sequences(docs)
print(x) #[[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17], [18], [19], [20, 21], [22, 23], [24], [2, 25], [1, 26], [4, 3, 27, 28], [4, 29]]
print(type(x)) #<class 'list'>
#크기가 다르니깐 padding을 해줌. 중요한 데이터를 뒤에다가 두고, 불필요한 데이터(0)을 앞에다가 둠.

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5)
#pre는 앞에서 부터, maxlen 길이를 늘린다(0을 채워 넣는다.), if maxlen=4로 하면 5개짜리 데이터는 앞이 짤려나감.
print(pad_x)

print(pad_x.shape) #(14, 5)

word_size = len(token.word_index)
print("단어사전의 개수는 : ", word_size) #단어사전의 개수는 :  29

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
#rnn으로 reshape (14,5,1)로 바꿔줌.

model = Sequential()
model.add(Dense(128, input_shape=(5))) # 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(pad_x, label, epochs=1000, callbacks=[es])
