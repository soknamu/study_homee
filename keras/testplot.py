
# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기
# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라

# 제공된 데이터 외 추가 데이터 사용 금지

# 1. 삼성전자 28일(화) 종가 맞추기(점수배점 0.3)
# 2. 삼성전자 29일(수) 아침 시가 맞추기(점수배점 0.7)
# 메일 제목 : 장승원 [삼성 1차] 60,350.07원
# 첨부 파일 : keras53_samsung2_jsw_submit.py
# 첨부 파일 : keras53_samsung4_jsw_submit.py
# 가중치    : _save/samsung/keras53_samsung2_jsw.j5
# 가중치    : _save/samsung/keras53_samsung4_jsw.j5
import tensorflow as tf
import pandas as pd
import random
import numpy as np
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import LSTM
import matplotlib.pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf. random.set_seed(seed)

# 1. data prepare
samsung=pd.read_csv('./_data/시험/삼성전자 주가2.csv', encoding='cp949',index_col=0)
hyundai=pd.read_csv('./_data/시험/현대자동차.csv', encoding='cp949',index_col=0)
samsung=samsung.drop(samsung.columns[4],axis=1)
hyundai=hyundai.drop(hyundai.columns[4],axis=1)

for col in samsung.columns:
    if samsung[col].dtype == 'object':
        samsung[col] = pd.to_numeric(samsung[col].str.replace(',', ''), errors='coerce')
for col in hyundai.columns:
    if hyundai[col].dtype == 'object':
        hyundai[col] = pd.to_numeric(hyundai[col].str.replace(',', ''), errors='coerce')
samsung=samsung.iloc[:180][::-1]
hyundai=hyundai.iloc[:180][::-1]

# '시가', '고가', '저가', '종가', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'
col_drop=[]

samsung=samsung.drop([samsung.columns[i] for i in col_drop],axis=1)
hyundai=hyundai.drop([hyundai.columns[i] for i in col_drop],axis=1)
print(samsung.info())
print(hyundai.info())


for i in range(len(samsung.columns)):
    plt.figure(f'{i}.{samsung.columns[i]}')

    samsung_col = samsung[samsung.columns[i]]
    hyundai_col = hyundai[hyundai.columns[i]]

    samsung_min = samsung_col.min()
    samsung_max = samsung_col.max()
    samsung_diff = samsung_max - samsung_min
    samsung_padding = 0.1 * samsung_diff

    hyundai_min = hyundai_col.min()
    hyundai_max = hyundai_col.max()
    hyundai_diff = hyundai_max - hyundai_min
    hyundai_padding = 0.1 * hyundai_diff

    plt.subplot(1, 2, 1)
    plt.plot(range(len(samsung_col)), np.array(samsung_col[0:len(samsung_col)]))
    plt.yticks(np.arange(samsung_min-samsung_padding, samsung_max+samsung_padding, samsung_diff/4))

    plt.subplot(1, 2, 2)
    plt.plot(range(len(hyundai_col)), np.array(hyundai_col[0:len(hyundai_col)]))
    plt.yticks(np.arange(hyundai_min-hyundai_padding, hyundai_max+hyundai_padding, hyundai_diff/4))

plt.show()

# x=np.concatenate((np.array(samsung),np.array(hyundai)),axis=1)
# y=samsung[samsung.columns[0]]
# print(x.shape)
# plt.plot(range(len(y)),y)
# plt.show()