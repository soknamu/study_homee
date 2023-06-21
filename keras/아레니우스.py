import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 아레니우스 함수 정의
def arrhenius(T, A, Ea):
    R = 8.314 # 기체 상수
    return A * np.exp(-Ea / (R * T))

# 데이터 정의
data = pd.DataFrame([
    [1, 4, 15, 25, 35],
    [2, 4, 15, 25, 35],
    [3, 4, 15, 25, 35],
    [4, 4, 15, 25, 35],
    [5, 4, 15, 25, 35],
    [6, 4, 15, 25, 35],
    [7, 4, 15, 25, 35],
    [8, 4, 15, 25, 35]
], columns=['주차', 'CFU_4', 'CFU_15', 'CFU_25', 'CFU_35'])

# 각 온도별 측정 값에 대한 로그를 취합니다.
for temp in [4, 15, 25, 35]:
    data[f'ln_CFU_{temp}'] = np.log(data[f'CFU_{temp}'])

# 아레니우스 방정식에 대한 curve fitting을 수행하고 최적의 A와 Ea 값을 찾습니다.
popt, pcov = curve_fit(arrhenius, data['주차'].values, data[f'ln_CFU_{temp}'].values)

# 출력
print(f"A: {popt[0]}, Ea: {popt[1]}")
