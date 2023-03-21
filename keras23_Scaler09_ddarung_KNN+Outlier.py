import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 데이터 로드
train_df = pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_df = pd.read_csv('./_data/ddarung/test.csv', index_col=0)

# 결측치 대체를 위한 KNN Imputation
imputer = KNNImputer(n_neighbors=5, add_indicator=True)
train_imputed = imputer.fit_transform(train_df)
test_imputed = imputer.transform(test_df)

# Min-Max Scaling
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_imputed)
test_scaled = scaler.transform(test_imputed)

# 이상치 처리를 위한 PCA와 Local Outlier Factor
pca = PCA(n_components=8)
pca_train = pca.fit_transform(train_scaled)
pca_test = pca.transform(test_scaled)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
train_mask = lof.fit_predict(pca_train) != -1
test_mask = lof.predict(pca_test) != -1

# 이상치 제거 후 최종 학습 데이터 생성
train_final = train_scaled[train_mask]
y_train = train_final[:, -1]
X_train = train_final[:, :-1]

# 이상치 제거 후 최종 테스트 데이터 생성
test_final = test_scaled[test_mask]

# 모델 정의
model = Sequential([
Dense(128, input_shape=(8,), activation='relu'),
Dropout(0.3),
Dense(64, activation='relu'),
Dropout(0.2),
Dense(32, activation='relu'),
Dropout(0.1),
Dense(1)
])

# 모델 컴파일
model.compile(loss='mse', optimizer='adam')

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=2)

# 모델 성능 평가
y_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred))

X_test = test_final[:, :-1]
y_pred = model.predict(X_test)

# 제출 파일 생성
submission = pd.read_csv('./_data/ddarung/submission.csv', index_col=0)
submission['count'] = y_pred
submission.to_csv('./_save/ddarung/submission_imputation_scaling_outliers.csv')

# 학습 과정 시각화
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()