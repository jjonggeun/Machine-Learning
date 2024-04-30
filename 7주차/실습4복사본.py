import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\5주차\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()    #data를 numpy로 불러와서 슬라이싱등 할 수 있도록 선언한다.

# 데이터 분리
x_data = temp_data[:, 0]  # 무게 데이터를 Wei저장
y_data = temp_data[:, 1]  # 길이 데이터를 Len에 저장

# 가우시안 기저 함수 정의
def gaussian_basis_function(X, K, k):
    x_min = X.min()  # 데이터의 최솟값
    x_max = X.max()  # 데이터의 최댓값
    mu = x_min + ((x_max - x_min) / (K - 1)) * k  # 각 가우시안 함수의 평균 계산
    
    v = (x_max - x_min) / (K - 1)  # 모든 가우스 함수의 분산
    
    simple = (X - mu) / v
    G = np.exp((-1/2) * (simple ** 2))
    
    return G

def calculate_weights(X, Y, K):
    # k의 배열 생성
    k_values = np.arange(K).reshape(-1, 1)
    # K에 따른 가우시안 기저 함수 계산
    X_b = np.column_stack([gaussian_basis_function(X, K, k) for k in k_values])
    # 가중치 계산 (K+1개의 가중치)
    weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y
    return weights

# K_values에 따른 가중치 계산
K_values = [3]
weights_list = [calculate_weights(x_data, y_data, K) for K in K_values]




# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', label='Original Data')  # 원본 데이터
for K, weights in zip(K_values, weights_list):
    x_range = np.linspace(x_data.min(), x_data.max(), 1000)
    y_pred = np.column_stack([gaussian_basis_function(x_range, K, k) for k in range(K)]) @ weights  # 예측값 계산
    plt.plot(x_range, y_pred, label=f'Regression Curve (K={K})')  # 회귀 곡선 그리기
plt.xlabel('Weight')
plt.ylabel('Length')
plt.title('Regression Curves with Different K')
plt.legend()
plt.grid(True)
plt.show()
