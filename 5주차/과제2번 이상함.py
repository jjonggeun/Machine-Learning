import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\5주차\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()  

W0 = 2
W1 = 2.5

Wei = temp_data[:, 0]  # 무게 데이터
Len = temp_data[:, 1]  # 길이 데이터
Wei_new = ()
Len_new = ()


def GDM(x, y, learning_rate=0.0001, epochs=10000):
    """
    경사 하강법을 사용하여 선형 회귀 모델을 최적화하는 함수
    
    Parameters:
    x (numpy.ndarray): 입력 데이터 (무게)
    y (numpy.ndarray): 목표 변수 (길이)
    learning_rate (float): 학습률 (기본값: 0.001)
    epochs (int): 경사 하강법 반복 횟수 (기본값: 1000)
    
    Returns:
    tuple: 최적화된 가중치 (w0, w1)를 반환
    """
    # 초기 가중치 설정
    w0 = 2
    w1 = 2.5
    
    # 경사 하강법을 위한 기록 리스트
    weights_history = []
    mse_history = []
    
    # 경사 하강법 반복
    for epoch in range(epochs):
        # 예측값 계산
        y_pred = w0 * x + w1
        
        # 평균 제곱 오차(MSE) 계산
        mse = np.mean((y_pred - y) ** 2)
        
        # 경사 하강법을 위한 편미분 계산
        gradient_w0 = 2 * np.mean((y_pred - y)*x)
        gradient_w1 = 2 * np.mean((y_pred - y))
        
        # 가중치 업데이트
        w0 -= learning_rate * gradient_w0
        w1 -= learning_rate * gradient_w1
        
        # 현재 가중치와 MSE를 기록
        weights_history.append((w0, w1))
        mse_history.append(mse)
        
    return w0, w1, weights_history, mse_history

# 경사 하강법을 사용하여 선형 회귀 모델 최적화
w0_opt, w1_opt, weights_history, mse_history = GDM(Wei, Len)

print("최적화된 가중치 (w0, w1):", w0_opt, w1_opt)

# 가중치와 MSE의 변화 그래프 그리기
plt.figure(figsize=(12, 6))

# 가중치 변화 그래프
plt.subplot(1, 2, 1)
plt.plot(range(len(weights_history)), [w[0] for w in weights_history], label='w0')
plt.plot(range(len(weights_history)), [w[1] for w in weights_history], label='w1')
plt.xlabel('Epoch')
plt.ylabel('Weights')
plt.title('Weights Update')
plt.legend()

# MSE 변화 그래프
plt.subplot(1, 2, 2)
plt.plot(range(len(mse_history)), mse_history, color='orange')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE)')

plt.tight_layout()
plt.show()
