import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\6주차\\lin_regression_data_02.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 데이터 분리
x = temp_data[:, :2]  # x 데이터
y = temp_data[:, 2].reshape(-1, 1)  # y 데이터 (reshape으로 2차원 배열로 변경)

# 더미 데이터 추가
x_with_dummy = np.hstack((x, np.ones((len(temp_data), 1))))

# 랜덤 가중치 생성
w = np.random.rand(3, 1) * 6   # rand는 0~1사이 값을 랜덤3개이므로 *6해서 0~6사이값 3개로 지정

# 학습률 설정
alpha = 0.001

# 반복횟수 설정
iterations = 10000

# 경사 하강법 함수 정의
def gradient_descent(X, y, w, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)  # 비용 함수의 변화를 저장하기 위한 배열 초기화
    
    for i in range(iterations):
        # 예측값 계산
        y_pred = np.dot(X, w)
        
        # 오차 계산
        error = y_pred - y
        
        # 경사 하강법 적용
        w = w - alpha * (1/m) * np.dot(X.T, error)
        
        # 비용 함수 계산
        cost_history[i] = np.sum(error ** 2) / (2 * m)
        
    return w, cost_history

# 경사 하강법 적용하여 최적 가중치 및 비용 함수의 변화 추출
optimal_w, cost_history = gradient_descent(x_with_dummy, y, w, alpha, iterations)

# 최적 가중치 출력
print("Optimal Weights (w0, w1, w2):", optimal_w.flatten())

# 비용 함수의 변화 그래프 그리기
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent: Cost per Iteration')
plt.show()

# 3D 산점도 그리기
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, c='blue')

# 축 레이블 설정
ax.set_xlabel('x_0')
ax.set_ylabel('x_1')
ax.set_zlabel('y')

# y^ 그래프용
x0_v = np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 100)
x1_v = np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 100)
x0_v, x1_v = np.meshgrid(x0_v, x1_v)
y_hat_surface = optimal_w[0] * x0_v + optimal_w[1] * x1_v + optimal_w[2]

# 3D 평면 그리기
ax.plot_surface(x0_v, x1_v, y_hat_surface, alpha=0.5, color='red')
plt.grid()
plt.show()
