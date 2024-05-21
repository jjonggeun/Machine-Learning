import numpy as np
import pandas as pd

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\NN_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 데이터 분리
x0 = temp_data[:, 0].reshape(-1,1)
x1 = temp_data[:, 1].reshape(-1,1)
x2 = temp_data[:, 2].reshape(-1,1)
y = temp_data[:, 3].reshape(-1,1)

# 시그모이드 함수 선언
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 미분
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 순전파 함수
def forward_propagation(x, v, w):
    A = v @ x.T
    b = sigmoid(A)
    b_with_dummy = np.vstack([b, np.ones([1, len(x)])])
    B = w @ b_with_dummy
    y_hat = sigmoid(B)
    return A, b, b_with_dummy, B, y_hat

# 역전파 함수
def backward_propagation(x, y_target, y_hat, A, b, b_with_dummy, B, v, w):
    error = y_hat - y_target
    wmse = (error * sigmoid_derivative(B)) @ b_with_dummy.T
    vmse = ((w[:, :-1].T @ (error * sigmoid_derivative(B))) * sigmoid_derivative(A)) @ x
    return wmse, vmse

# 입력 데이터 구성
xtotal_data = np.hstack((x0, x1, x2))
dummy_data = np.ones((len(xtotal_data), 1))
x_with_dummy = np.hstack((xtotal_data, dummy_data))

# y_target One-Hot 인코딩
y_target = np.zeros((len(x0), len(np.unique(y))))
for i in range(len(y)):
    y_target[i, int(y[i])-1] = 1

# 입력 속성 수 추출
M = x_with_dummy.shape[1]

# 출력 클래스 수 추출
output_size = y_target.shape[1]

# hidden layer의 노드 수
hidden_size = 10

# weight 초기화
v = np.random.rand(hidden_size, M)
w = np.random.rand(output_size, hidden_size + 1)

# 학습 파라미터 설정
learning_rate = 0.01
epochs = 1000

y_target_t = y_target.T

for epoch in range(epochs):
    # Forward propagation
    A, b, b_with_dummy, B, y_hat = forward_propagation(x_with_dummy, v, w)
    
    # Backward propagation
    wmse, vmse = backward_propagation(x_with_dummy, y_target_t, y_hat, A, b, b_with_dummy, B, v, w)
    
    # Update weights
    w -= learning_rate * wmse
    v -= learning_rate * vmse
