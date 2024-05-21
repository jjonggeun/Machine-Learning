import numpy as np
import pandas as pd

# 시그모이드 함수 선언
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 미분
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 순전파 함수
def forward_propagation(x_with_dummy, v, w):
    A = v @ x_with_dummy.T
    b = sigmoid(A)
    b_with_dummy = np.vstack([b, np.ones([1, len(x_with_dummy)])])
    B = w @ b_with_dummy
    y_hat = sigmoid(B)
    return A, b, b_with_dummy, B, y_hat

# 역전파 함수
def backward_propagation(x_with_dummy, y_one_hot, A, b, b_with_dummy, B, y_hat, v, w):
    error = y_hat - y_one_hot.T
    wmse = (error * sigmoid_derivative(B)) @ b_with_dummy.T / len(x_with_dummy)
    vmse = ((w[:, :-1].T @ (error * sigmoid_derivative(B))) * sigmoid_derivative(A)) @ x_with_dummy / len(x_with_dummy)
    return wmse, vmse

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\NN_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 데이터 분리
x = temp_data[:, :3]
y = temp_data[:, 3].reshape(-1, 1)

# 입력 속성 수와 출력 클래스 수 추출
M = x.shape[1]
output_size = len(np.unique(y))

# hidden layer의 노드 수
hidden_size = 10

# weight 초기화
v = np.random.rand(hidden_size, M + 1)
w = np.random.rand(output_size, hidden_size + 1)

# 학습 파라미터 설정
learning_rate = 0.01
epochs = 1000

# One-Hot Encoding
y_one_hot = np.zeros((len(y), output_size))
for i in range(len(y)):
    y_one_hot[i, int(y[i]) - 1] = 1

# 데이터에 더미 변수 추가
x_with_dummy = np.hstack((x, np.ones((len(x), 1))))

for epoch in range(epochs):
    # Forward propagation
    A, b, b_with_dummy, B, y_hat = forward_propagation(x_with_dummy, v, w)
    
    # Backward propagation
    wmse, vmse = backward_propagation(x_with_dummy, y_one_hot, A, b, b_with_dummy, B, y_hat, v, w)
    
    # Update weights
    w -= learning_rate * wmse
    v -= learning_rate * vmse
