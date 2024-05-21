import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\NN_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 데이터 분리 함수
def aug_data(data, train_ratio, test_ratio):
    assert train_ratio + test_ratio == 1
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    np.random.shuffle(data)
    train_set = data[:train_size]
    test_set = data[train_size:]
    return train_set, test_set

# 데이터 7:3으로 분리
train_data, test_data = aug_data(temp_data, 0.7, 0.3)

# 학습 데이터와 테스트 데이터 분리
x0_train = train_data[:, 0].reshape(-1, 1)
x1_train = train_data[:, 1].reshape(-1, 1)
x2_train = train_data[:, 2].reshape(-1, 1)
y_train = train_data[:, 3].reshape(-1, 1)

x0_test = test_data[:, 0].reshape(-1, 1)
x1_test = test_data[:, 1].reshape(-1, 1)
x2_test = test_data[:, 2].reshape(-1, 1)
y_test = test_data[:, 3].reshape(-1, 1)

# 시그모이드 함수 선언
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 미분
def Sigmoid_derivative(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

# 입력 데이터 
xtotal_train_data = np.hstack((x0_train, x1_train, x2_train))
dummy_train_data = np.ones((len(xtotal_train_data), 1))
x_with_dummy_train = np.hstack((xtotal_train_data, dummy_train_data))

# y_target One-Hot 인코딩 (학습 데이터)
y_target_train = np.zeros((len(x0_train), len(np.unique(y_train))))
for i in range(len(y_train)):
    y_target_train[i, int(y_train[i])-1] = 1

# 입력 속성 수 추출
M = x_with_dummy_train.shape[1]

# 출력 클래스 수 추출
output_size = y_target_train.shape[1]

# hidden layer의 노드 수
hidden_size = 10

# weight 초기화
v = np.random.rand(hidden_size, M)
w = np.random.rand(output_size, hidden_size + 1)

# 학습 파라미터 설정
learning_rate = 0.01
epochs = 1000

y_target_train_t = y_target_train.T

# 학습 과정 추적용 리스트 초기화
train_accuracy_list = []
train_mse_list = []

# 학습 과정
for epoch in range(epochs):
    # Forward propagation
    A = v @ x_with_dummy_train.T
    b = Sigmoid(A)
    b_with_dummy = np.vstack([b, np.ones([1, len(xtotal_train_data)])])
    B = w @ b_with_dummy
    y_hat = Sigmoid(B)
    
    # Error calculation
    error = y_hat - y_target_train_t
    
    # MSE 계산
    mse = np.mean(error**2)
    
    # Backward propagation
    wmse = (error * Sigmoid_derivative(B)) @ b_with_dummy.T
    vmse = ((w[:, :-1].T @ (error * Sigmoid_derivative(B))) * Sigmoid_derivative(A)) @ x_with_dummy_train
    
    # Update weights
    w -= learning_rate * wmse
    v -= learning_rate * vmse
    
    # 정확도 계산
    predictions = np.argmax(y_hat, axis=0)
    true_labels = np.argmax(y_target_train_t, axis=0)
    accuracy = np.mean(predictions == true_labels)
    
    # 리스트에 정확도와 MSE 저장
    train_accuracy_list.append(accuracy)
    train_mse_list.append(mse)
    


# 에포크에 따른 MSE 및 정확도 그래프
plt.figure(figsize=(12, 5))

# MSE 그래프
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_mse_list, label='MSE', color='r')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE over Epochs')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracy_list, label='Accuracy', color='b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
