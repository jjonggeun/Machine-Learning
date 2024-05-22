import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# 데이터 분할 함수
def aug_data(data, train_ratio, test_ratio):
    # 데이터를 분할하는 것이므로 분할한 것들의 합이 1이 나와야 함
    assert train_ratio + test_ratio == 1

    # 데이터의 총 개수
    total_samples = len(data)
    
    # 각 세트의 크기 계산
    train_size = int(total_samples * train_ratio)

    # 데이터를 랜덤하게 섞음
    np.random.shuffle(data)

    # 데이터 분할
    train_set = data[:train_size]
    test_set = data[train_size:]

    return train_set, test_set


# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\NN_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 데이터 분할
train_data, test_data = aug_data(temp_data, 0.7, 0.3)

# 데이터 분리
x_train = train_data[:, :3]
y_train = train_data[:, 3].reshape(-1, 1)

x_test = test_data[:,:3]
y_test = test_data[:,3].reshape(-1,1)

# 입력 속성 수와 출력 클래스 수 추출
M = x_train.shape[1]
#y_train에서 고유한 (중복되지 않는) 값들을 찾아서 정렬
output_size = len(np.unique(y_train))

# hidden layer의 노드 수
hidden_size = 10

# weight 초기화
v = np.random.rand(hidden_size, M + 1)
w = np.random.rand(output_size, hidden_size + 1)

# 학습 파라미터 설정
learning_rate = 0.1
epochs = 50

# One-Hot Encoding
y_train_one_hot = np.zeros((len(y_train), output_size))
for i in range(len(y_train)):
    y_train_one_hot[i, int(y_train[i]) - 1] = 1

# 데이터에 더미 변수 추가
x_train_with_dummy = np.hstack((x_train, np.ones((len(x_train), 1))))
x_test_with_dummy = np.hstack((x_test, np.ones((len(x_test), 1))))
# 전체 데이터 수
total_samples = len(x_train)

# 정확도와 MSE를 저장할 리스트 초기화
accuracy_list = []
mse_list = []

# 학습
for epoch in range(epochs):
    # 한 epoch에 대해 N step 진행
    for step in range(total_samples):
        # Forward propagation
        A, b, b_with_dummy, B, y_hat = forward_propagation(x_train_with_dummy[step:step+1], v, w)
        
        # Backward propagation
        wmse, vmse = backward_propagation(x_train_with_dummy[step:step+1], y_train_one_hot[step:step+1], A, b, b_with_dummy, B, y_hat, v, w)
        
        # Update weights
        w -= learning_rate * wmse
        v -= learning_rate * vmse
    A_test, b_test, b_with_dummy_test, B_test, y_hat_test = forward_propagation(x_test_with_dummy, v, w)
    
    # 전체 데이터에 대해 정확도 계산
    A_train, b_train, b_with_dummy_train, B_train, y_hat_train = forward_propagation(x_train_with_dummy, v, w)
    predicted_labels = np.argmax(y_hat_train, axis=0) + 1
    y_hat_test_index = np.argmax(y_hat_test, axis=0) + 1
    accuracy = np.mean(predicted_labels == y_train.flatten())
    accuracy_list.append(accuracy)
    
    # MSE 계산
    mse = np.mean((y_hat_train - y_train_one_hot.T) ** 2)
    mse_list.append(mse)

# 행렬 초기화 (1~6까지의 값을 가지므로 6x6 행렬)
confusion_matrix = np.zeros((6, 6))

# y_hat_test_index와 y_test를 이용하여 값 증가시키기
for i in range(len(y_hat_test_index)):
    row_index = int(y_hat_test_index[i]) - 1  # y_hat_test_index의 값이 1~6이므로 0~5로 조정
    col_index = int(y_test[i][0]) - 1          # y_test의 값도 1~6이므로 0~5로 조정
    confusion_matrix[row_index, col_index] += 1

# 결과 행렬 출력
print(confusion_matrix)

# 그래프 출력
plt.figure(figsize=(18, 6))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), accuracy_list, label='Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  # y 축 범위 설정

# MSE 그래프
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), mse_list, label='MSE', color='red')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE over epochs')
plt.legend()
plt.grid()
plt.ylim(0, 1)  # y 축 범위 설정

plt.show()
