import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 시그모이드 함수 선언
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 시그모이드 함수 정의

# 시그모이드 함수의 미분
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))  # 시그모이드 함수의 미분

# 순전파 함수
def forward_propagation(x_with_dummy, v, w):
    A = v @ x_with_dummy.T  # 입력과 가중치 v의 곱
    b = sigmoid(A)  # 활성화 함수 적용
    b_with_dummy = np.vstack([b, np.ones([1, len(x_with_dummy)])])  # 더미 변수를 포함한 b 생성
    B = w @ b_with_dummy  # 은닉층 출력과 가중치 w의 곱
    y_hat = sigmoid(B)  # 활성화 함수 적용
    return A, b, b_with_dummy, B, y_hat  # 순전파 결과 반환

# 역전파 함수
def backward_propagation(x_with_dummy, y_one_hot, A, b, b_with_dummy, B, y_hat, v, w):
    error = y_hat - y_one_hot.T  # 예측값과 실제값의 차이
    wmse = (error * sigmoid_derivative(B)) @ b_with_dummy.T / len(x_with_dummy)  # 출력층 가중치의 변화량 계산
    vmse = ((w[:, :-1].T @ (error * sigmoid_derivative(B))) * sigmoid_derivative(A)) @ x_with_dummy / len(x_with_dummy)  # 은닉층 가중치의 변화량 계산
    return wmse, vmse  # 가중치 변화량 반환

# 데이터 분할 함수
def aug_data(data, train_ratio, test_ratio):
    # 데이터를 분할하는 것이므로 분할한 것들의 합이 1이 나와야 함
    assert train_ratio + test_ratio == 1  # 학습 데이터와 테스트 데이터 비율의 합이 1인지 확인

    # 데이터의 총 개수
    total_samples = len(data)
    
    # 각 세트의 크기 계산
    train_size = int(total_samples * train_ratio)

    # 데이터를 랜덤하게 섞음
    np.random.shuffle(data)

    # 데이터 분할
    train_set = data[:train_size]
    test_set = data[train_size:]

    return train_set, test_set  # 학습 세트와 테스트 세트 반환

# confusion matrix 계산 함수
def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))  # 초기화된 혼동 행렬
    for i in range(len(y_true)):
        row_index = int(y_pred[i]) - 1  # 예측값의 인덱스 계산
        col_index = int(y_true[i]) - 1  # 실제값의 인덱스 계산
        confusion_matrix[row_index, col_index] += 1  # 혼동 행렬 업데이트
    return confusion_matrix  # 혼동 행렬 반환

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\NN_data.csv"
temp_data = pd.read_csv(fold_dir)  # CSV 파일에서 데이터 로드
temp_data = temp_data.to_numpy()  # numpy 배열로 변환

# 데이터 분할
train_data, test_data = aug_data(temp_data, 0.7, 0.3)  # 데이터를 70% 학습 데이터와 30% 테스트 데이터로 분할

# 데이터 분리
x_train = train_data[:, :3]  # 학습 데이터의 입력 값
y_train = train_data[:, 3].reshape(-1, 1)  # 학습 데이터의 출력 값

x_test = test_data[:, :3]  # 테스트 데이터의 입력 값
y_test = test_data[:, 3].reshape(-1, 1)  # 테스트 데이터의 출력 값

# 입력 속성 수와 출력 클래스 수 추출
M = x_train.shape[1]  # 입력 속성 수
output_size = len(np.unique(y_train))  # 출력 클래스 수

# hidden layer의 노드 수
hidden_size = 5  # 은닉층의 노드 수 설정

# weight 초기화
v = np.random.rand(hidden_size, M + 1)  # 은닉층 가중치 초기화
w = np.random.rand(output_size, hidden_size + 1)  # 출력층 가중치 초기화

# 학습 파라미터 설정
learning_rate = 0.1  # 학습률
epochs = 40  # 에포크 수

# One-Hot Encoding
y_train_one_hot = np.zeros((len(y_train), output_size))  # 초기화된 원-핫 인코딩 배열
for i in range(len(y_train)):
    y_train_one_hot[i, int(y_train[i]) - 1] = 1  # 원-핫 인코딩

# 데이터에 더미 변수 추가
x_train_with_dummy = np.hstack((x_train, np.ones((len(x_train), 1))))  # 더미 변수를 포함한 학습 입력 데이터
x_test_with_dummy = np.hstack((x_test, np.ones((len(x_test), 1))))  # 더미 변수를 포함한 테스트 입력 데이터
total_samples = len(x_train)  # 학습 데이터의 총 샘플 수

# 정확도와 MSE를 저장할 리스트 초기화
accuracy_list = []  # 에포크 별 정확도를 저장할 리스트
mse_list = []  # 에포크 별 MSE를 저장할 리스트

# 최적의 가중치를 저장할 변수 초기화
best_accuracy = 0  # 최적의 정확도 초기화
best_v = v  # 최적의 은닉층 가중치 초기화
best_w = w  # 최적의 출력층 가중치 초기화

# 학습
for epoch in range(epochs):
    for step in range(total_samples):
        A, b, b_with_dummy, B, y_hat = forward_propagation(x_train_with_dummy[step:step+1], v, w)  # 순전파
        wmse, vmse = backward_propagation(x_train_with_dummy[step:step+1], y_train_one_hot[step:step+1], A, b, b_with_dummy, B, y_hat, v, w)  # 역전파
        w -= learning_rate * wmse  # 출력층 가중치 업데이트
        v -= learning_rate * vmse  # 은닉층 가중치 업데이트
    
    # 테스트 데이터에 대해 정확도 계산
    A_test, b_test, b_with_dummy_test, B_test, y_hat_test = forward_propagation(x_test_with_dummy, v, w)  # 순전파
    y_hat_test_index = np.argmax(y_hat_test, axis=0) + 1  # 예측값의 클래스 인덱스
    test_accuracy = np.mean(y_hat_test_index == y_test.flatten())  # 테스트 정확도 계산
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy  # 최적의 정확도 갱신
        best_v = np.copy(v)  # 최적의 은닉층 가중치 갱신
        best_w = np.copy(w)  # 최적의 출력층 가중치 갱신

    A_train, b_train, b_with_dummy_train, B_train, y_hat_train = forward_propagation(x_train_with_dummy, v, w)  # 순전파
    predicted_labels = np.argmax(y_hat_train, axis=0) + 1  # 예측값의 클래스 인덱스
    accuracy = np.mean(predicted_labels == y_train.flatten())  # 학습 정확도 계산
    accuracy_list.append(accuracy)  # 학습 정확도 저장
    
    mse = np.mean((y_hat_train - y_train_one_hot.T) ** 2)  # 학습 MSE 계산
    mse_list.append(mse)  # 학습 MSE 저장

# 최적의 가중치로 모델 업데이트
v = best_v  # 최적의 은닉층 가중치 업데이트
w = best_w  # 최적의 출력층 가중치 업데이트

# confusion matrix 계산
confusion_matrix = compute_confusion_matrix(y_test, y_hat_test_index, output_size)  # 혼동 행렬 계산

print(confusion_matrix)  # 혼동 행렬 출력

# 그래프 출력
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), accuracy_list, label='Accuracy', color='blue')  # 정확도 그래프
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), mse_list, label='MSE', color='red')  # MSE 그래프
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE over epochs')
plt.legend()
plt.grid()
plt.ylim(0, 1)

plt.show()  # 그래프 표시
