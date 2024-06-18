import numpy as np  # 수치 연산을 위한 라이브러리
import pandas as pd  # 데이터 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리

# 각 행의 평균 계산 후 전체 평균을 계산. 가로축
def feature_1(input_data):
    return np.mean(np.mean(input_data, axis=1))

# 각 행의 분산을 계산 후 전체 평균을 계산. 가로축
def feature_2(input_data):
    return np.mean(np.var(input_data, axis=1))

# 각 행의 평균 계산 후 전체 평균 계산. 세로축
def feature_3(input_data):
    return np.mean(np.mean(input_data, axis=0))

# 각 행의 분산 계산 후 평균 계산. 세로축
def feature_4(input_data):
    return np.mean(np.var(input_data, axis=0))

# 대각선 요소 평균 계산. 좌상에서 우하로
def feature_5(input_data):
    diagonal = np.diag(input_data)
    return np.mean(diagonal)

# 대각선 요소의 분산을 계산
def feature_6(input_data):
    diagonal = np.diag(input_data)
    return np.var(diagonal)

# 대각선 요소의 평균 계산. 우상에서 좌하로
def feature_7(input_data):
    diagonal = np.fliplr(input_data).diagonal()
    return np.mean(diagonal)

# 대각선 요소의 분산 계산. 우상에서 좌하로
def feature_8(input_data):
    diagonal = np.fliplr(input_data).diagonal()
    return np.var(diagonal)

# 0의 개수 합
def feature_9(input_data):
    return np.sum(input_data == 0)

# 0이 아닌 개수 합
def feature_10(input_data):
    return np.sum(input_data != 0)

# 가로축 합을 계산 후 평균
def feature_11(input_data):
    return np.mean(np.sum(input_data, axis=1))

# 세로축 합을 계산 후 분산
def feature_12(input_data):
    return np.var(np.sum(input_data, axis=0))

# 평균보다 큰 것들의 합
def feature_13(input_data):
    mean_val = np.mean(input_data)
    return np.sum(input_data > mean_val)

# 평균보다 작은 것들의 합
def feature_14(input_data):
    mean_val = np.mean(input_data)
    return np.sum(input_data < mean_val)

# 특징 추출 함수를 불러와 특징 추출하는 함수
def feature_x(start, end, label, folder_path):
    feature_set = np.array([], dtype='float32')
    feature_set = np.resize(feature_set, (0, 8))  # 10개의 특징을 위해 크기 설정
    
    for i in range(start, end + 1):
        temp_name = f'{folder_path}{label}_{i}.csv'
        temp_image = pd.read_csv(temp_name, header=None).to_numpy(dtype='float32')

        x0 = feature_1(temp_image)  
        x8 = feature_9(temp_image)
        x9 = feature_10(temp_image)
        x10 = feature_11(temp_image)
        x12 = feature_13(temp_image)
        x13 = feature_14(temp_image)
        x5 = feature_5(temp_image)  
        x7 = feature_7(temp_image)  

        feature = np.array([x0, x8, x9, x10, x12, x13, x5, x7], dtype='float32').reshape(1, -1)
        feature_set = np.concatenate((feature_set, feature), axis=0)
    
    return feature_set

# 파일 경로 설정
folder_path = 'C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\MINIST Data\\'

# 특징 추출
x0_set = feature_x(1, 500, '0', folder_path)  # 숫자 0에 대한 특징 추출
x1_set = feature_x(1, 500, '1', folder_path)  # 숫자 1에 대한 특징 추출
x2_set = feature_x(1, 500, '2', folder_path)  # 숫자 2에 대한 특징 추출

# y값이 0, 1, 2이므로 zeros로 0을, ones로 1을, ones * 2로 2를 만들어서, vstack로 수직으로 쌓아줌
y = np.hstack((np.zeros(500), np.ones(500), 2 * np.ones(500))).reshape(-1, 1)

# 특징 추출된 0, 1, 2를 결합하고 y값 0~2까지 결합
xf_total = np.vstack([x0_set, x1_set, x2_set])
xy_total = np.hstack([xf_total, y])

# 데이터를 학습용과 테스트용으로 분할
def aug_data(data, train_ratio, test_ratio):
    assert train_ratio + test_ratio == 1  # 학습용과 테스트용 데이터 비율의 합이 1인지 확인

    total_samples = len(data)  # 전체 데이터 개수
    train_size = int(total_samples * train_ratio)  # 학습용 데이터 크기

    np.random.shuffle(data)  # 데이터를 랜덤하게 섞음

    train_set = data[:train_size]  # 학습용 데이터
    test_set = data[train_size:]  # 테스트용 데이터

    return train_set, test_set

train_data, test_data = aug_data(xy_total, 0.7, 0.3)  # 데이터 분할 (70% 학습용, 30% 테스트용)

x_train = train_data[:, :8]  # 학습용 데이터의 특징
y_train = train_data[:, 8].reshape(-1, 1)  # 학습용 데이터의 라벨
x_test = test_data[:, :8]  # 테스트용 데이터의 특징
y_test = test_data[:, 8].reshape(-1, 1)  # 테스트용 데이터의 라벨

#테스트 데이터를 표준화(standardize)하는 과정입니다. 표준화는 데이터의 평균을 0, 표준편차를 1로 변환하여 데이터의 스케일을 조정하는 과정입니다. 
# 이를 통해 각 특징(feature)의 값이 동일한 스케일을 가지게 되어, 학습 알고리즘이 특정 특징에 치우치지 않고 균등하게 학습할 수 있도록 합니다.
# 균등한 스케일링: 표준화를 통해 각 특징의 값이 동일한 스케일을 가지게 되어, 특정 특징이 다른 특징보다 더 큰 영향을 미치지 않도록 합니다.
# 학습 효율성 향상: 많은 머신러닝 알고리즘은 데이터가 동일한 스케일을 가질 때 더 잘 동작합니다. 특히 경사 하강법을 사용하는 알고리즘에서는 표준화가 매우 중요합니다.
# 일관된 평가: 학습 데이터의 통계값(평균과 표준편차)을 사용하여 테스트 데이터를 표준화함으로써, 학습과 테스트 데이터 간의 일관성을 유지하고, 모델의 성능을 공정하게 평가할 수 있습니다.
# 학습 데이터의 각 특징의 평균
x_train_mean = np.mean(x_train, axis=0)
# 표준편차 계산 배열의 표준편차(standard deviation)를 계산하는 함수입니다.
x_train_std = np.std(x_train, axis=0)
# 특징 표준화
x_train = (x_train - x_train_mean) / x_train_std
# 테스트 데이터의 각 특징을 학습 데이터의 평균과 표준편차를 이용한 값들로 표준화
x_test = (x_test - x_train_mean) / x_train_std

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

# confusion matrix 계산 함수
def confusion_matrix_(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        row_index = int(y_pred[i])
        col_index = int(y_true[i])
        confusion_matrix[row_index, col_index] += 1
    return confusion_matrix

# 모델 학습
M = x_train.shape[1]  # 입력 데이터의 특징 수
output_size = len(np.unique(y_train))  # 출력 클래스 수

hidden_size = 10  # 은닉층의 노드 수

v = np.random.randn(hidden_size, M + 1) * 0.01  # 은닉층 가중치 초기화
w = np.random.randn(output_size, hidden_size + 1) * 0.01  # 출력층 가중치 초기화

learning_rate = 0.01  # 학습률
epochs = 200  # 학습 에포크 수
batch_size = 1 # 배치 사이즈

# One-Hot Encoding
y_train_one_hot = np.zeros((len(y_train), output_size))
for i in range(len(y_train)):
    y_train_one_hot[i, int(y_train[i])] = 1

y_test_one_hot = np.zeros((len(y_test), output_size))
for i in range(len(y_test)):
    y_test_one_hot[i, int(y_test[i])] = 1

# 데이터에 더미 변수 추가
x_train_with_dummy = np.hstack((x_train, np.ones((len(x_train), 1))))
x_test_with_dummy = np.hstack((x_test, np.ones((len(x_test), 1))))
total_samples = len(x_train)

# 학습 과정에서의 정확도와 MSE를 저장할 리스트
accuracy_list = []
mse_list = []
mse_test_list = []
test_accuracy_list = []

best_accuracy = 0  # 최고 정확도 초기화
best_v = np.copy(v)  # 최고의 은닉층 가중치 초기화
best_w = np.copy(w)  # 최고의 출력층 가중치 초기화

# for epoch in range(epochs):  # 전체 데이터셋을 여러 번 반복하여 학습하는 외부 루프입니다.
#     for start in range(0, total_samples, batch_size):  # 배치사이즈 단위로 데이터를 나누기 위한 내부 루프입니다.
#         end = start + batch_size  # 배치의 끝 인덱스를 설정합니다.
#         x_batch = x_train_with_dummy[start:end]  # 현재 배치의 입력 데이터입니다.
#         y_batch = y_train_one_hot[start:end]  # 현재 배치의 출력 데이터(One-Hot 인코딩)입니다.

#         # 순전파: 입력 데이터(x_batch)를 통해 은닉층과 출력층의 출력(y_hat)을 계산합니다.
#         A, b, b_with_dummy, B, y_hat = forward_propagation(x_batch, v, w)
        
#         # 역전파: 출력 데이터(y_batch)와 예측 값(y_hat)의 차이를 통해 가중치의 변화량(wmse, vmse)을 계산합니다.
#         wmse, vmse = backward_propagation(x_batch, y_batch, A, b, b_with_dummy, B, y_hat, v, w)
        
#         # 가중치 업데이트: 학습률(learning_rate)과 가중치의 변화량을 사용하여 가중치(w, v)를 업데이트합니다.
#         w -= learning_rate * wmse
#         v -= learning_rate * vmse
    
#     # 학습 데이터에 대한 정확도와 MSE(평균 제곱 오차)를 계산합니다.
#     A_train, b_train, b_with_dummy_train, B_train, y_hat_train = forward_propagation(x_train_with_dummy, v, w)
#     predicted_labels_train = np.argmax(y_hat_train, axis=0)  # 예측 값(y_hat_train)에서 가장 높은 값을 가진 클래스의 인덱스를 가져옵니다.
#     train_accuracy = np.mean(predicted_labels_train == y_train.flatten())  # 예측 값과 실제 값이 일치하는 비율을 계산합니다.
#     train_mse = np.mean((y_hat_train - y_train_one_hot.T) ** 2)  # 예측 값과 실제 값의 차이의 제곱을 평균내어 MSE를 계산합니다.
#     accuracy_list.append(train_accuracy)  # 학습 데이터의 정확도를 기록합니다.
#     mse_list.append(train_mse)  # 학습 데이터의 MSE를 기록합니다.

#     # 테스트 데이터에 대한 정확도와 MSE(평균 제곱 오차)를 계산합니다.
#     A_test, b_test, b_with_dummy_test, B_test, y_hat_test = forward_propagation(x_test_with_dummy, v, w)
#     predicted_labels_test = np.argmax(y_hat_test, axis=0)  # 예측 값(y_hat_test)에서 가장 높은 값을 가진 클래스의 인덱스를 가져옵니다.
#     test_accuracy = np.mean(predicted_labels_test == y_test.flatten())  # 예측 값과 실제 값이 일치하는 비율을 계산합니다.
#     test_mse = np.mean((y_hat_test - y_test_one_hot.T) ** 2)  # 예측 값과 실제 값의 차이의 제곱을 평균내어 MSE를 계산합니다.
#     test_accuracy_list.append(test_accuracy)  # 테스트 데이터의 정확도를 기록합니다.
#     mse_test_list.append(test_mse)  # 테스트 데이터의 MSE를 기록합니다.

#     # 현재 에포크에서의 테스트 데이터 정확도가 최고 정확도보다 높으면, 가중치(v, w)를 최적의 가중치로 업데이트합니다.
#     if test_accuracy > best_accuracy:
#         best_accuracy = test_accuracy
#         best_v = np.copy(v)
#         best_w = np.copy(w)

# # 학습이 완료된 후, 최적의 가중치(best_v, best_w)로 모델의 가중치를 업데이트합니다.
# v = best_v
# w = best_w
for epoch in range(epochs):
    for start in range(0, total_samples, batch_size):
        end = start + batch_size
        x_batch = x_train_with_dummy[start:end]
        y_batch = y_train_one_hot[start:end]

        A, b, b_with_dummy, B, y_hat = forward_propagation(x_batch, v, w)
        wmse, vmse = backward_propagation(x_batch, y_batch, A, b, b_with_dummy, B, y_hat, v, w)
        w -= learning_rate * wmse
        v -= learning_rate * vmse
    
    # Train set accuracy and MSE
    A_train, b_train, b_with_dummy_train, B_train, y_hat_train = forward_propagation(x_train_with_dummy, v, w)
    predicted_labels_train = np.argmax(y_hat_train, axis=0)
    train_accuracy = np.mean(predicted_labels_train == y_train.flatten())
    train_mse = np.mean((y_hat_train - y_train_one_hot.T) ** 2)
    accuracy_list.append(train_accuracy)
    mse_list.append(train_mse)

    # Test set accuracy and MSE
    A_test, b_test, b_with_dummy_test, B_test, y_hat_test = forward_propagation(x_test_with_dummy, v, w)
    predicted_labels_test = np.argmax(y_hat_test, axis=0)
    test_accuracy = np.mean(predicted_labels_test == y_test.flatten())
    test_mse = np.mean((y_hat_test - y_test_one_hot.T) ** 2)
    test_accuracy_list.append(test_accuracy)
    mse_test_list.append(test_mse)

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_v = np.copy(v)
        best_w = np.copy(w)

v = best_v
w = best_w


# Test set의 최종 confusion matrix
confusion_matrix = confusion_matrix_(y_test, predicted_labels_test, output_size)
print("Confusion Matrix:")
print(confusion_matrix)

# 그래프 그리기
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), accuracy_list, label='Train Accuracy', color='blue')
plt.plot(range(1, epochs+1), test_accuracy_list, label='Test Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), mse_list, label='Train MSE', color='red')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE over Epochs')
plt.legend()
plt.grid(True)

plt.show()
