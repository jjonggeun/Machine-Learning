import numpy as np  # 수치 연산을 위한 라이브러리
import pandas as pd  # 데이터 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리

# 시그모이드 함수 선언
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 시그모이드 함수 적용

# 시그모이드 함수의 미분
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))  # 시그모이드 함수의 미분

# 순전파 함수
def forward_propagation(x_with_dummy, v, w):
    A = v @ x_with_dummy.T  # 입력 데이터와 은닉층 가중치의 선형 결합
    b = sigmoid(A)  # 시그모이드 함수 적용
    b_with_dummy = np.vstack([b, np.ones([1, len(x_with_dummy)])])  # 은닉층 출력에 더미 데이터 추가
    B = w @ b_with_dummy  # 은닉층 출력과 출력층 가중치의 선형 결합
    y_hat = sigmoid(B)  # 최종 출력 값 계산
    return A, b, b_with_dummy, B, y_hat  # 중간 값들과 최종 출력 값 반환

# 역전파 함수
def backward_propagation(x_with_dummy, y_one_hot, A, b, b_with_dummy, B, y_hat, v, w):
    error = y_hat - y_one_hot.T  # 출력 값과 실제 값의 오차
    wmse = (error * sigmoid_derivative(B)) @ b_with_dummy.T / len(x_with_dummy)  # 출력층 가중치의 경사 계산
    vmse = ((w[:, :-1].T @ (error * sigmoid_derivative(B))) * sigmoid_derivative(A)) @ x_with_dummy / len(x_with_dummy)  # 은닉층 가중치의 경사 계산
    return wmse, vmse  # 출력층과 은닉층 가중치의 경사 반환

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

    return train_set, test_set  # 훈련 세트와 테스트 세트 반환

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\NN_data.csv"  # 데이터 파일 경로
temp_data = pd.read_csv(fold_dir)  # CSV 파일을 읽어 데이터프레임으로 저장
temp_data = temp_data.to_numpy()  # 데이터프레임을 numpy 배열로 변환

# 데이터 분할
train_data, test_data = aug_data(temp_data, 0.7, 0.3)  # 데이터를 7:3 비율로 훈련 세트와 테스트 세트로 분할

# 데이터 분리
x_train = train_data[:, :3]  # 훈련 세트의 입력 데이터
y_train = train_data[:, 3].reshape(-1, 1)  # 훈련 세트의 타겟 데이터

x_test = test_data[:, :3]  # 테스트 세트의 입력 데이터
y_test = test_data[:, 3].reshape(-1, 1)  # 테스트 세트의 타겟 데이터

# 입력 속성 수와 출력 클래스 수 추출
M = x_train.shape[1]  # 입력 데이터의 속성 수
output_size = len(np.unique(y_train))  # 출력 클래스 수, 주어진 배열 y_train 내의 고유한 값을 찾아서 반환

# hidden layer의 노드 수
hidden_size = 10  # 은닉층의 노드 수 설정

# weight 초기화
v = np.random.rand(hidden_size, M + 1)  # 입력층에서 은닉층으로의 가중치 초기화
w = np.random.rand(output_size, hidden_size + 1)  # 은닉층에서 출력층으로의 가중치 초기화

# 학습 파라미터 설정
learning_rate = 0.1  # 학습률
epochs = 200  # 학습 에포크 수
batch_size = 5  # 배치 사이즈

# One-Hot Encoding
y_train_one_hot = np.zeros((len(y_train), output_size))  # y 데이터를 One-Hot 인코딩으로 변환할 배열 초기화
for i in range(len(y_train)):
    y_train_one_hot[i, int(y_train[i]) - 1] = 1  # 각 샘플의 y값에 해당하는 위치를 1로 설정

# 데이터에 더미 변수 추가
x_train_with_dummy = np.hstack((x_train, np.ones((len(x_train), 1))))  # x_train 데이터에 더미 변수를 추가
x_test_with_dummy = np.hstack((x_test, np.ones((len(x_test), 1))))  # x_test 데이터에 더미 변수를 추가

# 정확도와 MSE를 저장할 리스트 초기화
accuracy_list = []  # 정확도 기록 리스트
mse_list = []  # MSE 기록 리스트

# 최적의 가중치를 저장할 변수 초기화
best_accuracy = 0  # 최적의 정확도
best_v = v.copy()  # 최적의 입력층-은닉층 가중치
best_w = w.copy()  # 최적의 은닉층-출력층 가중치

# 학습
for epoch in range(epochs):  # 지정된 에포크 수만큼 반복
    # 한 epoch에 대해 배치사이즈만큼의 step 진행
    for i in range(0, len(x_train), batch_size):  # 배치 크기 단위로 훈련 데이터 반복
        # 배치 데이터 추출
        x_batch = x_train_with_dummy[i:i+batch_size]  # 현재 배치에 해당하는 입력 데이터 추출
        y_batch = y_train_one_hot[i:i+batch_size]  # 현재 배치에 해당하는 타겟 데이터 추출
        
        # Forward propagation (순전파)
        A, b, b_with_dummy, B, y_hat = forward_propagation(x_batch, v, w)  # 배치 데이터에 대한 순전파 수행
        
        # Backward propagation (역전파)
        wmse, vmse = backward_propagation(x_batch, y_batch, A, b, b_with_dummy, B, y_hat, v, w)  # 순전파 결과를 바탕으로 역전파 수행
        
        # Update weights (가중치 업데이트)
        w -= learning_rate * wmse  # 출력층 가중치를 경사 하강법을 사용해 업데이트
        v -= learning_rate * vmse  # 은닉층 가중치를 경사 하강법을 사용해 업데이트
    
    # 테스트 데이터에 대해 정확도 계산
    A_test, b_test, b_with_dummy_test, B_test, y_hat_test = forward_propagation(x_test_with_dummy, v, w)  # 테스트 데이터에 대한 순전파 수행
    y_hat_test_index = np.argmax(y_hat_test, axis=0) + 1  # 예측 값에서 최대값의 인덱스를 구함 (1부터 시작하는 레이블)
    test_accuracy = np.mean(y_hat_test_index == y_test.flatten())  # 예측 값과 실제 값의 일치 비율 계산 (정확도)
    
    if test_accuracy > best_accuracy:  # 현재 에포크의 테스트 정확도가 이전보다 높으면
        best_accuracy = test_accuracy  # 최적의 정확도로 갱신
        best_v = np.copy(v)  # 최적의 은닉층 가중치를 저장
        best_w = np.copy(w)  # 최적의 출력층 가중치를 저장

    # 전체 데이터에 대해 정확도 계산
    A_train, b_train, b_with_dummy_train, B_train, y_hat_train = forward_propagation(x_train_with_dummy, v, w)  # 전체 훈련 데이터에 대한 순전파 수행
    predicted_labels = np.argmax(y_hat_train, axis=0) + 1  # 예측 값에서 최대값의 인덱스를 구함 (1부터 시작하는 레이블)
    accuracy = np.mean(predicted_labels == y_train.flatten())  # 예측 값과 실제 값의 일치 비율 계산 (정확도)
    accuracy_list.append(accuracy)  # 현재 에포크의 정확도를 기록
    
    # MSE (Mean Squared Error) 계산
    mse = np.mean((y_hat_train - y_train_one_hot.T) ** 2)  # 예측 값과 실제 값의 제곱 오차의 평균 계산
    mse_list.append(mse)  # 현재 에포크의 MSE를 기록

# 최적의 가중치로 모델 업데이트
v = best_v  # 최적의 은닉층 가중치로 업데이트
w = best_w  # 최적의 출력층 가중치로 업데이트


# confusion matrix 계산
def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))  # 혼동 행렬 초기화
    for i in range(len(y_true)):
        row_index = int(y_pred[i]) - 1  # 예측 값 인덱스
        col_index = int(y_true[i]) - 1  # 실제 값 인덱스
        confusion_matrix[row_index, col_index] += 1  # 혼동 행렬 업데이트
    return confusion_matrix

# 예측 값으로 혼동 행렬 계산
confusion_matrix = compute_confusion_matrix(y_test, y_hat_test_index, output_size)

# 혼동 행렬 출력
print(confusion_matrix)

# 그래프 출력
plt.figure(figsize=(18, 6))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), accuracy_list, label='Accuracy', color='blue')  # 정확도 그래프
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  # y 축 범위 설정

# MSE 그래프
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), mse_list, label='MSE', color='red')  # MSE 그래프
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE over epochs')
plt.legend()
plt.grid()
plt.ylim(0, 1)  # y 축 범위 설정

plt.show()  # 그래프 출력

# 초기 설정

# epochs는 전체 학습 과정에서 데이터셋을 몇 번 반복할지를 결정합니다. 예를 들어, epochs = 200이면 데이터셋을 200번 반복해서 학습합니다.
# batch_size는 한 번에 처리할 데이터 샘플의 수를 나타냅니다. 예를 들어, batch_size = 5이면 5개의 데이터 샘플을 한 번에 처리합니다.
# learning_rate는 학습 속도를 조절하는 하이퍼파라미터로, 가중치 업데이트 시 얼마나 크게 변할지를 결정합니다.
# 데이터 배치 분할

# 전체 데이터를 batch_size 크기로 나누어 배치(batch)를 만듭니다. 예를 들어, len(x_train) = 100이고 batch_size = 5이면 20개의 배치가 만들어집니다

# 순전파 (Forward Propagation)

# 입력 데이터를 은닉층과 출력층을 거쳐 최종 출력 값을 계산하는 과정입니다.
# A는 입력 데이터와 은닉층 가중치의 선형 결합입니다.
# b는 A에 시그모이드 함수를 적용한 은닉층의 출력 값입니다.
# b_with_dummy는 b에 더미 데이터를 추가한 값입니다.
# B는 b_with_dummy와 출력층 가중치의 선형 결합입니다.
# y_hat는 B에 시그모이드 함수를 적용한 최종 출력 값입니다

# 역전파 (Backward Propagation)

# 출력 값과 실제 값의 오차를 계산하고, 이를 통해 가중치의 경사를 구하는 과정입니다.
# error는 출력 값(y_hat)과 실제 값(y_batch)의 차이입니다.
# wmse는 출력층 가중치의 경사로, 출력층 가중치의 업데이트에 사용됩니다.
# vmse는 은닉층 가중치의 경사로, 은닉층 가중치의 업데이트에 사용됩니다.

# 가중치 업데이트 (Weights Update)

# 학습률(learning_rate)을 곱한 경사 값을 빼서 가중치를 업데이트합니다.
# 이는 경사 하강법(Gradient Descent)으로, 오차를 줄이기 위해 가중치를 조정하는 방법입니다.

# 모델 평가 (Evaluation)

# 학습 도중에 테스트 데이터셋을 사용하여 모델의 성능을 평가합니다.
# 테스트 데이터셋을 순전파하여 출력 값을 계산하고, 이를 통해 정확도(accuracy)를 계산합니다.
# 모델이 예측한 값(y_hat_test_index)과 실제 값(y_test)을 비교하여 정확도를 계산합니다.
# 만약 현재 에포크의 테스트 정확도가 이전 에포크보다 높으면, 현재 가중치를 최적 가중치로 저장합니다.

# 정확도 및 MSE 기록 (Logging Accuracy and MSE)

# 훈련 데이터셋을 순전파하여 출력 값을 계산하고, 이를 통해 훈련 정확도와 MSE(Mean Squared Error)를 계산합니다.
# 계산된 정확도와 MSE를 리스트에 기록하여 나중에 학습 과정의 변화를 시각화할 수 있습니다.

