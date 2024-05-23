import numpy as np
import pandas as pd

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


fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\MINIST Data\\"

total_data = np.zeros((784, 1500))  # 784행 1500열의 배열을 초기화합니다.

for j in range(3):
    for i in range(500):  
        file_name = f"{j}_{i+1}.csv"  # i를 1부터 시작하도록 수정합니다.
        file_path = fold_dir + file_name
        temp_data = pd.read_csv(file_path, header=None)
        total_data[:, j * 500 + i] = temp_data.values.flatten()
y = np.hstack((np.zeros(500), np.ones(500), 2 * np.ones(500))).reshape(-1,1)
total_y = np.vstack((total_data, y.T))























        
