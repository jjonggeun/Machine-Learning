import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x0_set = np.array([], dtype='float32')
x0_set = np.resize(x0_set, (0, 5))
x1_set = np.array([], dtype='float32')
x1_set = np.resize(x0_set, (0, 5))
x2_set = np.array([], dtype='float32')
x2_set = np.resize(x0_set, (0, 5))

# y값이 0,1,2이므로 zeros로 0을, ones로 1을, ones*2로 2를 만들어서, vstack로 수직으로 쌓아줌
y = np.hstack((np.zeros(500), np.ones(500), 2 * np.ones(500))).reshape(-1,1)

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
    assert train_ratio + test_ratio == 1

    total_samples = len(data)
    train_size = int(total_samples * train_ratio)

    np.random.shuffle(data)

    train_set = data[:train_size]
    test_set = data[train_size:]

    return train_set, test_set

# confusion matrix 계산 함수
def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        row_index = int(y_pred[i])
        col_index = int(y_true[i])
        confusion_matrix[row_index, col_index] += 1
    return confusion_matrix

# 가로축 확률밀도함수로 변환 기대값
def feature_1(input_data):
    return np.mean(np.mean(input_data, axis=1))

# 가로축 확률밀도함수로 변환 분산
def feature_2(input_data):
    return np.mean(np.var(input_data, axis=1))

# 세로축 확률밀도함수로 변환 기대값
def feature_3(input_data):
    return np.mean(np.mean(input_data, axis=0))

 # 세로축 확률밀도함수로 변환 분산
def feature_4(input_data):
    return np.mean(np.var(input_data, axis=0))

# 대각원소 배열 추출 밀도함수로 변환 기대값
def feature_5(input_data):
    diagonal = np.diag(input_data)
    return np.mean(diagonal)

for i in range(1, 501):
    temp_name = 'C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\MINIST Data\\' + '0_' + str(i) + '.csv'
    temp_image = pd.read_csv(temp_name, header=None)
    temp_image = temp_image.to_numpy(dtype='float32')
    
    x0 = feature_1(temp_image)
    x1 = feature_2(temp_image)
    x2 = feature_3(temp_image)
    x3 = feature_4(temp_image)
    x4 = feature_5(temp_image)
    
    x0_feature = np.array([x0, x1, x2, x3, x4], dtype='float32').reshape(1, -1)
    x0_set = np.concatenate((x0_set, x0_feature), axis=0)

for i in range(1, 501):
    temp_name = 'C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\MINIST Data\\' + '1_' + str(i) + '.csv'
    temp_image = pd.read_csv(temp_name, header=None)
    temp_image = temp_image.to_numpy(dtype='float32')
    
    x0 = feature_1(temp_image)
    x1 = feature_2(temp_image)
    x2 = feature_3(temp_image)
    x3 = feature_4(temp_image)
    x4 = feature_5(temp_image)
    
    x1_feature = np.array([x0, x1, x2, x3, x4], dtype='float32').reshape(1, -1)
    x1_set = np.concatenate((x1_set, x1_feature), axis=0)
    
for i in range(1, 501):
    temp_name = 'C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\MINIST Data\\' + '2_' + str(i) + '.csv'
    temp_image = pd.read_csv(temp_name, header=None)
    temp_image = temp_image.to_numpy(dtype='float32')
    
    x0 = feature_1(temp_image)
    x1 = feature_2(temp_image)
    x2 = feature_3(temp_image)
    x3 = feature_4(temp_image)
    x4 = feature_5(temp_image)
    
    x2_feature = np.array([x0, x1, x2, x3, x4], dtype='float32').reshape(1, -1)
    x2_set = np.concatenate((x2_set, x2_feature), axis=0)
xf_total = np.vstack([x0_set, x1_set, x2_set])
xy_total = np.hstack([xf_total, y])

train_data, test_data = aug_data(xy_total, 0.7, 0.3)

