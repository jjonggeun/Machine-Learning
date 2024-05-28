import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 특징 추출 함수들
#0.65
def feature_1(input_data):
    return np.mean(np.mean(input_data, axis=1))

#0.65
def feature_2(input_data):
    return np.mean(np.var(input_data, axis=1))
#0.7
def feature_3(input_data):
    return np.mean(np.mean(input_data, axis=0))
#0.7
def feature_4(input_data):
    return np.mean(np.var(input_data, axis=0))
#0.63
def feature_5(input_data):
    diagonal = np.diag(input_data)
    return np.mean(diagonal)
#0.5
def feature_6(input_data):
    diagonal = np.diag(input_data)
    return np.var(diagonal)
#0.5
def feature_7(input_data):
    diagonal = np.fliplr(input_data).diagonal()
    return np.mean(diagonal)
#0.5
def feature_8(input_data):
    diagonal = np.fliplr(input_data).diagonal()
    return np.var(diagonal)
#0.74
def feature_9(input_data):
    return np.sum(input_data == 0)
#0.74
def feature_10(input_data):
    return np.sum(input_data != 0)
#0.7
def feature_11(input_data):
    return np.mean(np.sum(input_data, axis=1))
# #0.4
def feature_12(input_data):
    return np.var(np.sum(input_data, axis=0))
#0.74
def feature_13(input_data):
    mean_val = np.mean(input_data)
    return np.sum(input_data > mean_val)
#0.71
def feature_14(input_data):
    mean_val = np.mean(input_data)
    return np.sum(input_data < mean_val)

# 파일에서 특징을 추출하는 함수
def extract_features_from_files(start, end, label, folder_path):
    feature_set = np.array([], dtype='float32')
    feature_set = np.resize(feature_set, (0, 7))  # 14개의 특징을 위해 크기 설정
    
    for i in range(start, end + 1):
        temp_name = f'{folder_path}{label}_{i}.csv'
        temp_image = pd.read_csv(temp_name, header=None).to_numpy(dtype='float32')

        # x0 = feature_1(temp_image)
        # x1 = feature_2(temp_image)
        x2 = feature_3(temp_image)
        x3 = feature_4(temp_image)
        # x4 = feature_5(temp_image)
        # x5 = feature_6(temp_image)
        # x6 = feature_7(temp_image)
        # x7 = feature_8(temp_image)
        x8 = feature_9(temp_image)
        x9 = feature_10(temp_image)
        x10 = feature_11(temp_image)
        # x11 = feature_12(temp_image)
        x12 = feature_13(temp_image)
        x13 = feature_14(temp_image)

        feature = np.array([x2,x3,x8,x9,x10,x12,x13], dtype='float32').reshape(1, -1)
        feature_set = np.concatenate((feature_set, feature), axis=0)
    
    return feature_set

# 파일 경로 설정
folder_path = 'C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\MINIST Data\\'

# 특징 추출
x0_set = extract_features_from_files(1, 500, '0', folder_path)
x1_set = extract_features_from_files(1, 500, '1', folder_path)
x2_set = extract_features_from_files(1, 500, '2', folder_path)

# y값이 0,1,2이므로 zeros로 0을, ones로 1을, ones*2로 2를 만들어서, vstack로 수직으로 쌓아줌
y = np.hstack((np.zeros(500), np.ones(500), 2 * np.ones(500))).reshape(-1, 1)

# 특징 벡터와 레이블을 결합
xf_total = np.vstack([x0_set, x1_set, x2_set])
xy_total = np.hstack([xf_total, y])

# 데이터를 학습용과 테스트용으로 분할
def aug_data(data, train_ratio, test_ratio):
    assert train_ratio + test_ratio == 1

    total_samples = len(data)
    train_size = int(total_samples * train_ratio)

    np.random.shuffle(data)

    train_set = data[:train_size]
    test_set = data[train_size:]

    return train_set, test_set

train_data, test_data = aug_data(xy_total, 0.7, 0.3)

x_train = train_data[:, :7]
y_train = train_data[:, 7].reshape(-1, 1)
x_test = test_data[:, :7]
y_test = test_data[:, 7].reshape(-1, 1)

# 특징 스케일링
x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)
x_train = (x_train - x_train_mean) / x_train_std
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
def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        row_index = int(y_pred[i])
        col_index = int(y_true[i])
        confusion_matrix[row_index, col_index] += 1
    return confusion_matrix

# 모델 학습
M = x_train.shape[1]
output_size = len(np.unique(y_train))

hidden_size = 5

# np.random.seed(42)  # 가중치 초기화를 위한 시드 설정
v = np.random.randn(hidden_size, M + 1) * 0.01
w = np.random.randn(output_size, hidden_size + 1) * 0.01

learning_rate = 0.01  # 학습률 조정
epochs = 100

y_train_one_hot = np.zeros((len(y_train), output_size))
for i in range(len(y_train)):
    y_train_one_hot[i, int(y_train[i])] = 1

y_test_one_hot = np.zeros((len(y_test), output_size))
for i in range(len(y_test)):
    y_test_one_hot[i, int(y_test[i])] = 1

x_train_with_dummy = np.hstack((x_train, np.ones((len(x_train), 1))))
x_test_with_dummy = np.hstack((x_test, np.ones((len(x_test), 1))))
total_samples = len(x_train)

accuracy_list = []
mse_list = []
mse_test_list = []
test_accuracy_list = []

best_accuracy = 0
best_v = np.copy(v)
best_w = np.copy(w)

for epoch in range(epochs):
    for step in range(total_samples):
        A, b, b_with_dummy, B, y_hat = forward_propagation(x_train_with_dummy[step:step+1], v, w)
        wmse, vmse = backward_propagation(x_train_with_dummy[step:step+1], y_train_one_hot[step:step+1], A, b, b_with_dummy, B, y_hat, v, w)
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
confusion_matrix = compute_confusion_matrix(y_test, predicted_labels_test, output_size)
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
