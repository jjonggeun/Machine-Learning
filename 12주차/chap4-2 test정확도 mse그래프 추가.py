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

# def feature_1 (input_data):
#     #코드작성
#     return output_value
# x_0_set = np.array([],dtype='float32')
# x_0_set = np.resize(x_0_set, (0.5))
# for i in range(1,501):
#     temp_name='0_' + str(i) + '.csv'
#     temp_image = pd.read.csv(temp_name, header=None)
#     temp_image = temp_image.to_numpy(dtype = 'float32')
    
#     x0=feature_1()
    
# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\MINIST Data\\"
total_data = np.zeros((784, 1500))

for j in range(3):
    for i in range(500):
        file_name = f"{j}_{i+1}.csv"
        file_path = fold_dir + file_name
        temp_data = pd.read_csv(file_path, header=None)
        total_data[:, j * 500 + i] = temp_data.values.flatten()

# y값이 0,1,2이므로 zeros로 0을, ones로 1을, ones*2로 2를 만들어서, vstack로 수직으로 쌓아줌
y = np.hstack((np.zeros(500), np.ones(500), 2 * np.ones(500))).reshape(-1,1)
# 위에서 구한y와 연결
total_y = np.vstack((total_data, y.T))

train_data, test_data = aug_data(total_y.T, 0.7, 0.3)

x_train = train_data[:, :784]
y_train = train_data[:, 784].reshape(-1, 1)

x_test = test_data[:, :784]
y_test = test_data[:, 784].reshape(-1,1)

M = x_train.shape[1]
output_size = len(np.unique(y_train))

hidden_size = 5

v = np.random.rand(hidden_size, M + 1) * 0.01
w = np.random.rand(output_size, hidden_size + 1) * 0.01

learning_rate = 0.1
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
plt.plot(range(1, epochs+1), mse_test_list, label='Test MSE', color='blue')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE over Epochs')
plt.legend()
plt.grid(True)

plt.show()
