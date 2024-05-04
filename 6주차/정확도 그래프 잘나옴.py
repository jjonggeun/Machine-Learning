import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\logistic_regression_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(X, y, w):
    m = len(y)
    z = np.dot(X, w)
    p = Sigmoid(z)
    cost = (-1 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return cost

def predict(X, w):
    z = np.dot(X, w)
    p = Sigmoid(z)
    predictions = np.where(p >= 0.5, 1, 0)
    return predictions

def gradient_descent(X, y, alpha, rp):
    w_ = np.random.rand(3, 1) 
    w0_history, w1_history, w2_history, cee_history, accuracy_history = [], [], [], [], []

    for i in range(rp):
        z = np.dot(X, w_)
        p = Sigmoid(z)
        dif_cee = np.mean((p - y) * X, axis=0).reshape(-1, 1)
        w_ -= alpha * dif_cee  

        w0_history.append(w_[0][0])
        w1_history.append(w_[1][0])
        w2_history.append(w_[2][0])
        
        cee = compute_cost(X, y, w_)
        cee_history.append(cee)
        
        predictions = predict(X, w_)
        accuracy = np.sum(predictions == y) / len(y)
        accuracy_history.append(accuracy)
    
    return w0_history, w1_history, w2_history, cee_history, accuracy_history

def aug_data(augmented_data, train_ratio, test_ratio):
    # 데이터를 분할하는 것이므로 분할한 것들의 합이 1이 나와야 함
    assert train_ratio + test_ratio == 1

    # 데이터의 총 개수
    total_samples = len(augmented_data)
    
    # 각 세트의 크기 계산
    train_size = int(total_samples * train_ratio)

    # 데이터를 랜덤하게 섞음
    np.random.shuffle(augmented_data)

    # 데이터 분할
    train_set = augmented_data[:train_size]
    test_set = augmented_data[train_size:]

    return train_set, test_set

# 데이터를 7:3 비율로 분할
train_set, test_set = aug_data(temp_data, 0.7, 0.3)

# 더미 데이터 추가
x1_train = train_set[:, 1].reshape(-1,1)  # 무게 데이터를 Wei저장
x2_train = train_set[:, 2].reshape(-1,1)  # 길이 데이터를 Len에 자ㅓ장
y_train = train_set[:, 3].reshape(-1,1)
x_with_dummy_train = np.hstack((x1_train, x2_train, np.ones((len(x1_train), 1))))

x1_test = test_set[:, 1].reshape(-1,1)  # 무게 데이터를 Wei저장
x2_test = test_set[:, 2].reshape(-1,1)  # 길이 데이터를 Len에 자ㅓ장
y_test = test_set[:, 3].reshape(-1,1)
x_with_dummy_test = np.hstack((x1_test, x2_test, np.ones((len(x1_test), 1))))

# 경사 하강법을 통해 학습
w0_history, w1_history, w2_history, _, _ = gradient_descent(x_with_dummy_train, y_train, 0.3, 4000)

# 가중치 업데이트에 따른 예측 정확도 계산
accuracy_history = []
for i in range(len(w0_history)):
    w = np.array([[w1_history[i]], [w2_history[i]], [w0_history[i]]])
    predictions = np.where(Sigmoid(np.dot(x_with_dummy_test, w)) >= 0.5, 1, 0)
    accuracy = np.sum(predictions == y_test) / len(test_set)
    accuracy_history.append(accuracy)

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 가중치 변화 그래프

plt.plot(w0_history, label='w0')
plt.plot(w1_history, label='w1')
plt.plot(w2_history, label='w2')
plt.xlabel('Iterations')
plt.ylabel('Weights')
plt.title('Changes in Weights over Iterations')
plt.legend()
plt.show()

# 분류 정확도 변화 그래프

plt.plot(accuracy_history)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy over Iterations')

plt.show()
