import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\logistic_regression_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 데이터 분리
x1 = temp_data[:, 1].reshape(-1,1)  # 무게 데이터를 Wei저장
x2 = temp_data[:, 2].reshape(-1,1)  # 길이 데이터를 Len에 자ㅓ장
y = temp_data[:, 3].reshape(-1,1)

# 더미 데이터 추가
total_data=np.hstack((x1,x2,y))
new_x1 = total_data[:,0].reshape(-1,1)
new_x2 = total_data[:,1].reshape(-1,1)
new_y = total_data[:,2].reshape(-1,1)
dummy_data = np.ones((len(total_data), 1))
x_with_dummy = np.hstack((new_x1, new_x2, dummy_data)) 

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
    w0_history, w1_history, w2_history, cee_history, accuracy_history, test_accuracy_history = [], [], [], [], [], []

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

        # 테스트 데이터셋에 대한 정확도 계산
        test_predictions = predict(test_x_with_dummy, w_)
        test_accuracy = np.sum(test_predictions == test_y) / len(test_y)
        test_accuracy_history.append(test_accuracy)
    
    return w0_history, w1_history, w2_history, cee_history, accuracy_history, test_accuracy_history

def aug_data(data, train_ratio, test_ratio):
    # 데이터를 분할하는 것이므로 분할한 것들의 합이 1이 나와야 함
    assert train_ratio + test_ratio == 1

    # 데이터의 총 개수
    total_samples = len(total_data)
    
    # 각 세트의 크기 계산
    train_size = int(total_samples * train_ratio)

    # 데이터를 랜덤하게 섞음
    np.random.shuffle(total_data)

    # 데이터 분할
    train_set = total_data[:train_size]
    test_set = total_data[train_size:]

    return train_set, test_set

train_set, test_set = aug_data(total_data, 0.7, 0.3)

# 훈련 데이터셋으로 가중치 학습
train_x_with_dummy = np.hstack((train_set[:, :2], np.ones((len(train_set), 1))))
train_y = train_set[:, 2].reshape(-1, 1)

# 테스트 데이터셋 준비
test_x_with_dummy = np.hstack((test_set[:, :2], np.ones((len(test_set), 1))))
test_y = test_set[:, 2].reshape(-1, 1)

w0_history, w1_history, w2_history, cee_history, accuracy_history, test_accuracy_history = gradient_descent(train_x_with_dummy, train_y, 0.3, 4000)

# 가중치 변화 그래프
plt.figure(figsize=(8, 6))
plt.plot(w0_history, label='w0')
plt.plot(w1_history, label='w1')
plt.plot(w2_history, label='w2')
plt.xlabel('Iterations')
plt.ylabel('Weights')
plt.title('Changes in Weights over Iterations (Training Data)')
plt.grid()
plt.legend()
plt.show()

# 비용 함수 변화 그래프
plt.figure(figsize=(8, 6))
plt.plot(cee_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations (Training Data)')
plt.grid()
plt.show()

# 분류 정확도 변화 그래프
plt.figure(figsize=(8, 6))
plt.plot(accuracy_history, label='Train Accuracy')
plt.plot(test_accuracy_history, label='Test Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy over Iterations')
plt.legend()
plt.grid()
plt.show()

# 테스트 데이터셋에 대한 최종 정확도 출력
print("Final Test Accuracy:", test_accuracy_history[-1])
