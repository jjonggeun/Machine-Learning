import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\logistic_regression_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 데이터 분리
x1 = temp_data[0:, 1].reshape(-1,1)  # 무게 데이터를 Wei저장
x2 = temp_data[:, 2].reshape(-1,1)  # 길이 데이터를 Len에 자ㅓ장
y = temp_data[:,3].reshape(-1,1)

# 더미 데이터 추가
dummy_data = np.ones((len(temp_data), 1))
x_with_dummy = np.hstack((x1, x2, dummy_data)) 

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

w0_history, w1_history, w2_history, cee_history, accuracy_history = gradient_descent(x_with_dummy, y, 0.3, 4000)

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


# 비용 함수 변화 그래프
plt.plot(cee_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')

# 분류 정확도 변화 그래프
plt.plot(accuracy_history)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy over Iterations')


plt.show()

