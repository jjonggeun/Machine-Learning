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

# 경사 하강법 함수 정의
def gradient_descent(X, y, alpha, rp):
    # 초기 가중치 랜덤 설정
    w_ = np.random.rand(3, 1) 
    
    w0_history = []  # w0 변화 저장
    w1_history = []  # w1 변화 저장
    w2_history = []  # w2 변화 저장
    accuracy_history = []
    
    for i in range(rp):
        
        z = np.dot(X,w_)
        p = Sigmoid(z)
        p_oz = np.where(p >= 0.5, 1, 0) #예측값 0,1
        dif_cee=np.mean((p-y)*X, axis=0)
        w_ -= alpha * dif_cee.reshape(-1,1)  # 경사 하강법 업데이트
        accuracy=np.sum(p_oz == y)/len(y)
        # w0, w1, w2, MSE 값을 저장
        w0_history.append(w_[0][0])
        w1_history.append(w_[1][0])
        w2_history.append(w_[2][0])
        accuracy_history.append(accuracy)
    
    return w0_history, w1_history, w2_history, w_, accuracy_history



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
train_set, test_set = aug_data(temp_data[0:,:], 0.7, 0.3)



