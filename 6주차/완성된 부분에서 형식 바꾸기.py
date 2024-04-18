from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\6주차\\lin_regression_data_02.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 데이터 분리
x_0 = temp_data[:, 0].reshape(-1, 1)   #여기서 불러온 x값은 x.T값이므로 이 값들을 T해줘야함 그러면 x값
x_0=x_0.T
x_1 = temp_data[:, 1].reshape(-1, 1)
x_1=x_1.T
y = temp_data[:, 2].reshape(-1, 1) #여기서 리쉐입은 50,1이라는 형식을 만들어주려고 한거지 트렌스포즈가 아님.
y = y.T #

# 더미 데이터 추가
dummy_data = np.ones((len(temp_data), 1))
dummy_data = dummy_data.T
# 기존 x 데이터와 더미 데이터를 수직으로 결합하여 새로운 배열 생성 수직으로 결합은 아래로 쌓아 내린다.
x_with_dummy = np.vstack((x_0, x_1, dummy_data))   #여기는 알고보니 트렌스포즈 왜? 교제에 보면 M은 차원 수 , n은 인데긋



# 경사 하강법 함수 정의
def gradient_descent(X, y, alpha, rp):
    # 초기 가중치 랜덤 설정
    w_ = np.random.rand(3, 1) * 6
    
    w0_history = []  # w0 변화 저장
    w1_history = []  # w1 변화 저장
    w2_history = []  # w2 변화 저장
    mse_history = []  # MSE 변화 저장
    
    for i in range(rp):
        
        y_hat = np.dot(X, w_)   #x는 기존 데이터+ 더미, w는 랜덤 3개 , 이게 알고보니까 트렌스포즈다, 1xN이 나오도록 수정 
        error = y_hat - y  
        mse = np.mean(error ** 2)
        w_ -= alpha * np.dot(X.T, error) / len(y)  # 경사 하강법 업데이트
        
        # w0, w1, w2, MSE 값을 저장
        w0_history.append(w_[0][0])
        w1_history.append(w_[1][0])
        w2_history.append(w_[2][0])
        mse_history.append(mse)
    
    return w0_history, w1_history, w2_history, mse_history, w_

# 경사 하강법 실행
w0_history, w1_history, w2_history, mse_history, w_ = gradient_descent(x_with_dummy, y, 0.1, 100)

# 그래프 그리기
fig = plt.figure(figsize=(20, 10))

# w 그래프
ax1 = fig.add_subplot(121)
ax1.plot(w0_history, label='w0')
ax1.plot(w1_history, label='w1')
ax1.plot(w2_history, label='w2')
ax1.legend()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Values')
ax1.set_title('Weights')

# mse 그래프
ax2 = fig.add_subplot(122)
ax2.plot(mse_history, label='MSE')
ax2.legend()
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MSE')
ax2.set_title('Mean Squared Error')

# 3차원 그래프
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_0, x_1, y, c='blue', label='Original Data')

# 예측된 y^ 값을 계산
y_hat_surface = np.dot(x_with_dummy, w_).reshape(x_0.shape)

# 예측값 점으로 표시
ax.scatter(x_0, x_1, y_hat_surface, c='red', label='Predicted Data')

# 가중치 평면 그리기
x0_v, x1_v = np.meshgrid(np.linspace(x_0.min() - 1, x_0.max() + 1, 100), np.linspace(x_1.min() - 1, x_1.max() + 1, 100))
y_hat_surface = w_[0][0] * x0_v + w_[1][0] * x1_v + w_[2][0]
ax.plot_surface(x0_v, x1_v, y_hat_surface, alpha=0.5, color='green', label='Weight Plane')

# 축 레이블 설정
ax.set_xlabel('x_0')
ax.set_ylabel('x_1')
ax.set_zlabel('y')

plt.legend()
plt.show()