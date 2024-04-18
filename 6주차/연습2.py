from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\6주차\\lin_regression_data_02.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 데이터 분리
x_0 = temp_data[:, 0]
x_1 = temp_data[:, 1]
y = temp_data[:, 2]






# 더미 데이터 추가
dummy_data = np.ones((len(temp_data), 1))

# 기존 x 데이터와 더미 데이터를 수직으로 결합하여 새로운 배열 생성
x_with_dummy = np.hstack((temp_data[:, :2], dummy_data))

w_= np.random.rand(3,1)*6   #rand는 0~1사이 값을 랜덤3개이므로 *6해서 0~6사이값 3개로 지정


x0_v = np.linspace(x_0.min() - 1, x_0.max() + 1, 100)
x1_v = np.linspace(x_1.min() - 1, x_1.max() + 1, 100)

#알파값 설정
a=0.001
#반복횟수 설정
rp=10000

#y^예측값 설정 
y_hat=np.dot(x_with_dummy,w_)






x0_v, x1_v = np.meshgrid(x0_v, x1_v)


# 3D 산점도 그리기
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_0, x_1, y, c='blue')

# 축 레이블 설정
ax.set_xlabel('x_0')
ax.set_ylabel('x_1')
ax.set_zlabel('y')

# y^ 그래프용
y_hat_surface = w_[1]*x0_v + w_[2]*x1_v


# 예측된 y^ 값을 2차원 배열로 변형하여 그리드와 크기를 맞추기
y_hat_surface = np.dot(np.hstack((x0_v.reshape(-1, 1), x1_v.reshape(-1, 1), np.ones_like(x0_v).reshape(-1, 1))), w_).reshape(x0_v.shape)

# 3D 평면 그리기
ax.plot_surface(x0_v, x1_v, y_hat_surface, alpha=0.5, color='red')
plt.grid()
plt.legend(loc='upper left')
plt.show()
