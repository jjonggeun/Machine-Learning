import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\5주차\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()

# 데이터 분리
Wei = temp_data[:, 0]  # 무게 데이터
Len = temp_data[:, 1]  # 길이 데이터

# 초기 설정
w0 = 2
w1 = 4
Learn_R = 0.007
rp = 10000

# 경사 하강법 함수 정의
def GDM(x, y, Learn_R, rp):   # x는 w0, y는 w1, Learn_R은 알파, rp는 반복횟수
   
    Wei_new = []
    Len_new = []
    MSE_new = []

    for i in range(rp):
        new_y = x * Wei + y
        error = new_y - Len
        dif_W0 = np.mean(error * Wei)
        dif_W1 = np.mean(error)
        
        x = x - (Learn_R * dif_W0)
        y = y - (Learn_R * dif_W1)
        mse = np.mean(error ** 2)  # 평균 제곱 오차 계산
        
        Wei_new.append(x)
        Len_new.append(y)
        MSE_new.append(mse)
        
    return x, y, Wei_new, Len_new, MSE_new

# 경사 하강법 수행
x, y, Wei_new, Len_new, MSE_new = GDM(w0, w1, Learn_R, rp) 

# 그래프 그리기

# w0, w1 변화 그래프
plt.subplot(1, 3, 1)   #1행 3열 1번
plt.plot(Wei_new, label='W0 change')  # w0의 변화 그래프
plt.plot(Len_new, label='W1 change')  # w1의 변화 그래프
plt.xlabel('Epoch')
plt.ylabel('W0,W1 Value')
plt.title('Optimal Solution')
plt.grid()
plt.legend()

# MSE 변화 그래프
plt.subplot(1, 3, 2)
plt.plot(MSE_new, label='MSE')  # MSE의 변화 그래프
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Mean Squared Error')
plt.grid()
plt.legend()

# 예측 결과 그래프
plt.subplot(1, 3, 3)
plt.scatter(Wei, Len, label='Real Data')  # 실제 데이터 산점도
x_values = np.arange(Wei.min() - 1, Wei.max() + 1, (Wei.max() + 1 - (Wei.min() - 1)) / 1000)
yy = x * x_values + y  # 회귀선
plt.plot(x_values, yy, c='r', label='Regression Line')  # 회귀선 그래프
plt.xlabel('Weight[g]')
plt.ylabel('Length[cm]')
plt.title('DATA SET')
plt.grid()
plt.legend(loc='upper left')


plt.show()
