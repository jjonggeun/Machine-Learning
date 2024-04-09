import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\5주차\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()



Wei = temp_data[:, 0]  # 무게 데이터
Len = temp_data[:, 1]  # 길이 데이터

w0 = 2
w1 = 4
Learn_R = 0.007
rp = 10000

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

x, y, Wei_new, Len_new, MSE_new = GDM(w0, w1, Learn_R, rp)  #최종 값을 출력함 
#여기서 x는 W0의 최종값, y는 W1의최종 값들을 나타내고 나머지는 위에서 선언한 값들 그대로 나타냄
#GDM함수 안에 들어가는 변수들은 위에서 따로 선언해줌


# # 그래프 그리기
# plt.scatter(Wei, Len, label='Real Data')  # 실제 데이터 산점도
# plt.xlabel('Weight[g]')  # x축을 무게 Weight[g]으로 설정
# plt.ylabel('Length[cm]')  # y축을 길이 Length[cm]로 설정
# plt.title('Project 1')  # 그래프 제목 설정
# plt.grid(True)  # 격자를 표시해준다. 
# x_values = np.arange(Wei.min() - 1, Wei.max() + 1, (Wei.max() + 1 - (Wei.min() - 1)) / 1000)
# #x의 범위 설정에서 일단 min을 이용해 Wei의 최소값에서 -1을 한 값을 시작으로 하고, max를 이용해 Wei의 최대값에 +1을 이용해 마지막 값을 선언한다
# #그 후 간격을 설정하는데 그 간격을 임의로 1000개로 선언하기 위해 처음값과 끝값을 더해 1000으로 나누면 그 간격이 된다.
# y_values = w0 * x_values + w1  # y^(예측값)의 식을 써주면 y의 범위가 정해진다.
# plt.plot(x_values, y_values, color='red', label='Regression Line')  # 회귀선 그리기
# fore_value = x * Wei + y  #위에서 예측값을 위한 Wo과 Wi를 구했으므로 y=Wo x+ Wi 식을 이용해 식을 작성해준다.
# mse = (sum((fore_value-Len)**2))/50  #mse의 식을 위에서 정해준 값들로 식을 작성해준다.
# print("MSE:", mse) #mse의 값을 표현

# plt.legend()  # 범례 표시
# plt.show  # 그래프 출력
x_values = np.arange(Wei.min() - 1, Wei.max() + 1, (Wei.max() + 1 - (Wei.min() - 1)) / 1000)
plt.subplot(1, 3, 1)
plt.plot(Wei_new)
plt.plot(Len_new)
plt.xlabel('step')
plt.ylabel('w0, w1')
plt.title('Optimal Solution')
plt.grid()
plt.legend(["w0", "w1"], loc='upper left')

plt.subplot(1, 3, 2)
plt.plot(MSE_new)
plt.xlabel('step')
plt.ylabel('MSE')
plt.title('E_mse values')
plt.legend(["e_mse"], loc='upper left')
plt.grid()

plt.subplot(1, 3, 3)
plt.scatter(Wei, Len)
yy = x_values * x + y
plt.plot(x_values, yy, c='r')
plt.xlabel('weight')
plt.ylabel('length')
plt.title('total graph')
plt.grid()
plt.legend(["optimal", "training set"], loc='upper left')
plt.show()