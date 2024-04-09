import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt 

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\5주차\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()

Wei = temp_data[:, 0]  # 무게 데이터
Len = temp_data[:, 1]  # 길이 데이터

def GDM(x, y, Learn_R, rp, Wei, Len):   # x는 w0, y는 w1, Learn_R은 알파, rp는 반복횟수
   
    Wei_new = []
    Len_new = []
    MSE_new = []
    new_y = []  

    for i in range(rp):
        new_y = x * Wei + y
        error = new_y - Len
        dif_W0 = np.mean(error * Wei)
        dif_W1 = np.mean(error)
        
        x = x - (Learn_R * dif_W0)
        y = y - (Learn_R * dif_W1)
        
        Wei_new.append(x)
        Len_new.append(y)
        
        mse = np.mean(error ** 2)  # 평균 제곱 오차 계산
        MSE_new.append(mse)
        
    return x, y, Wei_new, Len_new, MSE_new, new_y





w0 = 1 
w1 = 2
Learn_R = 0.001
rp = 20000

x, y, Wei_new, Len_new, MSE_new, new_y = GDM(w0, w1, Learn_R, rp, Wei, Len)

# new_y 출력
print("new_y:", new_y)
