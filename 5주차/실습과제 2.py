import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\5주차\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()  #위에서 읽은 temp_data는 dataframe형태이므로 이를 numpy배열로 변환하여 slice이용한다.


Wei = temp_data[:, 0]  # 무게 데이터
Len = temp_data[:, 1]  # 길이 데이터


Wei_new = []
Len_new = []
MSE_new = []


def GDM (x, y, Learn_R, rp):    #x는 w0,y는w1, Learn_R은 알파, rp는 반복횟수

    
    for i in range(rp):
        new_y = x*Wei+y
        error = new_y - Len
        dif_W0 = np.mean(error * Wei)
        dif_W1 = np.mean(error)
        
        x = x - (rp*dif_W0)
        y = y - (rp*dif_W1)
        
        
        Wei_new.append(x)
        Len_new.append(y)
        
        mse=error**2
        
        MSE_new.append(mse)
        
    return x, y, Wei_new, Len_new, MSE_new

x,y,Learn_R,rp=GDM(1,2,0.001,1000)
        
        
        
        

    



    
    
    
    
        












































# Wei = np.zeros([50])  # x축 값인 Wei(무게)를 zeros 50으로 선언해준다.
# Len = np.zeros([50])  # y축 값인 Len(길이)를 zeros 50으로 선언해준다.


# for j in range(50):  # 파일을 열어보면 총 개수가 50개이므로 50번 반복하도록 반복문 실행
#     Wei[j] = temp_data[j, 0]  # 위에서 선언한 데이터는 50by2가 되는데 여기서 0번째 열의 데이터만 Wei에 저장
#     Len[j] = temp_data[j, 1]  # 1번째 열의 데이터만 Len에 저장해준다.

# Wei_Len = sum(Wei * Len)   #Wo과 Wi를 구하기 위해 Wei_Len이라는 것을 따로 선언하여 이를 위에 선언한 Wei와 Len배열의 곱한 값의 합으로 선언
# Wei_squared = sum(Wei**2)   #Wo과 Wi를 구하기 위해 Wei_squared를 따로 선언해 이를 Wei의 제곱의 합으로 선언


# Wei_Sum=sum(Wei)  #Wei의 총 합을 Wei_Sum으로 선언
# Len_Sum=sum(Len)    #Len의 총 합을 Len_Sum으로 선언


# Wo=((Wei_Len/50)-((Wei_Sum/50)*(Len_Sum/50)))/((Wei_squared/50)-(Wei_Sum/50)**2)  #위에서 선언해준 것들을 이용해 Wo의 식 정리
# Wi=(Len_Sum-Wo*Wei_Sum)/50  #Wi의 식을 정리한다. 












