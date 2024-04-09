import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\5주차\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()  #위에서 읽은 temp_data는 dataframe형태이므로 이를 numpy배열로 변환하여 slice이용한다.

Wei = np.zeros([50])  # x축 값인 Wei(무게)를 zeros 50으로 선언해준다.
Len = np.zeros([50])  # y축 값인 Len(길이)를 zeros 50으로 선언해준다.


for j in range(50):  # 파일을 열어보면 총 개수가 50개이므로 50번 반복하도록 반복문 실행
    Wei[j] = temp_data[j, 0]  # 위에서 선언한 데이터는 50by2가 되는데 여기서 0번째 열의 데이터만 Wei에 저장
    Len[j] = temp_data[j, 1]  # 1번째 열의 데이터만 Len에 저장해준다.

Wei_Len = sum(Wei * Len)   #Wo과 Wi를 구하기 위해 Wei_Len이라는 것을 따로 선언하여 이를 위에 선언한 Wei와 Len배열의 곱한 값의 합으로 선언
Wei_squared = sum(Wei**2)   #Wo과 Wi를 구하기 위해 Wei_squared를 따로 선언해 이를 Wei의 제곱의 합으로 선언

plt.scatter(Wei, Len)  # 점으로 Wei와 Len을 선언한다.
plt.legend(['RealData']) #위에서 선언한 점의 이름을 화면에 띄운다.


Wei_Sum=sum(Wei)  #Wei의 총 합을 Wei_Sum으로 선언
Len_Sum=sum(Len)    #Len의 총 합을 Len_Sum으로 선언


Wo=((Wei_Len/50)-((Wei_Sum/50)*(Len_Sum/50)))/((Wei_squared/50)-(Wei_Sum/50)**2)  #위에서 선언해준 것들을 이용해 Wo의 식 정리
Wi=(Len_Sum-Wo*Wei_Sum)/50  #Wi의 식을 정리한다. 




# 그래프 그리기
plt.scatter(Wei, Len, label='Real Data')  # 실제 데이터 산점도
plt.xlabel('Weight[g]')  # x축을 무게 Weight[g]으로 설정
plt.ylabel('Length[cm]')  # y축을 길이 Length[cm]로 설정
plt.title('Project 1')  # 그래프 제목 설정
plt.grid(True)  # 격자를 표시해준다. 


x_values = np.arange(Wei.min() - 1, Wei.max() + 1, (Wei.max() + 1 - (Wei.min() - 1)) / 1000)  # x 값 범위 설정
#x의 범위 설정에서 일단 min을 이용해 Wei의 최소값에서 -1을 한 값을 시작으로 하고, max를 이용해 Wei의 최대값에 +1을 이용해 마지막 값을 선언한다
#그 후 간격을 설정하는데 그 간격을 임의로 1000개로 선언하기 위해 처음값과 끝값을 더해 1000으로 나누면 그 간격이 된다. 
y_values = Wo * x_values + Wi  # y^(예측값)의 식을 써주면 y의 범위가 정해진다.
plt.plot(x_values, y_values, color='red', label='Regression Line')  # 회귀선 그리기

fore_value = Wo * Wei + Wi  #위에서 예측값을 위한 Wo과 Wi를 구했으므로 y=Wo x+ Wi 식을 이용해 식을 작성해준다.
mse = (sum((fore_value-Len)**2))/50  #mse의 식을 위에서 정해준 값들로 식을 작성해준다.
print("MSE:", mse) #mse의 값을 표현

plt.legend()  # 범례 표시
plt.show  # 그래프 출력






