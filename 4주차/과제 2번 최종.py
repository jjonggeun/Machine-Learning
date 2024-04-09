import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 



fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_2_data.csv"
temp_data=pd.read_csv(fold_dir)


# NaN 값을 ''으로 대체하여 300x5 크기를 유지
new_td = temp_data.fillna('')

total_counts = np.zeros([1,5])    #값을 1by5에 저장하라고 하였으므로 1,5의 빈 배열을 선언한다. 전체 데이터 값
sam_counts = np.zeros([1,5])      #위와 같은 내용이고 이 부분에는 샘플링 값들만 넣어 저장


for i in range(5):
    total_counts[0, i] = sum(new_td.iloc[:, i] != '')    #i열에서 nan을 대체한 ''값을 제외하고 남은 데이터의 개수를 구한다
    sam_counts[0, i] = sum(new_td.iloc[:, i] != '') // 2  #열에서 nan을 대체한 ''값을 제외하고 남은 데이터의 개수를 구한다
                                                          #샘플링한 값이고, 2초의 시간이라 하였으므로 2로 나눠준 값이 저장된다.
  

    
fmin = np.min(sam_counts)  #np.mi()을 이용해 괄호 안에 최소값을 구하고자 하는 배열을 넣으면 된다.

DS_data=np.zeros([300,5])  #300by5의 제로스 빈 배열을 선언해준다.

    
for i in range(5):15261111242124124415524
    # 다운샘플링 간격 계산
    # 현재 열의 전체 데이터 수인 total_counts를 30으로 나눈 값을 간격으로 설정
    
    DS_interval = int(sam_counts[0, i] // fmin)
    
    # 빈 문자열을 가진 행을 제외하고 다운샘플링하여 DS_data에 저장
    # nan값을 ''으로 설정하였기에 먼저 데이터프레임이므로 iloc[:,i]를 이용해 현재 신호의 i번째 열 선택
    # 그 다음 조건으로 ''을 제외한 값을 선택하고 이 선택된 부분중 다운샘플링 간격으로 샘플링을 한다. 이 모든 조건을 대괄호[]을 붙여서 사용해주면 한줄에 작성이 가능하다.
    DS_col = new_td.iloc[:, i][new_td.iloc[:, i] != ''][::DS_interval]
    DS_data[:len(DS_col), i] = DS_col

plt.figure(figsize=(10,6))
for i in range(5):
    plt.plot(np.arange(0,2,1/fmin),DS_data[0:60,i])
plt.title("Signal Graphs")
plt.xlabel("Time [s]")
plt.ylabel("Value")
plt.grid(True)
plt.legend(["Signal 1","Signal 2","Signal 3","Signal 4","Signal 5"],loc="upper right")

    
    