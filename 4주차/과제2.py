import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 



fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_2_data.csv"
temp_data=pd.read_csv(fold_dir)


# NaN 값을 ''으로 대체하여 300x5 크기를 유지
new_td = temp_data.fillna('')

total_counts = np.zeros([1,5])
sam_counts = np.zeros([1,5])


for i in range(5):
    total_counts[0, i] = sum(new_td.iloc[:, i] != '')    #여기가 그냥 개수
    sam_counts[0, i] = sum(new_td.iloc[:, i] != '') // 2  #여기가 2로 나눈거
    # 이거는 배열이잖아

    
fmin = np.min(sam_counts)  #여기가 이제 제일작은거 


# 다운샘플링된 데이터를 저장할 배열 초기화
ds_data = np.zeros((fmin, 5))

# 다운샘플링 간격 계산
downsample_interval = total_counts // int(fmin)
# 각 시그널에 대해 다운 샘플링하여 샘플링 주파수를 fmin 값으로 만듭니다. 
#각 시그널마다 샘플링 간격이 다름 샘플링간격을 설정해줘야함
# downsampled_signals = []
# for i in range(5): 
#     signal = new_td.iloc[:, i].values.tolist()  

ds_data=[]
#     downsampled_signal = signal[::int(total_counts[0, i]//fmin)] 
#     downsampled_signals[:,i]=np.append(downsampled_signals, [downsampled_signal]) 

for i in range(5):
    signal = new_td.iloc[:, i].values  # 시그널 데이터
    downsampled_signal = signal[::downsample_interval[i]]  # 다운샘플링된 시그널
    ds_data[:, i] = downsampled_signal[:fmin]  # 다운샘플링된 데이터 저장

plt.figure(figsize=(10,6))
for i in range(5):
    plt.plot(np.arange(0,2,1/fmin),ds_data[0:60,i])
plt.title("Signal Graphs")
plt.xlabel("Time [sec]")
plt.ylabel("Value [V]")
plt.grid(True)
plt.legend(["Signal1","Signal2","Signal3","Signal4","Signal5"],loc="upper right")


    # downsampled_signal = signal[::각 열의 행의 개수를 30으로 나눈 몫] 간격 이렇게 된느거 맞제?
    # 여기서는 tolist 이거 써서 그 행을 리스트로 가져왔으니까 리스트에서 [처음:끝:간격]이렇게 해주잖아
    # 그렇게 해서 짤라고오 붙이면되는거아님?왜 안짤리지

#데이터의 값(행)이 각 시그널(열)에 대해 무작위 개수로 채워져있음 시그널은 총 1~5의 5개가 있고
#for문을 사용해서 그 채워진 값들 중에 ''을 제외한 행의 개수를 계산해야함 총 5개의 값을 1by5 배열에 저장한다.


