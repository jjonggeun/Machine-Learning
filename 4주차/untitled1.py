import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_2_data.csv"
temp_data = pd.read_csv(fold_dir)

# NaN 값을 None으로 대체하여 300x5 크기를 유지
new_td = temp_data.fillna('')

total_counts = np.zeros((1, 5))
sam_counts = np.zeros((1, 5))
for i in range(5):
    total_counts[0, i] = sum(new_td.iloc[:, i] != '')
    sam_counts[0, i] = sum(new_td.iloc[:, i] != '') // 2

fmin = np.min(sam_counts)

# 다운 샘플링할 간격 계산
downsample_interval = int(total_counts.min() / fmin)

# 각 시그널에 대해 다운 샘플링하여 샘플링 주파수를 fmin 값으로 만듭니다.
downsampled_signals = []
for i in range(5):
    signal = new_td.iloc[:, i].values.tolist()
    downsampled_signal = signal[::downsample_interval]
    downsampled_signals.append(downsampled_signal)

downsampled_data = pd.DataFrame(downsampled_signals).T

print("다운 샘플링된 데이터 크기:", downsampled_data.shape)

plt.figure(figsize=(10,6))
for i in range(5):
    plt.plot(np.arange(0, 2, 1 / fmin), downsampled_signals[i])
plt.title("Signal Graphs")
plt.xlabel("Time [sec]")
plt.ylabel("Value [V]")
plt.grid(True)
plt.legend(["Signal1", "Signal2", "Signal3", "Signal4", "Signal5"], loc="upper right")
plt.show()
