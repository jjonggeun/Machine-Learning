import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\5주차\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()

# 데이터 분리
Wei = temp_data[:, 0]  # 무게 데이터
Len = temp_data[:, 1]  # 길이 데이터

# 그래프 그리기
plt.figure(figsize=(10, 6))

# 원본 데이터 그리기
plt.scatter(Wei, Len, color='darkgreen', label='Original Data', alpha=0.5)

# 데이터 증강 (원본 데이터 점 주변에 20개의 추가 데이터 생성하면서 그래프에 추가)
for i in range(len(Wei)):
    for _ in range(20):
        augmented_Wei = Wei[i] + np.random.normal(0, 0.2)  # 주변에 노이즈 추가
        augmented_Len = Len[i] + np.random.normal(0, 0.2)  # 주변에 노이즈 추가
        plt.scatter(augmented_Wei, augmented_Len, color='red', alpha=0.2, marker='o', label='Augmented Data')  # 증강된 데이터 그래프에 추가
        plt.legend()

plt.xlabel('Weight')
plt.ylabel('Length')
plt.title('Original Data vs. Augmented Data')
plt.legend()
plt.grid(True)
plt.show()
