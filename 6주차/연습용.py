import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\6주차\\lin_regression_data_02.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()    #data를 numpy로 불러와서 슬라이싱등 할 수 있도록 선언한다.

# 데이터 분리
XL = temp_data[1:,0]
YL = temp_data[1:,1]
ZL = temp_data[1:,2]

#3차원공간선언
fig = plt.figure()
ax = plt.axes(111,projection='3d')  #3차원에 표시
ax.scatter(XL, YL, ZL, c=ZL, cmap='Greens', marker='o', s=15)


ax.set_xlabel('X') #x축표시
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

n = 100
xmin, xmax, ymin, ymax, zmin, zmax = 0, 20, 0, 20, 0, 50
cmin, cmax = 0, 2

xs = np.array([(xmax - xmin) * np.random.random_sample() + xmin for i in range(n)])
ys = np.array([(ymax - ymin) * np.random.random_sample() + ymin for i in range(n)])
zs = np.array([(zmax - zmin) * np.random.random_sample() + zmin for i in range(n)])
color = np.array([(cmax - cmin) * np.random.random_sample() + cmin for i in range(n)])

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c=color, marker='o', s=15, cmap='Greens')

plt.show()


x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

X, Y = np.meshgrid(x, y)

print(X)
print(Y)