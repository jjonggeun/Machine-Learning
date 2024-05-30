import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\Clustering_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

K = 3

np.random.shuffle(temp_data)

#step 2
center = temp_data[:K]
u1 = center[0,:].reshape(-1,1).T
u2 = center[1,:].reshape(-1,1).T
u3 = center[2,:].reshape(-1,1).T

c1 = []
c2 = []
c3 = []

for i in range(480):
    point = temp_data[i, :].reshape(-1, 1).T
    d_u1 = np.linalg.norm(point - u1)
    d_u2 = np.linalg.norm(point - u2)
    d_u3 = np.linalg.norm(point - u3)
    
    if d_u1 < d_u2 and d_u1 < d_u3:
        c1.append(point)
    elif d_u2 < d_u1 and d_u2 < d_u3:
        c2.append(point)
    else:
        c3.append(point)

c1 = np.array(c1).reshape(-1, 2)
c2 = np.array(c2).reshape(-1, 2)
c3 = np.array(c3).reshape(-1, 2)


#step 3
new_u1 = np.mean(c1, axis=0).reshape(-1,1).T
new_u2 = np.mean(c2, axis=0).reshape(-1,1).T
new_u3 = np.mean(c3, axis=0).reshape(-1,1).T

new_c1 = []
new_c2 = []
new_c3 = []

for i in range(480):
    point = temp_data[i, :].reshape(-1, 1).T
    d_u1 = np.linalg.norm(point - new_u1)
    d_u2 = np.linalg.norm(point - new_u2)
    d_u3 = np.linalg.norm(point - new_u3)
    
    if d_u1 < d_u2 and d_u1 < d_u3:
        new_c1.append(point)
    elif d_u2 < d_u1 and d_u2 < d_u3:
        new_c2.append(point)
    else:
        new_c3.append(point)

new_c1 = np.array(new_c1).reshape(-1, 2)
new_c2 = np.array(new_c2).reshape(-1, 2)
new_c3 = np.array(new_c3).reshape(-1, 2)



plt.figure(figsize=(18, 6))


plt.subplot(1, 3, 1)
plt.scatter(temp_data[:, 0], temp_data[:, 1], c='blue', s=20, label='Data Points')
plt.scatter(center[:, 0], center[:, 1], c='red', marker='x', s=100, label='Initial Centers')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title('Data Points and Initial Centers')
plt.legend()
plt.grid()


plt.subplot(1, 3, 2)
plt.scatter(c1[:, 0], c1[:, 1], c='blue', s=20, label='Cluster 1')
plt.scatter(c2[:, 0], c2[:, 1], c='red', s=20, label='Cluster 2')
plt.scatter(c3[:, 0], c3[:, 1], c='green', s=20, label='Cluster 3')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title('Clustered Data Points')
plt.legend()
plt.grid()

plt.subplot(1, 3, 3)
plt.scatter(new_c1[:, 0], new_c1[:, 1], c='blue', s=20, label='Cluster 1')
plt.scatter(new_c2[:, 0], new_c2[:, 1], c='red', s=20, label='Cluster 2')
plt.scatter(new_c3[:, 0], new_c3[:, 1], c='green', s=20, label='Cluster 3')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title('STEP3 lustered Data Points')
plt.legend()
plt.grid()
plt.show()
