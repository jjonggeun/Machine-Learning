import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\Clustering_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

K = 3

np.random.shuffle(temp_data)

# 초기 중심 설정
center = temp_data[:K]
u1, u2, u3 = center[0], center[1], center[2]

iteration_count = 0

while True:
    iteration_count += 1
    # 각 포인트를 가장 가까운 중심에 할당
    c1, c2, c3 = [], [], []
    for point in temp_data:
        d_u1 = np.linalg.norm(point - u1)
        d_u2 = np.linalg.norm(point - u2)
        d_u3 = np.linalg.norm(point - u3)
        
        if d_u1 < d_u2 and d_u1 < d_u3:
            c1.append(point)
        elif d_u2 < d_u1 and d_u2 < d_u3:
            c2.append(point)
        else:
            c3.append(point)
    
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)
    
    # 새로운 중심 계산
    new_u1 = np.mean(c1, axis=0)
    new_u2 = np.mean(c2, axis=0)
    new_u3 = np.mean(c3, axis=0)
    
    # 중심이 변화하지 않으면 종료
    if np.allclose(u1, new_u1) and np.allclose(u2, new_u2) and np.allclose(u3, new_u3):
        break
    
    u1, u2, u3 = new_u1, new_u2, new_u3

# 결과 시각화
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(temp_data[:, 0], temp_data[:, 1], c='blue', s=20, label='Data Points')
plt.scatter(center[:, 0], center[:, 1], c='red', marker='x', s=100, label='Initial Centers')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title('Data Points and Initial Centers')
plt.legend()
plt.grid()

# 초기 중심에 대한 클러스터링 결과
c1_initial, c2_initial, c3_initial = [], [], []
for point in temp_data:
    d_u1 = np.linalg.norm(point - center[0])
    d_u2 = np.linalg.norm(point - center[1])
    d_u3 = np.linalg.norm(point - center[2])
    
    if d_u1 < d_u2 and d_u1 < d_u3:
        c1_initial.append(point)
    elif d_u2 < d_u1 and d_u2 < d_u3:
        c2_initial.append(point)
    else:
        c3_initial.append(point)

c1_initial = np.array(c1_initial)
c2_initial = np.array(c2_initial)
c3_initial = np.array(c3_initial)

plt.subplot(1, 3, 2)
plt.scatter(c1_initial[:, 0], c1_initial[:, 1], c='blue', s=20, label='Cluster 1')
plt.scatter(c2_initial[:, 0], c2_initial[:, 1], c='red', s=20, label='Cluster 2')
plt.scatter(c3_initial[:, 0], c3_initial[:, 1], c='green', s=20, label='Cluster 3')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title('Initial Clustered Data Points')
plt.legend()
plt.grid()

plt.subplot(1, 3, 3)
plt.scatter(c1[:, 0], c1[:, 1], c='blue', s=20, label='Cluster 1')
plt.scatter(c2[:, 0], c2[:, 1], c='red', s=20, label='Cluster 2')
plt.scatter(c3[:, 0], c3[:, 1], c='green', s=20, label='Cluster 3')
plt.scatter([new_u1[0]], [new_u1[1]], c='black', marker='x', s=100, label='Final Center 1')
plt.scatter([new_u2[0]], [new_u2[1]], c='black', marker='x', s=100, label='Final Center 2')
plt.scatter([new_u3[0]], [new_u3[1]], c='black', marker='x', s=100, label='Final Center 3')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title('Final Clustered Data Points')
plt.legend()
plt.grid()

plt.show()

print(f'Number of iterations: {iteration_count}')
