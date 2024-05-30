import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def k_means_clustering(data, K):
    np.random.shuffle(data)

    # 초기 중심 설정
    center = data[:K]
    u = center.copy()

    iteration_count = 0

    while True:
        iteration_count += 1
        clusters = [[] for _ in range(K)]
        
        # 각 포인트를 가장 가까운 중심에 할당
        for point in data:
            distances = [np.linalg.norm(point - u[j]) for j in range(K)]
            closest_center = np.argmin(distances)
            clusters[closest_center].append(point)
        
        clusters = [np.array(cluster) for cluster in clusters]
        
        # 새로운 중심 계산
        new_u = [np.mean(cluster, axis=0) if len(cluster) > 0 else u[i] for i, cluster in enumerate(clusters)]
        new_u = np.array(new_u)
        
        # 중심이 변화하지 않으면 종료
        if np.allclose(u, new_u):
            break
        
        u = new_u

    return clusters, u, center, iteration_count

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\Clustering_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 원하는 K 값 설정
K = 3 # 이 값을 변경하여 다른 K 값을 테스트할 수 있습니다

# K-means Clustering 실행
clusters, final_centers, initial_centers, iteration_count = k_means_clustering(temp_data, K)

# 결과 시각화
plt.figure(figsize=(18, 6))

# 초기 데이터 및 중심 시각화
plt.subplot(1, 3, 1)
plt.scatter(temp_data[:, 0], temp_data[:, 1], c='blue', s=20, label='Data Points')
plt.scatter(initial_centers[:, 0], initial_centers[:, 1], c='red', marker='x', s=100, label='Initial Centers')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title(f'K={K}: Data Points and Initial Centers')
plt.legend()
plt.grid()

# 초기 중심에 대한 클러스터링 결과 시각화
initial_clusters = [[] for _ in range(K)]
for point in temp_data:
    distances = [np.linalg.norm(point - initial_centers[j]) for j in range(K)]
    closest_center = np.argmin(distances)
    initial_clusters[closest_center].append(point)

initial_clusters = [np.array(cluster) for cluster in initial_clusters]

plt.subplot(1, 3, 2)
colors = plt.cm.rainbow(np.linspace(0, 1, K))
for i, cluster in enumerate(initial_clusters):
    if cluster.size > 0:
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], s=20, label=f'Cluster {i+1}')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title(f'K={K}: Initial Clustered Data Points')
plt.legend()
plt.grid()

# 최종 클러스터링 결과 시각화
plt.subplot(1, 3, 3)
for i, cluster in enumerate(clusters):
    if cluster.size > 0:
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], s=20, label=f'Cluster {i+1}')
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='black', marker='x', s=100, label='Final Centers')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title(f'K={K}: Final Clustered Data Points')
plt.legend()
plt.grid()

plt.show()

print(f'K={K}, Number of iterations: {iteration_count}')
