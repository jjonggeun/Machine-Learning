import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))

def k_means_clustering(data, K):
    
    # K-means++ 초기화
    n_samples, _ = data.shape
    centers = np.empty((K, data.shape[1]))
    centers[0] = data[np.random.randint(n_samples)]

    for i in range(1, K):
        distances = np.min(np.array([euclidean_distance(data, center.reshape(1, -1)) for center in centers[:i]]), axis=0)
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        new_center = data[np.searchsorted(cumulative_probabilities, r)]
        centers[i] = new_center

    u = centers.copy()

    iteration_count = 0

    while True:
        iteration_count += 1
        clusters = [[] for _ in range(K)]

        for point in data:
            distances = [euclidean_distance(point, u[j].reshape(1, -1)) for j in range(K)]
            closest_center = np.argmin(distances)
            clusters[closest_center].append(point)

        for i in range(K):
            clusters[i] = np.array(clusters[i])

        new_u = np.empty_like(u)
        for i in range(K):
            if len(clusters[i]) > 0:
                new_u[i] = np.mean(clusters[i], axis=0)
            else:
                new_u[i] = u[i]

        if np.allclose(u, new_u):
            break

        u = new_u

    return clusters, u, centers, iteration_count

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\Clustering_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 원하는 K 값 설정
K = 3  # 클러스터의 개수를 설정합니다. 이 값을 변경하여 다른 K 값을 테스트할 수 있습니다.

# K-means Clustering 실행
clusters, final_centers, initial_centers, iteration_count = k_means_clustering(temp_data, K)

# 초기 중심에 대한 클러스터링 결과 계산
dists_initial = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_initial[:, i] = euclidean_distance(temp_data, initial_centers[i].reshape(1, -1))
closest_initial = np.argmin(dists_initial, axis=1)

# 최종 클러스터링 결과 계산
dists_final = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_final[:, i] = euclidean_distance(temp_data, final_centers[i].reshape(1, -1))
closest_final = np.argmin(dists_final, axis=1)

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
colors = plt.cm.rainbow(np.linspace(0, 1, K))
plt.subplot(1, 3, 2)
for k in range(K):
    mask = closest_initial == k
    plt.scatter(temp_data[mask, 0], temp_data[mask, 1], c=colors[k], s=20, label=f'Cluster {k+1}')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title(f'K={K}: Initial Clustered Data Points')
plt.legend()
plt.grid()

# 최종 클러스터링 결과 시각화
plt.subplot(1, 3, 3)
for k in range(K):
    mask = closest_final == k
    plt.scatter(temp_data[mask, 0], temp_data[mask, 1], c=colors[k], s=20, label=f'Cluster {k+1}')
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='black', marker='x', s=100, label='Final Centers')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title(f'K={K}: Final Clustered Data Points')
plt.legend()
plt.grid()

plt.show()

print(f'K={K}, Number of iterations: {iteration_count}')
