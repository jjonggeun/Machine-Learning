import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def k_means_clustering(data, K):
    # 데이터를 무작위로 섞어줍니다.
    np.random.shuffle(data)

    # 초기 중심 설정: 데이터를 무작위로 섞은 후 처음 K개의 데이터를 초기 중심으로 선택합니다.
    center = data[:K]
    u = center.copy()

    iteration_count = 0

    while True:
        iteration_count += 1
        # 각 클러스터를 저장할 빈 리스트를 생성합니다.
        clusters = []
        for _ in range(K):
            clusters.append([])

        # 각 포인트를 가장 가까운 중심에 할당합니다.
        for point in data:
            # 각 중심과의 거리를 계산합니다.
            distances = []
            for j in range(K):
                distances.append(np.linalg.norm(point - u[j]))
            # 가장 가까운 중심의 인덱스를 찾습니다.
            closest_center = np.argmin(distances)
            # 해당 중심에 포인트를 할당합니다.
            clusters[closest_center].append(point)
        
        # 각 클러스터를 numpy 배열로 변환합니다.
        for i in range(K):
            clusters[i] = np.array(clusters[i])
        
        # 새로운 중심을 계산합니다.
        new_u = []
        for i in range(K):
            if len(clusters[i]) > 0:
                new_u.append(np.mean(clusters[i], axis=0))
            else:
                new_u.append(u[i])
        new_u = np.array(new_u)
        
        # 중심이 변화하지 않으면 종료합니다.
        if np.allclose(u, new_u):
            break
        
        # 새로운 중심으로 업데이트합니다.
        u = new_u

    # 최종 클러스터와 중심, 초기 중심, 반복 횟수를 반환합니다.
    return clusters, u, center, iteration_count

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\Clustering_data.csv"
temp_data = pd.read_csv(fold_dir) # CSV 파일을 pandas 데이터프레임으로 불러옵니다.
temp_data = temp_data.to_numpy() # 데이터프레임을 numpy 배열로 변환합니다.

# 원하는 K 값 설정
K = 6# 클러스터의 개수를 설정합니다. 이 값을 변경하여 다른 K 값을 테스트할 수 있습니다.

# K-means Clustering 실행
clusters, final_centers, initial_centers, iteration_count = k_means_clustering(temp_data, K)

# 초기 중심에 대한 클러스터링 결과 계산
dists_initial = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_initial[:, i] = np.linalg.norm(temp_data - initial_centers[i], axis=1)
closest_initial = np.argmin(dists_initial, axis=1)

# 최종 클러스터링 결과 계산
dists_final = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_final[:, i] = np.linalg.norm(temp_data - final_centers[i], axis=1)
closest_final = np.argmin(dists_final, axis=1)

# 결과 시각화
plt.figure(figsize=(18, 6))

# 초기 데이터 및 중심 시각화
plt.subplot(1, 3, 1) # 그래프의 위치를 설정합니다.
plt.scatter(temp_data[:, 0], temp_data[:, 1], c='blue', s=20, label='Data Points') # 데이터 포인트를 파란색으로 표시합니다.
plt.scatter(initial_centers[:, 0], initial_centers[:, 1], c='red', marker='x', s=100, label='Initial Centers') # 초기 중심을 빨간색으로 표시합니다.
plt.xlabel('X0') # X축 레이블을 설정합니다.
plt.ylabel('X1') # Y축 레이블을 설정합니다.
plt.title(f'K={K}: Data Points and Initial Centers') # 그래프 제목을 설정합니다.
plt.legend() # 범례를 표시합니다.
plt.grid() # 격자를 표시합니다.

# 초기 중심에 대한 클러스터링 결과 시각화
colors = plt.cm.rainbow(np.linspace(0, 1, K)) # K 개의 색상을 생성합니다.
plt.subplot(1, 3, 2)
for k in range(K):
    mask = closest_initial == k
    plt.scatter(temp_data[mask, 0], temp_data[mask, 1], c=colors[k], s=20, label=f'Cluster {k+1}')
plt.xlabel('X0') # X축 레이블을 설정합니다.
plt.ylabel('X1') # Y축 레이블을 설정합니다.
plt.title(f'K={K}: Initial Clustered Data Points') # 그래프 제목을 설정합니다.
plt.legend() # 범례를 표시합니다.
plt.grid() # 격자를 표시합니다.

# 최종 클러스터링 결과 시각화
plt.subplot(1, 3, 3)
for k in range(K):
    mask = closest_final == k
    plt.scatter(temp_data[mask, 0], temp_data[mask, 1], c=colors[k], s=20, label=f'Cluster {k+1}')
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='black', marker='x', s=100, label='Final Centers')
plt.xlabel('X0') # X축 레이블을 설정합니다.
plt.ylabel('X1') # Y축 레이블을 설정합니다.
plt.title(f'K={K}: Final Clustered Data Points') # 그래프 제목을 설정합니다.
plt.legend() # 범례를 표시합니다.
plt.grid() # 격자를 표시합니다.

plt.show() # 그래프를 출력합니다.

print(f'K={K}, Number of iterations: {iteration_count}') # K 값과 반복 횟수를 출력합니다.
