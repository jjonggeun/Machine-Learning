import numpy as np  # 수치 연산을 위한 라이브러리
import pandas as pd  # 데이터 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\Clustering_data.csv"  # 데이터 파일 경로 설정
temp_data = pd.read_csv(fold_dir, header=0)  # CSV 파일을 데이터프레임으로 불러오기, 첫 번째 행을 헤더로 사용
temp_data = temp_data.to_numpy()  # 데이터프레임을 numpy 배열로 변환

# K-means Clustering 함수 정의
def k_means_clustering(data, K, distance_metric='euclidean'):
    np.random.shuffle(data)  # 데이터를 랜덤하게 섞음

    # 초기 중심 설정
    centers = data[:K]  # 데이터에서 처음 K개의 점을 초기 중심으로 선택

    iteration_count = 0  # 반복 횟수 초기화

    while True:
        iteration_count += 1  # 반복 횟수 증가
        clusters = []  # 빈 리스트를 초기화하여 클러스터를 저장
        for _ in range(K):
            clusters.append([])  # K개의 빈 클러스터 리스트 생성

        # 각 포인트를 가장 가까운 중심에 할당
        for point in data:
            distances = []  # 각 중심까지의 거리를 저장할 리스트
            for center in centers:
                if distance_metric == 'euclidean':  # 유클리드 거리 계산, 벡터나 행렬의 노름(norm)을 계산하는 기능을 합니다. 노름은 벡터의 크기 또는 길이를 나타내는 값
                    distances.append(np.linalg.norm(point - center))
                elif distance_metric == 'manhattan':  # 맨해튼 거리 계산, 주어진 배열의 각 요소에 대한 절대값을 계산
                    distances.append(np.sum(np.abs(point - center)))
            closest_center_idx = np.argmin(distances)  # 가장 가까운 중심의 인덱스 찾기
            clusters[closest_center_idx].append(point)  # 해당 클러스터에 포인트 추가

        for i in range(K):
            clusters[i] = np.array(clusters[i])  # 리스트를 numpy 배열로 변환

        # 새로운 중심 계산
        new_centers = []  # 새로운 중심을 저장할 리스트
        for cluster in clusters:
            new_centers.append(np.mean(cluster, axis=0))  # 각 클러스터의 평균을 새로운 중심으로 설정

        # 중심이 변화하지 않으면 종료
        stop = True  # 반복 종료 여부를 결정하는 변수
        for center, new_center in zip(centers, new_centers):  # 이전 중심과 새로운 중심을 비교
            if not np.allclose(center, new_center):  # 중심이 달라졌다면
                stop = False  # 반복을 계속함
                break

        if stop:  # 모든 중심이 변화하지 않았다면
            break  # 반복 종료

        centers = new_centers  # 새로운 중심으로 업데이트

    return clusters, centers, iteration_count  # 클러스터, 최종 중심, 반복 횟수를 반환

# K 값을 사용자로부터 입력받음
K = int(input("Enter the value of K: "))  # 사용자로부터 클러스터의 수 K를 입력받음

# L2 Norm (유클리드 거리) 사용
clusters_l2, final_centers_l2, iteration_count_l2 = k_means_clustering(temp_data, K, distance_metric='euclidean')  # 유클리드 거리를 사용하여 K-means 클러스터링 수행
print(f'L2 Norm: Number of iterations: {iteration_count_l2}')  # 반복 횟수 출력

# L1 Norm (맨해튼 거리) 사용
clusters_l1, final_centers_l1, iteration_count_l1 = k_means_clustering(temp_data, K, distance_metric='manhattan')  # 맨해튼 거리를 사용하여 K-means 클러스터링 수행
print(f'L1 Norm: Number of iterations: {iteration_count_l1}')  # 반복 횟수 출력

# 결과 시각화
plt.figure(figsize=(12, 6))  # 그래프 크기 설정

# L2 Norm 결과
plt.subplot(1, 2, 1)  # 두 개의 서브플롯 중 첫 번째 플롯 설정
colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'brown']  # 각 클러스터의 색상 설정
for i in range(len(clusters_l2)):  # 각 클러스터에 대해
    cluster_data = clusters_l2[i]  # 클러스터의 데이터 포인트를 가져옴
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i % len(colors)], s=20, label=f'Cluster {i+1}')  # 클러스터의 데이터 포인트를 플롯

final_centers_l2_x = []  # 최종 중심의 x 좌표를 저장할 리스트
final_centers_l2_y = []  # 최종 중심의 y 좌표를 저장할 리스트
for center in final_centers_l2:
    final_centers_l2_x.append(center[0])
    final_centers_l2_y.append(center[1])
plt.scatter(final_centers_l2_x, final_centers_l2_y, c='black', marker='x', s=100, label='Final Centers')  # 최종 중심을 플롯

plt.xlabel('X0')  # x축 레이블
plt.ylabel('X1')  # y축 레이블
plt.title('L2 Norm (Euclidean) Final Clustered Data Points')  # 그래프 제목
plt.legend()  # 범례 추가
plt.grid()  # 그리드 추가

# L1 Norm 결과
plt.subplot(1, 2, 2)  # 두 개의 서브플롯 중 두 번째 플롯 설정
for i in range(len(clusters_l1)):  # 각 클러스터에 대해
    cluster_data = clusters_l1[i]  # 클러스터의 데이터 포인트를 가져옴
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i % len(colors)], s=20, label=f'Cluster {i+1}')  # 클러스터의 데이터 포인트를 플롯

final_centers_l1_x = []  # 최종 중심의 x 좌표를 저장할 리스트
final_centers_l1_y = []  # 최종 중심의 y 좌표를 저장할 리스트
for center in final_centers_l1:
    final_centers_l1_x.append(center[0])
    final_centers_l1_y.append(center[1])
plt.scatter(final_centers_l1_x, final_centers_l1_y, c='black', marker='x', s=100, label='Final Centers')  # 최종 중심을 플롯

plt.xlabel('X0')  # x축 레이블
plt.ylabel('X1')  # y축 레이블
plt.title('L1 Norm (Manhattan) Final Clustered Data Points')  # 그래프 제목
plt.legend()  # 범례 추가
plt.grid()  # 그리드 추가

plt.show()  # 그래프 표시

# 최적의 K 도출
def determine_optimal_k(data, max_k, distance_metric='euclidean'):
    sse = []  # SSE (Sum of Squared Errors)를 저장할 리스트
    for k in range(1, max_k + 1):  # 1부터 max_k까지의 K 값에 대해 반복
        clusters, centers, _ = k_means_clustering(data, k, distance_metric)  # K-means 클러스터링 수행
        sse_k = 0  # 현재 K에 대한 SSE 초기화
        for i in range(len(clusters)):  # 각 클러스터에 대해
            for point in clusters[i]:  # 클러스터의 각 포인트에 대해
                if distance_metric == 'euclidean':  # 유클리드 거리
                    sse_k += np.sum((point - centers[i]) ** 2)
                elif distance_metric == 'manhattan':  # 맨해튼 거리
                    sse_k += np.sum(np.abs(point - centers[i]))
        sse.append(sse_k)  # SSE 값을 리스트에 추가
    
    return sse  # SSE 리스트 반환

max_k = 10  # 최적의 K를 찾기 위해 최대 K 값 설정
sse_l2 = determine_optimal_k(temp_data, max_k, distance_metric='euclidean')  # 유클리드 거리를 사용한 SSE 계산
sse_l1 = determine_optimal_k(temp_data, max_k, distance_metric='manhattan')  # 맨해튼 거리를 사용한 SSE 계산

# SSE 그래프
plt.figure(figsize=(12, 6))  # 그래프 크기 설정

plt.subplot(1, 2, 1)  # 두 개의 서브플롯 중 첫 번째 플롯 설정
plt.plot(range(1, max_k + 1), sse_l2, marker='o')  # 유클리드 거리 SSE 플롯
plt.xlabel('Number of clusters (K)')  # x축 레이블
plt.ylabel('Sum of Squared Errors (SSE)')  # y축 레이블
plt.title('Elbow Method for Optimal K (L2 Norm)')  # 그래프 제목
plt.grid()  # 그리드 추가

plt.subplot(1, 2, 2)  # 두 개의 서브플롯 중 두 번째 플롯 설정
plt.plot(range(1, max_k + 1), sse_l1, marker='o')  # 맨해튼 거리 SSE 플롯
plt.xlabel('Number of clusters (K)')  # x축 레이블
plt.ylabel('Sum of Squared Errors (SSE)')  # y축 레이블
plt.title('Elbow Method for Optimal K (L1 Norm)')  # 그래프 제목
plt.grid()  # 그리드 추가

plt.show()  # 그래프 표시


# Elbow Method에서 분산을 사용한다는 것은 각 클러스터 내의 데이터 포인트들이 클러스터 
# 중심으로부터 얼마나 퍼져 있는지를 측정하는 것입니다. 각 클러스터에 대해 중심으로부터의 거리의 제곱합(SSE, Sum of Squared Errors)을 계산하여, 
# K 값을 증가시킴에 따라 SSE가 급격히 감소하는 지점을 찾는 것이 Elbow Method의 핵심입니다.

