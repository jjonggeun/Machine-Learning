# fold_dir: 데이터 파일 경로를 나타내는 문자열 변수.
# temp_data: CSV 파일에서 읽어온 데이터프레임을 numpy 배열로 변환한 변수.
# x_data: temp_data 배열의 첫 번째 열, 무게 데이터.
# y_data: temp_data 배열의 두 번째 열, 길이 데이터.
# gaussian_basis_function(X, K, k): 주어진 X에 대해 K 개의 가우시안 기저 함수 중 k 번째 기저 함수 값을 계산하는 함수.
# X: 입력 데이터.
# K: 가우시안 기저 함수의 개수.
# k: 현재 기저 함수의 인덱스.
# x_min, x_max: 입력 데이터의 최소값과 최대값.
# mu: 가우시안 함수의 평균.
# v: 가우시안 함수의 분산.
# G: 계산된 가우시안 기저 함수 값.
# calculate_weights(X, Y, K): 주어진 데이터 X와 Y, 그리고 K 개의 가우시안 기저 함수를 사용하여 가중치를 계산하는 함수.
# k_values: K 값 배열.
# X_b: 가우시안 기저 함수와 bias를 추가한 입력 데이터.
# weights: 계산된 가중치.
# mse(X, Y, K): 주어진 데이터 X와 Y, 그리고 K 개의 가우시안 기저 함수를 사용하여 MSE를 계산하는 함수.
# mse_value: 계산된 MSE 값.
# K_values: 가우시안 기저 함수의 개수 K 값들의 범위.
# mse_values: 각 K 값에 대한 MSE 값.
# weights_list: 설정된 K 값에 대한 계산된 가중치 리스트.
# x_range: 회귀 곡선을 그릴 때 사용할 x 값 범위.
# y_pred: 회귀 곡선의 예측 값.

# import numpy as np  # 수치 연산을 위한 라이브러리
# import pandas as pd  # 데이터 처리를 위한 라이브러리
# import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리

# # 데이터 불러오기
# fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\5주차\\lin_regression_data_01.csv"  # 데이터 파일 경로
# temp_data = pd.read_csv(fold_dir, header=None)  # CSV 파일을 읽어 데이터프레임으로 저장, 헤더 없음
# temp_data = temp_data.to_numpy()  # 데이터프레임을 numpy 배열로 변환

# # 데이터 분리
# x_data = temp_data[:, 0]  # 무게 데이터를 x_data에 저장
# y_data = temp_data[:, 1]  # 길이 데이터를 y_data에 저장

# # 가우시안 기저 함수 정의
# def gaussian_basis_function(X, K, k):
#     x_min = X.min()  # 데이터의 최솟값
#     x_max = X.max()  # 데이터의 최댓값
#     mu = x_min + ((x_max - x_min) / (K - 1)) * k  # 각 가우시안 함수의 평균 계산
#     v = (x_max - x_min) / (K - 1)  # 모든 가우스 함수의 분산
#     simple = (X - mu) / v  # 가우시안 함수의 입력값 계산
#     G = np.exp((-1/2) * (simple ** 2))  # 가우시안 함수 적용
#     return G  # 계산된 가우시안 기저 함수 값 반환

# # 가중치 계산 함수
# def calculate_weights(X, Y, K): 
#     k_values = np.arange(K).reshape(-1, 1)  # K 값 배열 생성
    
#     # K에 따른 가우시안 기저 함수 계산
#     basis_functions = []
#     for k in k_values:
#         basis_function = gaussian_basis_function(X, K, k)
#         basis_functions.append(basis_function)
#     X_b = np.column_stack(basis_functions)
    
#     X_b = np.hstack([X_b, np.ones((len(X), 1))])  # bias 추가
#     weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y  # 가중치 계산 (정규 방정식 사용)
#     return weights  # 계산된 가중치 반환

# # MSE 계산 함수
# def mse(X, Y, K):
#     k_values = np.arange(K).reshape(-1, 1)  # K 값 배열 생성
    
#     # K에 따른 가우시안 기저 함수 계산
#     basis_functions = []
#     for k in k_values:
#         basis_function = gaussian_basis_function(X, K, k)
#         basis_functions.append(basis_function)
#     X_b = np.column_stack(basis_functions)
    
#     X_b = np.hstack([X_b, np.ones((len(X), 1))])  # bias 추가
#     weights = calculate_weights(X, Y, K)  # 가중치 계산
#     mse_value = np.mean(((X_b @ weights) - Y) ** 2)  # MSE 계산
#     return mse_value  # 계산된 MSE 반환

# # K 값에 따른 MSE 계산
# K_values = np.arange(3, 11)  # K 값 범위 설정 (3부터 10까지)
# mse_values = []
# for K in K_values:
#     mse_value = mse(x_data, y_data, K)
#     mse_values.append(mse_value)

# # MSE 그래프 그리기
# plt.figure(figsize=(10, 6))  # 그래프 크기 설정
# plt.plot(K_values, mse_values, marker='o')  # K 값에 따른 MSE 그래프
# plt.xlabel('Number of Basis Functions (K)')  # x축 레이블
# plt.ylabel('Mean Squared Error (MSE)')  # y축 레이블
# plt.title('MSE vs. Number of Basis Functions')  # 그래프 제목
# plt.grid(True)  # 그리드 추가
# plt.show()  # 그래프 출력

# # 가중치 계산
# K_values = [3,6,9]  # K 값 설정 (10)
# weights_list = []
# for K in K_values:
#     weights = calculate_weights(x_data, y_data, K)
#     weights_list.append(weights)

# # 회귀 곡선 그리기
# plt.figure(figsize=(10, 6))  # 그래프 크기 설정
# plt.scatter(x_data, y_data, color='blue', label='Original Data')  # 원본 데이터 산점도
# for K, weights in zip(K_values, weights_list):  # 각 K 값에 대한 회귀 곡선 그리기
#     x_range = np.linspace(x_data.min(), x_data.max(), 1000)  # x 값 범위 설정
    
#     # K에 따른 가우시안 기저 함수 계산
#     basis_functions = []
#     for k in range(K):
#         basis_function = gaussian_basis_function(x_range, K, k)
#         basis_functions.append(basis_function)
#     y_pred = np.column_stack(basis_functions) @ weights[:-1] + weights[-1]  # bias 추가하여 예측값 계산
    
#     plt.plot(x_range, y_pred, label=f'Regression Curve (K={K})')  # 회귀 곡선 그리기
# plt.xlabel('Weight')  # x축 레이블
# plt.ylabel('Length')  # y축 레이블
# plt.title('Regression Curves with Different K')  # 그래프 제목
# plt.legend()  # 범례 추가
# plt.grid(True)  # 그리드 추가
# plt.show()  # 그래프 출력
import numpy as np  # 수치 연산을 위한 라이브러리
import pandas as pd  # 데이터 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\5주차\\lin_regression_data_01.csv"  # 데이터 파일 경로
temp_data = pd.read_csv(fold_dir, header=None)  # CSV 파일을 읽어 데이터프레임으로 저장, 헤더 없음
temp_data = temp_data.to_numpy()  # 데이터프레임을 numpy 배열로 변환

# 데이터 분리
x_data = temp_data[:, 0]  # 무게 데이터를 x_data에 저장
y_data = temp_data[:, 1]  # 길이 데이터를 y_data에 저장

# 가우시안 기저 함수 정의
def gaussian_basis_function(X, K, k):
    x_min = X.min()  # 데이터의 최솟값
    x_max = X.max()  # 데이터의 최댓값
    mu = x_min + ((x_max - x_min) / (K - 1)) * k  # 각 가우시안 함수의 평균 계산
    v = (x_max - x_min) / (K - 1)  # 모든 가우스 함수의 분산
    simple = (X - mu) / v  # 가우시안 함수의 입력값 계산
    G = np.exp((-1/2) * (simple ** 2))  # 가우시안 함수 적용
    return G  # 계산된 가우시안 기저 함수 값 반환

# 가중치 계산 함수
def calculate_weights(X, Y, K): 
    k_values = np.arange(K).reshape(-1, 1)  # K 값 배열 생성
    
    # K에 따른 가우시안 기저 함수 계산
    basis_functions = []
    for k in k_values:
        basis_function = gaussian_basis_function(X, K, k)
        basis_functions.append(basis_function)
    X_b = np.column_stack(basis_functions)
    
    X_b = np.hstack([X_b, np.ones((len(X), 1))])  # bias 추가
    weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y  # 가중치 계산 (정규 방정식 사용)
    return weights  # 계산된 가중치 반환

# MSE 계산 함수
def mse(X, Y, K):
    k_values = np.arange(K).reshape(-1, 1)  # K 값 배열 생성
    
    # K에 따른 가우시안 기저 함수 계산
    basis_functions = []
    for k in k_values:
        basis_function = gaussian_basis_function(X, K, k)
        basis_functions.append(basis_function)
    X_b = np.column_stack(basis_functions)
    
    X_b = np.hstack([X_b, np.ones((len(X), 1))])  # bias 추가
    weights = calculate_weights(X, Y, K)  # 가중치 계산
    mse_value = np.mean(((X_b @ weights) - Y) ** 2)  # MSE 계산
    return mse_value  # 계산된 MSE 반환

# K 값에 따른 MSE 계산
K_values = np.arange(3, 11)  # K 값 범위 설정 (3부터 10까지)
mse_values = []
for K in K_values:
    mse_value = mse(x_data, y_data, K)
    mse_values.append(mse_value)

# MSE 그래프 그리기
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
plt.plot(K_values, mse_values, marker='o')  # K 값에 따른 MSE 그래프
plt.xlabel('Number of Basis Functions (K)')  # x축 레이블
plt.ylabel('Mean Squared Error (MSE)')  # y축 레이블
plt.title('MSE vs. Number of Basis Functions')  # 그래프 제목
plt.grid(True)  # 그리드 추가
plt.show()  # 그래프 출력

# 최적의 K 값 찾기
optimal_K = K_values[np.argmin(mse_values)]  # MSE 값이 최소인 K 값을 최적의 K 값으로 설정
print(f'The optimal number of basis functions (K) is: {optimal_K}')  # 최적의 K 값 출력

# 최적의 K 값을 사용하여 가중치 계산
optimal_weights = calculate_weights(x_data, y_data, optimal_K)

# 최적의 K 값을 사용하여 회귀 곡선 그리기
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
plt.scatter(x_data, y_data, color='blue', label='Original Data')  # 원본 데이터 산점도

x_range = np.linspace(x_data.min(), x_data.max(), 1000)  # x 값 범위 설정

# 최적의 K 값을 사용한 가우시안 기저 함수 계산
basis_functions = []
for k in range(optimal_K):
    basis_function = gaussian_basis_function(x_range, optimal_K, k)
    basis_functions.append(basis_function)
y_pred = np.column_stack(basis_functions) @ optimal_weights[:-1] + optimal_weights[-1]  # bias 추가하여 예측값 계산

plt.plot(x_range, y_pred, label=f'Regression Curve (K={optimal_K})', color='red')  # 회귀 곡선 그리기
plt.xlabel('Weight')  # x축 레이블
plt.ylabel('Length')  # y축 레이블
plt.title('Regression Curve with Optimal K')  # 그래프 제목
plt.legend()  # 범례 추가
plt.grid(True)  # 그리드 추가
plt.show()  # 그래프 출력

