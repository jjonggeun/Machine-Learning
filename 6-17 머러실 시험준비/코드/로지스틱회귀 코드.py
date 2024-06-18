import numpy as np  # 수치 연산을 위한 라이브러리
import pandas as pd  # 데이터 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\logistic_regression_data.csv"  # 데이터 파일 경로
temp_data = pd.read_csv(fold_dir)  # CSV 파일을 읽어 데이터프레임으로 저장
temp_data = temp_data.to_numpy()  # 데이터프레임을 numpy 배열로 변환

# 시그모이드 함수 정의
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 시그모이드 함수 구현

# 손실 함수 (비용 함수) 계산
def compute_cost(X, y, w):
    m = len(y)  # 샘플 수
    z = np.dot(X, w)  # 선형 결합
    p = Sigmoid(z)  # 시그모이드 함수 적용
    cost = (-1 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))  # 비용 함수 계산
    return cost  # 계산된 비용 반환

# 예측 함수
def predict(X, w):
    z = np.dot(X, w)  # 선형 결합
    p = Sigmoid(z)  # 시그모이드 함수 적용
    predictions = np.where(p >= 0.5, 1, 0)  # 0.5 이상이면 1, 그렇지 않으면 0으로 예측
    return predictions  # 예측 결과 반환

# 경사 하강법 함수
def gradient_descent(X, y, alpha, rp):
    w_ = np.random.rand(3, 1)  # 초기 가중치 랜덤 설정
    w0_history, w1_history, w2_history, cee_history, accuracy_history = [], [], [], [], []  # 가중치와 비용, 정확도 히스토리 저장 리스트

    for i in range(rp):  # rp 횟수만큼 반복
        z = np.dot(X, w_)  # 선형 결합
        p = Sigmoid(z)  # 시그모이드 함수 적용
        dif_cee = np.mean((p - y) * X, axis=0).reshape(-1, 1)  # 비용 함수의 그래디언트 계산
        w_ -= alpha * dif_cee  # 가중치 업데이트

        # 히스토리 저장
        w0_history.append(w_[0][0])
        w1_history.append(w_[1][0])
        w2_history.append(w_[2][0])
        
        cee = compute_cost(X, y, w_)  # 비용 함수 계산
        cee_history.append(cee)  # 비용 히스토리 저장
        
        predictions = predict(X, w_)  # 현재 가중치로 예측
        accuracy = np.sum(predictions == y) / len(y)  # 예측 정확도 계산
        accuracy_history.append(accuracy)  # 정확도 히스토리 저장
    
    return w0_history, w1_history, w2_history, cee_history, accuracy_history  # 가중치와 비용, 정확도 히스토리 반환

# 데이터 분할 함수
def aug_data(augmented_data, train_ratio, test_ratio):
    assert train_ratio + test_ratio == 1  # 훈련 비율과 테스트 비율의 합이 1인지 확인

    total_samples = len(augmented_data)  # 총 데이터 샘플 수
    train_size = int(total_samples * train_ratio)  # 훈련 세트 크기 계산

    np.random.shuffle(augmented_data)  # 데이터를 랜덤하게 섞음

    train_set = augmented_data[:train_size]  # 훈련 세트
    test_set = augmented_data[train_size:]  # 테스트 세트

    return train_set, test_set  # 훈련 세트와 테스트 세트 반환

# 데이터를 7:3 비율로 분할
train_set, test_set = aug_data(temp_data, 0.7, 0.3)  # 70% 훈련, 30% 테스트로 데이터 분할

# 더미 변수 추가 및 훈련 데이터 준비
x1_train = train_set[:, 1].reshape(-1, 1)  # 무게 데이터
x2_train = train_set[:, 2].reshape(-1, 1)  # 길이 데이터
y_train = train_set[:, 3].reshape(-1, 1)  # 타겟 값
x_with_dummy_train = np.hstack((x1_train, x2_train, np.ones((len(x1_train), 1))))  # 더미 변수 추가된 훈련 데이터

# 더미 변수 추가 및 테스트 데이터 준비
x1_test = test_set[:, 1].reshape(-1, 1)  # 무게 데이터
x2_test = test_set[:, 2].reshape(-1, 1)  # 길이 데이터
y_test = test_set[:, 3].reshape(-1, 1)  # 타겟 값
x_with_dummy_test = np.hstack((x1_test, x2_test, np.ones((len(x1_test), 1))))  # 더미 변수 추가된 테스트 데이터

# 경사 하강법을 통해 학습
w0_history, w1_history, w2_history, _, _ = gradient_descent(x_with_dummy_train, y_train, 0.3, 4000)  # 학습 수행

# 가중치 업데이트에 따른 예측 정확도 계산
accuracy_history = []
for i in range(len(w0_history)):
    w = np.array([[w0_history[i]], [w1_history[i]], [w2_history[i]]])  # 각 반복마다의 가중치 설정
    predictions = np.where(Sigmoid(np.dot(x_with_dummy_test, w)) >= 0.5, 1, 0)  # 예측
    accuracy = np.sum(predictions == y_test) / len(test_set)  # 정확도 계산
    accuracy_history.append(accuracy)  # 정확도 히스토리 저장

# 그래프 그리기
plt.figure(figsize=(12, 6))  # 그래프 크기 설정

# 가중치 변화 그래프
plt.plot(w0_history, label='w0')  # w0 가중치 변화
plt.plot(w1_history, label='w1')  # w1 가중치 변화
plt.plot(w2_history, label='w2')  # w2 가중치 변화
plt.xlabel('Iterations')  # x축 레이블
plt.ylabel('Weights')  # y축 레이블
plt.title('Changes in Weights over Iterations')  # 그래프 제목
plt.legend()  # 범례 추가
plt.show()  # 그래프 출력

# 분류 정확도 변화 그래프
plt.figure(figsize=(12, 6))  # 그래프 크기 설정
plt.plot(accuracy_history)  # 정확도 변화
plt.xlabel('Iterations')  # x축 레이블
plt.ylabel('Accuracy')  # y축 레이블
plt.title('Accuracy over Iterations')  # 그래프 제목
plt.show()  # 그래프 출력
