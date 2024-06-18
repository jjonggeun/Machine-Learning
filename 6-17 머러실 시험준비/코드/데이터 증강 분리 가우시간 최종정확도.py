import numpy as np  # 수치 연산을 위한 라이브러리
import pandas as pd  # 데이터 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\logistic_regression_data.csv"  # 데이터 파일 경로
temp_data = pd.read_csv(fold_dir)  # CSV 파일을 읽어 데이터프레임으로 저장
temp_data = temp_data.to_numpy()  # 데이터프레임을 numpy 배열로 변환

# 데이터 분리
x1 = temp_data[:, 1].reshape(-1,1)  # temp_data 배열의 두 번째 열, 무게 데이터를 저장
x2 = temp_data[:, 2].reshape(-1,1)  # temp_data 배열의 세 번째 열, 길이 데이터를 저장
y = temp_data[:, 3].reshape(-1,1)   # temp_data 배열의 네 번째 열, 타겟 데이터를 저장

# 더미 데이터 추가
total_data = np.hstack((x1, x2, y))  # 무게, 길이, 타겟 데이터를 합침
new_x1 = total_data[:,0].reshape(-1,1)  # 합쳐진 데이터에서 첫 번째 열, 무게 데이터를 저장
new_x2 = total_data[:,1].reshape(-1,1)  # 합쳐진 데이터에서 두 번째 열, 길이 데이터를 저장
new_y = total_data[:,2].reshape(-1,1)  # 합쳐진 데이터에서 세 번째 열, 타겟 데이터를 저장

dummy_data = np.ones((len(total_data), 1))  # 더미 데이터를 생성, 값은 모두 1
x_with_dummy = np.hstack((new_x1, new_x2, dummy_data))  # 무게, 길이, 더미 데이터를 합쳐 최종 입력 데이터로 설정

# 시그모이드 함수 정의
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 시그모이드 함수 구현

# 비용 함수 (Cee) 정의
def Cee(X, y, w):
    m = len(y)  # 샘플 수
    z = np.dot(X, w)  # 선형 결합
    p = Sigmoid(z)  # 시그모이드 함수 적용
    cost = (-1 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))  # 비용 함수 계산
    return cost  # 계산된 비용 반환

# 예측 함수
def predict(X, w):
    z = np.dot(X, w)  # 선형 결합
    p = Sigmoid(z)  # 시그모이드 함수 적용
    predictions = np.where(p >= 0.5, 1, 0)  # 예측값 0.5 이상이면 1, 아니면 0
    return predictions  # 예측 결과 반환

# 경사 하강법 함수
def gradient_descent(X, y, alpha, rp):
    w_ = np.random.rand(3, 1)  # 가중치 초기값을 랜덤으로 설정
    w0_history, w1_history, w2_history, cee_history, accuracy_history = [], [], [], [], []  # 가중치와 비용, 정확도 기록 리스트 초기화
    train_accuracy_history, test_accuracy_history = [], []  # 훈련 세트와 테스트 세트의 정확도 기록 리스트 초기화

    for i in range(rp):  # rp 횟수만큼 반복
        z = np.dot(X, w_)  # 선형 결합
        p = Sigmoid(z)  # 시그모이드 함수 적용
        dif_cee = np.mean((p - y) * X, axis=0).reshape(-1, 1)  # 비용 함수의 그래디언트 계산
        w_ -= alpha * dif_cee  # 가중치 업데이트

        w0_history.append(w_[0][0])  # 가중치 w0 기록
        w1_history.append(w_[1][0])  # 가중치 w1 기록
        w2_history.append(w_[2][0])  # 가중치 w2 기록
        
        cee = Cee(X, y, w_)  # 비용 함수 계산
        cee_history.append(cee)  # 비용 기록
        
        predictions = predict(X, w_)  # 현재 가중치로 예측
        accuracy = np.sum(predictions == y) / len(y)  # 예측 정확도 계산
        accuracy_history.append(accuracy)  # 정확도 기록

        # 훈련 세트에 대한 정확도 계산
        train_predictions = predict(train_x_with_dummy, w_)
        train_accuracy = np.sum(train_predictions == train_y) / len(train_y)
        train_accuracy_history.append(train_accuracy)
        
        # 테스트 데이터셋에 대한 정확도 계산
        test_predictions = predict(test_x_with_dummy, w_)
        test_accuracy = np.sum(test_predictions == test_y) / len(test_y)
        test_accuracy_history.append(test_accuracy)
    
    return w0_history, w1_history, w2_history, cee_history, accuracy_history, train_accuracy_history, test_accuracy_history

# 데이터 분할 함수
def aug_data(data, train_ratio, test_ratio):
    assert train_ratio + test_ratio == 1  # 훈련 비율과 테스트 비율의 합이 1인지 확인

    total_samples = len(total_data)  # 총 데이터 샘플 수
    train_size = int(total_samples * train_ratio)  # 훈련 세트 크기 계산

    np.random.shuffle(total_data)  # 데이터를 랜덤하게 섞음

    train_set = total_data[:train_size]  # 훈련 세트
    test_set = total_data[train_size:]  # 테스트 세트

    return train_set, test_set  # 훈련 세트와 테스트 세트 반환

# 데이터를 7:3 비율로 분할
train_set, test_set = aug_data(total_data, 0.7, 0.3)

# 훈련 세트에서 더미 변수를 포함한 입력 데이터와 타겟 데이터 생성
train_x_with_dummy = np.hstack((train_set[:, :2], np.ones((len(train_set), 1))))
train_y = train_set[:, 2].reshape(-1, 1)

# 테스트 세트에서 더미 변수를 포함한 입력 데이터와 타겟 데이터 생성
test_x_with_dummy = np.hstack((test_set[:, :2], np.ones((len(test_set), 1))))
test_y = test_set[:, 2].reshape(-1, 1)

# 경사 하강법을 통해 가중치와 비용, 정확도 기록
w0_history, w1_history, w2_history, cee_history, accuracy_history, train_accuracy_history, test_accuracy_history = gradient_descent(train_x_with_dummy, train_y, 0.3, 4000)

# 가중치 변화 그래프
plt.figure(figsize=(8, 6))
plt.plot(w0_history, label='w0')  # w0 가중치 변화
plt.plot(w1_history, label='w1')  # w1 가중치 변화
plt.plot(w2_history, label='w2')  # w2 가중치 변화
plt.xlabel('Iterations')  # x축 레이블
plt.ylabel('Weights')  # y축 레이블
plt.title('Training Data Change')  # 그래프 제목
plt.grid()  # 그리드 추가
plt.legend()  # 범례 추가
plt.show()  # 그래프 출력

# 비용 함수 변화 그래프
plt.figure(figsize=(8, 6))
plt.plot(cee_history)  # 비용 함수 변화
plt.xlabel('Iterations')  # x축 레이블
plt.ylabel('Cost')  # y축 레이블
plt.title('Training Data Cost Function')  # 그래프 제목
plt.grid()  # 그리드 추가
plt.show()  # 그래프 출력

# 분류 정확도 변화 그래프 (훈련 세트와 테스트 세트)
plt.figure(figsize=(8, 6))
plt.plot(train_accuracy_history, label='Train Accuracy')  # 훈련 세트 정확도 변화
plt.plot(test_accuracy_history, label='Test Accuracy')  # 테스트 세트 정확도 변화
plt.xlabel('Iterations')  # x축 레이블
plt.ylabel('Accuracy')  # y축 레이블
plt.title('Accuracy Train and Test')  # 그래프 제목
plt.legend()  # 범례 추가
plt.grid()  # 그리드 추가
plt.show()  # 그래프 출력

# 테스트 데이터셋에 대한 최종 정확도 출력
print("Final Test Accuracy:", test_accuracy_history[-1])  # 테스트 데이터 최종 정확도

# 결정 경계 계산
x_values = np.linspace(np.min(train_set[:, 0]), np.max(train_set[:, 0]), 100)  # x축 값 범위 설정
y_values = (-w0_history[-1] / w1_history[-1]) * x_values - (w2_history[-1] / w1_history[-1])  # 결정 경계 계산

# 훈련 세트에 대한 결정 경계 그래프
plt.figure(figsize=(8, 6))
plt.scatter(train_set[:, 0], train_set[:, 1], c=train_set[:, 2], cmap='viridis', marker='o', label='Train Data Points')  # 훈련 데이터 산점도
plt.plot(x_values, y_values, color='red', label='Decision Boundary')  # 결정 경계 그리기
plt.xlabel('Attendance rate[%]')  # x축 레이블
plt.ylabel('Exam score')  # y축 레이블
plt.title('Training Data Decision Boundary')  # 그래프 제목
plt.legend(loc='upper right', title='Legend', labels=['Decision Boundary', 'Class 0', 'Class 1'])  # 범례 추가
plt.grid()  # 그리드 추가
plt.show()  # 그래프 출력

# 테스트 세트에 대한 결정 경계 그래프
plt.figure(figsize=(8, 6))
plt.scatter(test_set[:, 0], test_set[:, 1], c=test_set[:, 2], cmap='viridis', marker='o', label='Test Data Points')  # 테스트 데이터 산점도
plt.plot(x_values, y_values, color='red', label='Decision Boundary')  # 결정 경계 그리기
plt.xlabel('Attendance rate[%]')  # x축 레이블
plt.ylabel('Exam score')  # y축 레이블
plt.title('Test Data Decision Boundary')  # 그래프 제목
plt.legend(loc='upper right', title='Legend', labels=['Decision Boundary', 'Class 0', 'Class 1'])  # 범례 추가
plt.grid()  # 그리드 추가
plt.show()  # 그래프 출력
# 변수 설명
# fold_dir: 데이터 파일 경로를 나타내는 문자열 변수.
# temp_data: CSV 파일에서 읽어온 데이터프레임을 numpy 배열로 변환한 변수.
# x1: temp_data 배열의 두 번째 열, 무게 데이터.
# x2: temp_data 배열의 세 번째 열, 길이 데이터.
# y: temp_data 배열의 네 번째 열, 타겟 데이터.
# total_data: x1, x2, y 데이터를 합친 numpy 배열.
# new_x1: total_data 배열의 첫 번째 열, 무게 데이터.
# new_x2: total_data 배열의 두 번째 열, 길이 데이터.
# new_y: total_data 배열의 세 번째 열, 타겟 데이터.
# dummy_data: 모든 값이 1인 더미 데이터 배열.
# x_with_dummy: new_x1, new_x2, dummy_data를 합친 최종 입력 데이터 배열.
# Sigmoid(x): 시그모이드 함수를 구현하는 함수.
# Cee(X, y, w): 비용 함수를 계산하는 함수.
# predict(X, w): 예측 값을 계산하는 함수.
# gradient_descent(X, y, alpha, rp): 경사 하강법을 구현하는 함수.
# w_: 초기 가중치.
# w0_history, w1_history, w2_history: 가중치 변화 기록 리스트.
# cee_history: 비용 함수 값 기록 리스트.
# accuracy_history: 전체 데이터 정확도 기록 리스트.
# train_accuracy_history: 훈련 세트 정확도 기록 리스트.
# test_accuracy_history: 테스트 세트 정확도 기록 리스트.
# aug_data(data, train_ratio, test_ratio): 데이터를 훈련 세트와 테스트 세트로 분할하는 함수.
# train_ratio: 훈련 데이터 비율.
# test_ratio: 테스트 데이터 비율.
# train_set: 훈련 세트.
# test_set: 테스트 세트.
# train_x_with_dummy: 훈련 세트 입력 데이터에 더미 변수를 추가한 배열.
# train_y: 훈련 세트 타겟 데이터.
# test_x_with_dummy: 테스트 세트 입력 데이터에 더미 변수를 추가한 배열.
# test_y: 테스트 세트 타겟 데이터.
# w0_history, w1_history, w2_history: 경사 하강법으로 계산된 가중치 변화 기록.
# cee_history: 경사 하강법으로 계산된 비용 함수 값 기록.
# accuracy_history: 경사 하강법으로 계산된 전체 데이터 정확도 기록.
# train_accuracy_history: 경사 하강법으로 계산된 훈련 세트 정확도 기록.
# test_accuracy_history: 경사 하강법으로 계산된 테스트 세트 정확도 기록.
# x_values: 결정 경계를 그릴 때 사용할 x 값 범위.
# y_values: 결정 경계를 계산한 y 값 범위.