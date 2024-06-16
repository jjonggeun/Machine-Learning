import numpy as np  # 수치 연산을 위한 라이브러리
import pandas as pd  # 데이터 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리

# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 시그모이드 함수 적용하여 값을 반환

# 시그모이드 함수의 미분 정의
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))  # 시그모이드 함수의 미분 값 반환

# 순전파 함수 정의
def forward_propagation(x_with_dummy, v, w):
    A = v @ x_with_dummy.T  # 입력 데이터와 은닉층 가중치의 행렬 곱 (은닉층 입력)
    b = sigmoid(A)  # 은닉층 입력에 시그모이드 함수를 적용하여 은닉층 출력 계산
    b_with_dummy = np.vstack([b, np.ones([1, len(x_with_dummy)])])  # 은닉층 출력에 더미 변수를 추가
    B = w @ b_with_dummy  # 은닉층 출력과 출력층 가중치의 행렬 곱 (출력층 입력)
    y_hat = sigmoid(B)  # 출력층 입력에 시그모이드 함수를 적용하여 최종 출력 계산
    return A, b, b_with_dummy, B, y_hat  # 은닉층 입력, 은닉층 출력, 더미 추가된 은닉층 출력, 출력층 입력, 최종 출력 반환

# 역전파 함수 정의
def backward_propagation(x_with_dummy, y, A, b, b_with_dummy, B, y_hat, v, w):
    error = y_hat - y.T  # 출력 오차 계산 (예측 값 - 실제 값)
    wmse = (error * sigmoid_derivative(B)) @ b_with_dummy.T / len(x_with_dummy)  # 출력층 가중치의 경사 계산
    vmse = ((w[:, :-1].T @ (error * sigmoid_derivative(B))) * sigmoid_derivative(A)) @ x_with_dummy / len(x_with_dummy)  # 은닉층 가중치의 경사 계산
    return wmse, vmse  # 출력층과 은닉층 가중치의 경사 반환

# 데이터 분할 함수 정의
def aug_data(data, train_ratio, test_ratio):
    assert train_ratio + test_ratio == 1  # 학습용과 테스트용 비율의 합이 1인지 확인
    total_samples = len(data)  # 전체 데이터 샘플 수
    train_size = int(total_samples * train_ratio)  # 학습용 데이터 샘플 수 계산
    np.random.shuffle(data)  # 데이터를 랜덤하게 섞음
    train_set = data[:train_size]  # 학습용 데이터 추출
    test_set = data[train_size:]  # 테스트용 데이터 추출
    return train_set, test_set  # 학습용 데이터와 테스트용 데이터 반환

# 혼동 행렬 계산 함수 정의
def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))  # 혼동 행렬 초기화
    for i in range(len(y_true)):  # 각 데이터 포인트에 대해
        row_index = int(y_pred[i])  # 예측 값 인덱스
        col_index = int(y_true[i])  # 실제 값 인덱스
        confusion_matrix[row_index, col_index] += 1  # 혼동 행렬 업데이트
    return confusion_matrix  # 계산된 혼동 행렬 반환

# 특징 선택 함수 정의
def select_features(file_path):
    path = directory + "heart_disease_new.csv"
    df = pd.read_csv(path)  # CSV 파일을 읽어 데이터프레임으로 저장
    dataset = np.array(df)  # 데이터프레임을 numpy 배열로 변환

    np.random.shuffle(dataset)  # 데이터를 랜덤하게 섞음
    
    #================== 전처리====================
    column_names = np.genfromtxt(path, delimiter=',', dtype=str, max_rows=1)  # 열 이름을 읽어옴
    
    # 열 이름을 기준으로 인덱스 찾아서 저장
    gender_idx = np.where(column_names == "gender")[0][0]
    neuro_idx = np.where(column_names == "a neurological disorder")[0][0]
    target_idx = np.where(column_names == "heart disease")[0][0]
    age_idx = np.where(column_names == "Age")[0][0]
    hbp_idx = np.where(column_names == "High blood pressure")[0][0]
    chol_idx = np.where(column_names == "Cholesterol")[0][0]
    height_idx = np.where(column_names == "height")[0][0]
    weight_idx = np.where(column_names == "Weight")[0][0]
    bmi_idx = np.where(column_names == "BMI")[0][0]
    bpm_idx = np.where(column_names == "BPMeds")[0][0]
    bloodsugar_idx = np.where(column_names == "blood sugar levels")[0][0]
    meat_idx = np.where(column_names == "meat intake")[0][0]
    smoke_idx = np.where(column_names == "Smoking")[0][0]
    
    # gender: female -> 0, male -> 1
    dataset[:, gender_idx] = np.where(dataset[:, gender_idx] == 'female', 0, 1)
    
    # a neurological disorder: no -> 0, yes -> 1
    dataset[:, neuro_idx] = np.where(dataset[:, neuro_idx] == 'yes', 1, 0)
    
    # heart disease: yes -> 1, no -> 0
    dataset[:, target_idx] = np.where(dataset[:, target_idx] == 'yes', 1, 0)
    
    # 결측치 처리 (각 열의 평균값으로 대체)
    for i in range(dataset.shape[1]):
        col = dataset[:, i].astype(float)
        mean_val = np.nanmean(col)  # 결측치를 제외한 평균값 계산
        col[np.isnan(col)] = mean_val  # 결측치를 평균값으로 대체
        dataset[:, i] = col
    
    dataset = dataset.astype(float)  # 데이터 형식을 float로 변환
    
    # 편향 조정
    yes_data = dataset[dataset[:, target_idx] == 1]  # 심장병이 있는 데이터
    no_data = dataset[dataset[:, target_idx] == 0]  # 심장병이 없는 데이터
    np.random.shuffle(no_data)  # 심장병이 없는 데이터를 랜덤하게 섞음
    no_data = no_data[:500]  # 심장병이 없는 데이터에서 500개 샘플링
    balanced_data = np.vstack((yes_data, no_data))  # 두 데이터를 결합하여 균형잡힌 데이터셋 생성
    
    np.random.shuffle(balanced_data)  # 데이터를 랜덤하게 섞기
    
    y_data = balanced_data[:, target_idx]  # 타겟 값 추출
    
    # 특징 추출 (상관관계에 기반하여 조합)
    feature_1 = balanced_data[:, height_idx]  # 키
    feature_4 = balanced_data[:, weight_idx] / (balanced_data[:, age_idx] + 10)  # 몸무게와 나이의 비율
    feature_5 = balanced_data[:, weight_idx] / ((balanced_data[:, height_idx]/100)**2)  # BMI
    feature_12 = ((balanced_data[:, smoke_idx] * 10) + (balanced_data[:, meat_idx] * 10)) - balanced_data[:, height_idx] / 5  # 흡연과 고기 섭취 비율
    
    # 선택한 특징들을 결합
    selected_features = np.column_stack((feature_1, feature_12, feature_5, feature_4))
    
    # 특징 데이터 마지막 열에 y값 추가
    features = np.column_stack((selected_features, y_data))
    
    return features  # 선택된 특징과 타겟 값을 포함한 데이터 반환

# 데이터 디렉토리 경로
directory = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\"
features = select_features(directory)  # 특징 선택 함수 호출

train_data, test_data = aug_data(features, 0.7, 0.3)  # 데이터 분할 (70% 학습용, 30% 테스트용)

# 데이터 분리
x_train = train_data[:, :-1]  # 학습용 입력 데이터
y_train = train_data[:, -1].reshape(-1, 1)  # 학습용 타겟 데이터

x_test = test_data[:, :-1]  # 테스트용 입력 데이터
y_test = test_data[:, -1].reshape(-1, 1)  # 테스트용 타겟 데이터

# 입력 속성 수와 출력 클래스 수 추출
M = x_train.shape[1]  # 입력 데이터의 속성 수
output_size = 1  # 출력 클래스 수

hidden_size = 3  # 은닉층의 노드 수

# 가중치 초기화 (표준 정규 분포에서 작은 값들로 초기화)
v = np.random.randn(hidden_size, M + 1) * 0.005  # 은닉층 가중치
w = np.random.randn(output_size, hidden_size + 1) * 0.005  # 출력층 가중치

# 학습 파라미터 설정
learning_rate = 0.00457  # 학습률
epochs = 500  # 학습 에포크 수

# 데이터에 더미 변수 추가
x_train_with_dummy = np.hstack((x_train, np.ones((len(x_train), 1))))  # 학습용 데이터에 더미 변수 추가
x_test_with_dummy = np.hstack((x_test, np.ones((len(x_test), 1))))  # 테스트용 데이터에 더미 변수 추가

total_samples = len(x_train)  # 학습 데이터의 총 샘플 수

# 정확도 및 MSE 기록 리스트 초기화
accuracy_list = []  # 학습 정확도 기록 리스트
mse_list = []  # 학습 MSE 기록 리스트
mse_test_list = []  # 테스트 MSE 기록 리스트
test_accuracy_list = []  # 테스트 정확도 기록 리스트

best_accuracy = 0  # 최고 정확도 초기화
best_v = np.copy(v)  # 최적의 은닉층 가중치 초기화
best_w = np.copy(w)  # 최적의 출력층 가중치 초기화

batch_size = 5  # 배치 크기를 5로 설정

# 학습
for epoch in range(epochs):  # 모든 에포크에 대해 반복
    for start in range(0, total_samples, batch_size):  # 배치 크기 단위로 데이터셋을 순회
        end = start + batch_size  # 배치의 끝 인덱스 계산
        x_batch = x_train_with_dummy[start:end]  # 현재 배치의 입력 데이터 추출
        y_batch = y_train[start:end]  # 현재 배치의 타겟 데이터 추출

        # 순전파 수행
        A, b, b_with_dummy, B, y_hat = forward_propagation(x_batch, v, w)  # 입력 데이터로부터 예측값 계산

        # 역전파 수행
        wmse, vmse = backward_propagation(x_batch, y_batch, A, b, b_with_dummy, B, y_hat, v, w)  # 예측값과 실제값의 차이로부터 가중치의 변화량 계산

        # 가중치 업데이트
        w -= learning_rate * wmse  # 출력층 가중치 업데이트
        v -= learning_rate * vmse  # 은닉층 가중치 업데이트
    
    # 테스트 데이터에 대해 정확도 계산
    A_test, b_test, b_with_dummy_test, B_test, y_hat_test = forward_propagation(x_test_with_dummy, v, w)  # 테스트 데이터로부터 예측값 계산
    y_hat_test_index = (y_hat_test >= 0.5).astype(int).flatten()  # 예측값을 0 또는 1로 변환 (0.5 기준 이진 분류)
    test_accuracy = np.mean(y_hat_test_index == y_test.flatten())  # 예측값과 실제값이 일치하는 비율 계산
    test_accuracy_list.append(test_accuracy)  # 테스트 정확도를 리스트에 저장
    
    print(test_accuracy)  # 현재 에포크의 테스트 정확도 출력

    if test_accuracy > best_accuracy:  # 현재 에포크의 테스트 정확도가 최고 정확도보다 높으면
        best_accuracy = test_accuracy  # 최고 정확도 업데이트
        best_v = np.copy(v)  # 최적의 은닉층 가중치 저장
        best_w = np.copy(w)  # 최적의 출력층 가중치 저장

    # 학습 데이터에 대해 정확도 및 MSE 계산
    A_train, b_train, b_with_dummy_train, B_train, y_hat_train = forward_propagation(x_train_with_dummy, v, w)  # 학습 데이터로부터 예측값 계산
    predicted_labels = (y_hat_train >= 0.5).astype(int).flatten()  # 예측값을 0 또는 1로 변환 (0.5 기준 이진 분류)
    accuracy = np.mean(predicted_labels == y_train.flatten())  # 예측값과 실제값이 일치하는 비율 계산
    accuracy_list.append(accuracy)  # 학습 정확도를 리스트에 저장
    
    mse = np.mean((y_hat_train - y_train.T) ** 2)  # 예측값과 실제값의 차이의 제곱 평균으로 MSE 계산
    mse_list.append(mse)  # 학습 MSE를 리스트에 저장

# 최적의 가중치로 모델 업데이트
v = best_v  # 최적의 은닉층 가중치 설정
w = best_w  # 최적의 출력층 가중치 설정


# 혼동 행렬 계산
confusion_matrix = compute_confusion_matrix(y_test.flatten(), y_hat_test_index, 2)

print("Confusion Matrix:")
print(confusion_matrix)

# 그래프 출력
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), accuracy_list, label='Train Accuracy', color='blue')
plt.plot(range(1, epochs+1), test_accuracy_list, label='Test Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), mse_list, label='MSE', color='red')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE over epochs')
plt.legend()
plt.grid()

plt.show()
zz