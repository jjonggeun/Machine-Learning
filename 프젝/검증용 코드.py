import numpy as np
import pandas as pd

def select_features(file_path):
    df = pd.read_csv(file_path)  # pandas를 사용하여 CSV 파일 읽기
    dataset = np.array(df)  # pandas DataFrame을 numpy 배열로 변환

    # 데이터 섞기
    np.random.shuffle(dataset)
    
    #================== 전처리====================
    
    # 파일 경로 데이터에서 ,를 사용해 구분하고 문자열로 1행 분리
    column_names = np.genfromtxt(file_path, delimiter=',', dtype=str, max_rows=1)
    
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
        col = np.where(np.isnan(col), mean_val, col)  # 결측치를 평균값으로 대체
        dataset[:, i] = col
    
    # 데이터 형식을 float로 변환
    dataset = dataset.astype(float)
    
    # y_data 추출
    y_data = dataset[:, target_idx]  # 타겟 값 추출
    
    # 특징 추출 (상관관계에 기반하여 조합)
    feature_1 = dataset[:, height_idx]
    feature_5 = dataset[:, weight_idx] / ((dataset[:, height_idx]/100)**2) #내가 구한 bmi
    feature_12 = ((dataset[:,smoke_idx]*10) + (dataset[:,meat_idx] * 10)) - dataset[:,height_idx] / 5
    
    # 선택한 특징들을 결합
    selected_features = np.column_stack((feature_1, feature_5, feature_12))
    
    # 특징 데이터 마지막 열에 y값 추가
    features = np.column_stack((selected_features, y_data))
    
    return features

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # sigmoid 계산해 반환

def aug_data(data, train_ratio, test_ratio):
    assert train_ratio + test_ratio == 1
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    np.random.shuffle(data)
    train_set = data[:train_size]
    test_set = data[train_size:]
    return train_set, test_set

def forward_propagation(x_with_dummy, v, w):
    A = v @ x_with_dummy.T
    b = sigmoid(A)
    b_with_dummy = np.vstack([b, np.ones([1, len(x_with_dummy)])])
    B = w @ b_with_dummy
    y_hat = sigmoid(B)
    return A, b, b_with_dummy, B, y_hat

def compute_confusion_matrix_with_probabilities(y_true, y_pred, num_classes):
    #각 정확도 값을 추가하기 위해 행렬 크기 변경
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1))
    #기존 confusion_matrix로 예측과 실제값으로 
    for i in range(len(y_true)):
        row_index = int(y_pred[i])
        col_index = int(y_true[i])
        confusion_matrix[row_index, col_index] += 1
        #정확도 class를 만들어 이 값을 행열에 대입
    class_accuracies = []
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        total = np.sum(confusion_matrix[i, :num_classes]) + np.sum(confusion_matrix[:num_classes, i]) - TP
        accuracy = TP / total if total != 0 else 0
        class_accuracies.append(accuracy)
        confusion_matrix[i, num_classes] = accuracy  # Add row accuracy
        confusion_matrix[num_classes, i] = accuracy  # Add column accuracy

    total_correct = np.trace(confusion_matrix[:num_classes, :num_classes])
    total_predictions = np.sum(confusion_matrix[:num_classes, :num_classes])
    overall_accuracy = total_correct / total_predictions if total_predictions != 0 else 0
    confusion_matrix[num_classes, num_classes] = overall_accuracy  # Add overall accuracy
    #대각선 전체 정확도 계산해서 추가
    
    #확률 표현된 것을 추가한다
    prob_matrix = confusion_matrix[:num_classes, :num_classes] / total_predictions

    return confusion_matrix, prob_matrix

#원핫 함수
def One_Hot(data):
    num_classes = len(np.unique(data[:, -1]))
    one_hot_encoded = np.eye(num_classes)[data[:, -1].astype(int)]
    return one_hot_encoded
#정확도 계산
def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy
#mse계산
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
#순전파만 진행한 신경망
def Neural_Network(data, w_hidden, w_output):
    x_with_dummy = np.hstack([data[:, :-1], np.ones((data.shape[0], 1))])
    _, _, _, _, y_hat = forward_propagation(x_with_dummy, w_hidden, w_output)
    y_pred = np.round(y_hat).astype(int)
    return y_hat, y_pred

# 데이터 불러와서 저장
file_path = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\heart_disease_new.csv"
data = select_features(file_path)

np.random.shuffle(data)

yes_index = np.where(data[:, -1] == 1)[0]
yes_ = data[yes_index, :]  # yes 데이터 추출

no_data_index = np.where(data[:, -1] == 0)[0]
no_data = data[no_data_index, :]   # no 데이터 추출

yes = 200  # yes와 no의 개수를 결정해서 뽑아온다
no = 200
data_set = np.vstack([yes_[:yes, :], no_data[:no, :]])  # 뽑아온 yes와 no를 합쳐준다.

train_set, test_set = aug_data(data_set, 0.8, 0.2)  # 데이터 분할

One_hot_Test = One_Hot(test_set)  # Test 데이터에 대한 One_hot 생성

# 가중치 불러오기
best_v = pd.read_csv('C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\프젝\\weight\\w_hidden0.6 - 1.csv', header=None).to_numpy()
best_w = pd.read_csv('C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\프젝\\weight\\w_output0.6 - 1.csv', header=None).to_numpy()
w_hidden, w_output = best_v, best_w

# Test Set 순전파 진행해서 y_hat들 받아오기
y_hat_Test, Y_hat_Test = Neural_Network(test_set, w_hidden, w_output)

# Test 정확도 저장
Test_Acc = accuracy(test_set[:, -1].reshape(-1, 1), Y_hat_Test)

# Test 평균제곱오차 저장
MSE_Test = mse(test_set[:, -1].reshape(-1, 1), y_hat_Test)

# Confusion Matrix 생성
Confusion_Matrix_Test, Prob_Matrix_Test = compute_confusion_matrix_with_probabilities(test_set[:, -1], Y_hat_Test, num_classes=2)

print("Test Accuracy:", Test_Acc)
print("Test MSE:", MSE_Test)
print("Confusion Matrix (Counts):")
print(Confusion_Matrix_Test)
print("Confusion Matrix (Probabilities):")
print(Prob_Matrix_Test)
