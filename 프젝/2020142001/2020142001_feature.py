import pandas as pd
import numpy as np

def select_features(file_path):
    path = directory + "heart_disease_new.csv"
    df = pd.read_csv(path)  # pandas를 사용하여 CSV 파일 읽기
    dataset = np.array(df)  # pandas DataFrame을 numpy 배열로 변환

    # 데이터 섞기
    np.random.shuffle(dataset)
    
    #================== 전처리====================
    
    #path데이터(heart_disease_new.csv)에서 ,를 사용해 구분하고 문자열로 1행 분리
    column_names = np.genfromtxt(path, delimiter=',', dtype=str, max_rows=1)
    
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
    
    # 데이터 형식을 float로 변환
    dataset = dataset.astype(float)
    
    #===========================================================#
    # y_data 추출
    y_data = dataset[:, -1]  # 타겟 값 추출
    
    
    # 특징 추출 (상관관계에 기반하여 조합)
    # 다양한 특징을 시도하였고 최종적으로 1, 5, 12를 사용
    # 키가 상관관계가 매우 커서 사용
    feature_1 = dataset[:, height_idx]
    # 특징 5번 이 데이터에서 구해져있는 bmi는 값이 이상한 것들이 많아 따로 bmi를 구해보았다.
    feature_2 = dataset[:, weight_idx] / ((dataset[:, height_idx]/100)**2) 
    # 특징 12번 데이터에서 흡연이 yes이면 무조건 고기를 섭취하므로 두 값을 더하고, 큰 상관관계인 키/5를 빼주어 특징을 구현
    feature_3 = ((dataset[:,smoke_idx]*10) + (dataset[:,meat_idx] * 10)) - dataset[:,height_idx] / 5
    
    # 선택한 특징들을 결합
    selected_features = np.column_stack((feature_1, feature_2, feature_3))

    
    # 특징 데이터 마지막 열에 y값 추가
    features = np.column_stack((selected_features, y_data))
    
    return features