import numpy as np
import pandas as pd

# 예시 사용 코드
directory = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\"
path = directory + "heart_disease_new.csv"
df = pd.read_csv(path)  # pandas를 사용하여 CSV 파일 읽기
dataset = np.array(df)  # pandas DataFrame을 numpy 배열로 변환
 # 데이터 섞기
np.random.shuffle(dataset)
 
    #================== 전처리====================
    
    #path데이터(heart_disease_new.csv)에서 ,를 사용해 구분하고 문자열로 1행 분리
column_names = np.genfromtxt(path, delimiter=',', dtype=str, max_rows=1)
      
    # 열 이름을 기준으로 인덱스 찾아서 저장
    # column_names는 데이터의 column_names를 받아 파일 첫 열을 추출함 따라서 0~14까지있고, 아래 코드는 그 위치가 어딘지 찾아줌   
gender_idx = np.where(column_names == "gender")[0][0]
neuro_idx = np.where(column_names == "a neurological disorder")[0][0]
target_idx = np.where(column_names == "heart disease")[0][0]
age_idx = np.where(column_names == "Age")[0][0]
hbp_idx = np.where(column_names == "High blood pressure")[0][0]
chol_idx = np.where(column_names == "Cholesterol")[0][0]
target_idx = np.where(column_names == "heart disease")[0][0]
height_idx = np.where(column_names == "height")[0][0]
weight_idx = np.where(column_names == "Weight")[0][0]
bmi_idx = np.where(column_names == "BMI")[0][0]
    
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
        
    # BMI 값 계산 및 업데이트 (Weight / (Height in meters)^2)
weight = dataset[:, weight_idx].astype(float)
height = dataset[:, height_idx].astype(float) / 100  # cm를 m로 변환
bmi = weight / (height ** 2)
dataset[:, bmi_idx] = bmi
    
    # 데이터 형식을 float로 변환
dataset = dataset.astype(float)
    #===========================================================#
    # y_data 추출
y_data = dataset[:, -1]
    
    ## 특징 추출: 40세 이상, 고혈압이 있으며, 콜레스테롤이 250 이상인 경우
    # 특징 추출 (예시로 열 0, 1, 2을 선택한다고 가정)
feature_1 = dataset[:, 0] 
feature_2 = dataset[:, 1]
feature_n = dataset[:, 2]
    
    # 선택한 특징들을 결합
selected_features = np.column_stack((feature_1, feature_2, feature_n))
    
    # 특징 데이터 마지막 열에 y값 추가
features = np.column_stack((selected_features, y_data))
    



